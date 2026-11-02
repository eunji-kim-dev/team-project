"""TensorRT PTQ 캘리브레이션 캐시의 활성화 scale 을 QuantConv2d 에 가져오기 (qat_train --trt-calib, eval.quant_sensitivity 가 씀)

python -m deploy.trt_scales --calib models/fast_scnn_int8_entropy.calib --fp32-onnx models/fast_scnn.onnx \\
    --ckpt checkpoints/best_mIoU_iter_1600.pth        # 대응표 출력: QuantConv2d ↔ ONNX conv 입력 텐서 ↔ scale

왜: EMA/max 캘리브레이션은 꼬리값 때문에 LTD.dsconv1 같은 층의 scale 이 TensorRT 엔트로피 scale 의 2배가 되어 INT8 손상이
    남는다 (docs/quant_sensitivity.md). PTQ 엔진이 고른 scale 을 그대로 Q/DQ 에 넣으면 PTQ 와 같은 정확도에서 QAT 를 시작함
캐시 형식 (deploy.build_engine --precision int8 이 저장. 예: models/fast_scnn_int8_entropy.calib)
    1줄 헤더 TRT-<버전>-EntropyCalibration2, 이후 "<ONNX 텐서 이름>: <float32 hex (big-endian)>" = 그 텐서의 scale (|x|max/127)
대응: 캐시를 만들 때 쓴 FP32 ONNX 의 Conv 노드 → 입력 텐서(node.input[0]) 가 캐시 항목, 노드 이름 → QuantConv2d 모듈 이름
    (torch export 가 Sequential 자식을 '부모/부모.번호' 로 적는 것을 되돌림)
확인: QuantConv2d 마다 Conv 노드가 정확히 하나, 그 노드의 가중치 모양 == 모듈 conv.weight 모양, 입력 텐서가 캐시에 있음.
    하나라도 어긋나면 오류 (조용히 EMA scale 로 남겨 두지 않음). 캐시의 나머지 항목은 conv 입력이 아닌 텐서(사진 입력, add·pool 출력)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy import quant  # noqa: E402

SHORT_PREFIXES = (("core.backbone.learning_to_downsample.", "LTD."),
                  ("core.backbone.global_feature_extractor.", "GFE."),
                  ("core.backbone.feature_fusion.", "FFM."),
                  ("core.decode_head.", "HEAD."))


# ---------- 이름 ----------

def short_name(conv_name: str) -> str:
    """QuantConv2d 모듈 이름 → 표·옵션에 쓰는 짧은 이름
    core.backbone.learning_to_downsample.dsconv1.depthwise_conv.conv → LTD.dsconv1.depthwise_conv"""
    name = conv_name
    for prefix, short in SHORT_PREFIXES:
        if name.startswith(prefix):
            name = short + name[len(prefix):]
            break
    if name.endswith(".conv"):
        name = name[:-5]
    return name


def node_to_module(node_name: str) -> str:
    """ONNX 노드 이름 → 모듈 이름. torch export 는 Sequential 자식을 '부모/부모.번호' 로 적으므로 앞부분을 떼어 냄
    /core/backbone/.../bottleneck1/bottleneck1.0/conv/conv.0/conv/Conv → core.backbone....bottleneck1.0.conv.0.conv"""
    raw = node_name.strip("/").split("/")[:-1]
    parts = []
    for prev, segment in zip([None] + raw[:-1], raw):
        if prev is not None and segment.startswith(prev + "."):
            segment = segment[len(prev) + 1:]
        parts.append(segment)
    return ".".join(parts)


def select_convs(names: list[str], wanted: list[str] | None) -> list[str]:
    """wanted: 비었거나 'all' 이면 전부. 아니면 모듈 이름 또는 짧은 이름(LTD, LTD.dsconv1, HEAD.convs.0.pointwise_conv …)의 접두사"""
    if not wanted or "all" in wanted:
        return list(names)

    def hit(name, token):
        s = short_name(name)
        return name == token or name.startswith(token + ".") or s == token or s.startswith(token + ".")

    chosen = [name for name in names if any(hit(name, token) for token in wanted)]
    unknown = [token for token in wanted if not any(hit(name, token) for name in names)]
    if unknown:
        raise ValueError(f"QuantConv2d 이름과 안 맞는 지정: {unknown}. 예: {', '.join(short_name(n) for n in names[:3])} …")
    return chosen


# ---------- 캐시 ----------

def read_calib_cache(path: Path) -> tuple[str, dict[str, float]]:
    """(헤더, {텐서 이름: scale})"""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0].strip() if lines else ""
    if not header.startswith("TRT-"):
        raise ValueError(f"{path}: TensorRT 캘리브레이션 캐시가 아님 (첫 줄 {header!r})")
    entries = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, hexval = line.rsplit(":", 1)
        entries[name.strip()] = struct.unpack("!f", bytes.fromhex(hexval.strip()))[0]
    return header, entries


def cache_provenance(calib_path: Path, header: str, n_entries: int) -> dict:
    """캐시 출처: 경로·헤더·해시·수정 시각 + build_engine 이 옆에 남긴 <stem>_int8.engine.json 의 캘리브레이션 사진 정보"""
    info = {"path": str(calib_path), "header": header, "entries": n_entries,
            "sha256_16": hashlib.sha256(calib_path.read_bytes()).hexdigest()[:16],
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(calib_path.stat().st_mtime))}
    engine_json = calib_path.with_name(f"{calib_path.name.split('_int8_')[0]}_int8.engine.json")
    if engine_json.exists():
        manifest = json.loads(engine_json.read_text(encoding="utf-8"))
        build = manifest.get("build", {})
        if Path(build.get("calib_cache", "")).name == calib_path.name:
            info["engine_json"] = str(engine_json)
            info["engine_source_onnx"] = manifest.get("source_onnx")
            info.update({key: build[key] for key in ("calib_type", "calib_images", "calib_list", "trt_version") if key in build})
    return info


def onnx_conv_inputs(fp32_onnx: Path) -> dict[str, dict]:
    """FP32 ONNX 의 Conv 노드 → {모듈 이름: {node, tensor, weight_shape}}"""
    import onnx
    graph = onnx.load(str(fp32_onnx)).graph
    dims = {init.name: list(init.dims) for init in graph.initializer}
    convs = {}
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        name = node_to_module(node.name)
        if name in convs:
            raise ValueError(f"{fp32_onnx}: 모듈 {name} 에 Conv 노드가 둘 ({convs[name]['node']}, {node.name})")
        convs[name] = {"node": node.name, "tensor": node.input[0], "weight_shape": dims.get(node.input[1]),
                       "output": node.output[0]}   # output: Add 입력 등 독립 양자화기(ActQuantizer.source) 대응용
    return convs


def match_trt_scales(model, calib_path: Path, fp32_onnx: Path) -> dict:
    """QuantConv2d 하나하나를 캐시 항목에 대응시키고 검사. 결과 dict 는 그대로 매니페스트(json)에 남길 수 있음"""
    header, entries = read_calib_cache(calib_path)
    convs = onnx_conv_inputs(fp32_onnx)
    scales, problems = {}, []
    for name, module in quant.named_quant_convs(model):
        info = convs.get(name)
        if info is None:
            problems.append(f"{short_name(name)}: FP32 ONNX 에 같은 이름의 Conv 노드가 없음")
        elif info["weight_shape"] != list(module.conv.weight.shape):
            problems.append(f"{short_name(name)}: 가중치 모양 다름 ONNX {info['weight_shape']} vs 모델 {list(module.conv.weight.shape)}")
        elif info["tensor"] not in entries:
            problems.append(f"{short_name(name)}: 입력 텐서 {info['tensor']} 가 캐시에 없음")
        else:
            scales[name] = {"tensor": info["tensor"], "scale": entries[info["tensor"]], "onnx_node": info["node"]}
    # conv 입력이 아닌 독립 양자화기 (quant.quantize_fusion_add 의 lower_quant 등). source = ("conv_output", conv 모듈 이름)
    extra = {}
    for name, quantizer in quant.named_extra_quantizers(model):
        kind, ref = quantizer.source
        info = convs.get(ref)
        if kind != "conv_output":
            problems.append(f"{short_name(name)}: 모르는 source 종류 {kind}")
        elif info is None:
            problems.append(f"{short_name(name)}: FP32 ONNX 에 conv {ref} 가 없음")
        elif info["output"] not in entries:
            problems.append(f"{short_name(name)}: conv 출력 텐서 {info['output']} 가 캐시에 없음")
        else:
            extra[name] = {"tensor": info["output"], "scale": entries[info["output"]], "onnx_node": info["node"]}
    if problems:
        raise ValueError(f"TensorRT 캐시 대응 실패 ({len(problems)}개):\n  " + "\n  ".join(problems))
    used = {v["tensor"] for v in scales.values()} | {v["tensor"] for v in extra.values()}
    return {"cache": cache_provenance(calib_path, header, len(entries)), "fp32_onnx": str(fp32_onnx),
            "n_onnx_convs": len(convs), "n_matched": len(scales), "n_extra": len(extra), "n_cache_entries": len(entries),
            "unused_cache_tensors": sorted(set(entries) - used),   # conv 입력이 아닌 텐서. 정상
            "scales": scales, "extra_scales": extra}


@torch.no_grad()
def apply_trt_scales(model, matched: dict, only: list[str] | None = None) -> list[str]:
    """matched(match_trt_scales 결과)의 scale 을 QuantConv2d.input_quant.scale 에 넣음. only 로 일부만. 바꾼 모듈 이름 반환"""
    convs = dict(quant.named_quant_convs(model))
    targets = select_convs(list(convs), only)
    for name in targets:
        convs[name].input_quant.scale.fill_(matched["scales"][name]["scale"])
    if not only or "all" in only:   # 독립 양자화기(Add 입력)는 전체 적용일 때 함께
        extras = dict(quant.named_extra_quantizers(model))
        for name, info in matched.get("extra_scales", {}).items():
            extras[name].scale.fill_(info["scale"])
            targets.append(name)
    return targets


def default_fp32_onnx(calib_path: Path) -> Path:
    """build_engine 이 <stem>_int8_<type>.calib 으로 저장하므로 <stem>.onnx 가 그 캐시를 만든 FP32 ONNX"""
    return calib_path.with_name(f"{calib_path.name.split('_int8_')[0]}.onnx")


def describe(matched: dict, applied: list[str] | None = None, act_scales: dict[str, float] | None = None) -> str:
    """대응표 (짧은 이름 | 캐시 텐서 | TRT scale | 현재 scale/TRT 비)"""
    cache = matched["cache"]
    lines = [f"캐시 {cache['path']} ({cache['header']}, 항목 {cache['entries']}개, "
             f"{cache.get('calib_type', '?')} {cache.get('calib_images', '?')}장, sha256 {cache['sha256_16']})",
             f"FP32 ONNX {matched['fp32_onnx']}: Conv {matched['n_onnx_convs']}개 ↔ QuantConv2d {matched['n_matched']}개 대응, "
             f"캐시의 나머지 {len(matched['unused_cache_tensors'])}개 항목은 conv 입력이 아닌 텐서"]
    rows = list(matched["scales"].items()) + list(matched.get("extra_scales", {}).items())
    for name, info in rows:
        row = f"  {short_name(name):40s} {info['tensor']:70s} {info['scale']:.5g}"
        if act_scales is not None and name in act_scales:
            row += f"   기존/TRT {act_scales[name] / info['scale']:.2f}"
        if applied is not None:
            row += "   ← 적용" if name in applied else "   (EMA 유지)"
        lines.append(row)
    return "\n".join(lines)


# ---------- main (대응 확인용) ----------

def _build_quant_model(ckpt: Path, model_py: Path, preprocess: Path | None):
    from deploy.export_onnx import ExportWrapper, build_from_model_py, load_checkpoint, spec_from_mmseg_cfg
    from deploy.preprocess import load_spec
    state_dict, meta = load_checkpoint(ckpt)
    spec = load_spec(preprocess) if preprocess else spec_from_mmseg_cfg(meta["cfg"])
    core = build_from_model_py(model_py, state_dict, spec.num_classes, strict=True)
    model = ExportWrapper(core, spec, argmax=False).eval()
    quant.fuse_conv_bn(model)
    quant.quantize_convs(model)
    return model


def main():
    parser = argparse.ArgumentParser(description="TensorRT 캘리브레이션 캐시 ↔ QuantConv2d 대응 확인")
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--fp32-onnx", type=Path, default=None, help="기본: 캐시 이름에서 유추 (<stem>.onnx)")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--model-py", type=Path, default=Path(__file__).with_name("fastscnn_mmseg.py"))
    parser.add_argument("--preprocess", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="일부만 적용해 볼 때 (예: LTD). 기본 전부")
    args = parser.parse_args()

    fp32_onnx = args.fp32_onnx or default_fp32_onnx(args.calib)
    model = _build_quant_model(args.ckpt, args.model_py, args.preprocess)
    matched = match_trt_scales(model, args.calib, fp32_onnx)
    applied = apply_trt_scales(model, matched, args.only)
    print(describe(matched, applied))
    print(f"\n[ok] QuantConv2d {matched['n_matched']}개 전부 대응됨, {len(applied)}개에 적용")
    print("conv 입력이 아닌 캐시 항목:", ", ".join(matched["unused_cache_tensors"]))


if __name__ == "__main__":
    main()
