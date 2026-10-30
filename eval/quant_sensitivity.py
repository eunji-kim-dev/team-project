"""어느 블록의 INT8 양자화가 바닥 판별을 망가뜨리는지 측정 (PyTorch 가짜 양자화). Jetson GPU 나 PC 에서 실행

python -m eval.quant_sensitivity --ckpt checkpoints/best_mIoU_iter_1600.pth --data-root data \\
    --qdq-onnx models/fast_scnn_qat_0iter.onnx --out results/quant_sensitivity

원리
  qat_train 과 같은 모델(BN 접음 + QuantConv2d 45개)을 만들고, 가중치와 활성화·가중치 scale 을 기존 Q/DQ ONNX 에서
  그대로 읽어 옴 → PyTorch 가짜 양자화가 그 ONNX 로 만든 엔진(예: qdq_0iter = EMA·64장)과 같은 상태가 됨
  (확인: 전부 켜면 <onnx>.json 의 qat.results 와 같은 mIoU, 전부 끄면 fp32_fused 와 같은 mIoU 가 나와야 함)
  그 상태에서 양자화를 단위별로 끄고(leave-one-out: 그 단위만 float, 나머지 INT8) 켜서(only-one: 그 단위만 INT8)
  val 의 바닥 IoU·장애물→바닥% 가 얼마나 움직이는지 잼. conv 단위에선 활성화 쪽만/가중치 쪽만 끄는 것도 따로 잼
  → 활성화 쪽이 원인이면 그 층의 scale(캘리브레이션) 문제, 가중치 쪽이면 per-channel 가중치 반올림 문제

단위
  stage  LTD / GFE.bottleneck1·2·3 / GFE.ppm_out / FFM / HEAD                                          (7)
  block  LTD.conv·dsconv1·dsconv2 / GFE.bottleneckN.M / GFE.ppm / GFE.out / FFM / HEAD.convs0·convs1·conv_seg (18)
  conv   QuantConv2d 하나                                                                              (45)

출력  <out>/<onnx 이름>/sensitivity.json (전부. 이미지별 혼동행렬 포함), sensitivity.md (표)
  - Δ 는 leave-one-out 이면 "전부 INT8" 대비, only-one 이면 "전부 float" 대비
    장애물→바닥 Δ 가 음수 = 그 단위를 float 로 두니 장애물을 바닥으로 찍는 게 줄었다
  - 95% CI 는 val 사진(26장) paired bootstrap. CI 가 0 을 걸치면 그 차이는 val 안의 노이즈와 구분 안 됨
  - TensorRT 엔트로피 캘리브레이션 캐시(기본 models/fast_scnn_int8_entropy.calib)가 있으면 conv 표에 "EMA/entropy" 열을
    붙임 = 이 ONNX 의 활성화 scale ÷ TensorRT PTQ 가 고른 scale. 1 보다 많이 크면 max 기반 scale 이 꼬리값 때문에
    너무 커져 해상도를 낭비한 층 (PTQ 가 더 잘 나오는 이유 후보)
  - --cumulative K: 장애물→바닥 회복 순 / fp32 일치율 회복 순으로 conv 를 1개, 2개, … K개 float 로 두며 측정
  - --export-top K: 장애물→바닥 회복이 큰 conv K 개를 float 로 둔 Q/DQ ONNX 를 <out>/<이름>/ 에 내보냄
    → python -m deploy.build_engine --onnx <그 파일> --precision qat → eval.bench_engines 로 실제 엔진에서 확인
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy import quant  # noqa: E402
from deploy.export_onnx import (ExportWrapper, build_from_model_py, export, freeze_adaptive_pools,  # noqa: E402
                                load_checkpoint, simplify_and_check, spec_from_mmseg_cfg)
from deploy.preprocess import ModelSpec, load_spec, save_spec  # noqa: E402
from deploy.qat_train import DEFAULT_PALETTE_RGB, IGNORE, RGBMaskDataset, read_split  # noqa: E402

CLASS_KO = {"floor": "바닥", "obstacle": "장애물", "other": "그 외"}


# ---------- 이름 ----------

def short_name(conv_name: str) -> str:
    """QuantConv2d 모듈 이름 → 표에 쓰는 짧은 이름
    core.backbone.learning_to_downsample.dsconv1.depthwise_conv.conv → LTD.dsconv1.depthwise_conv"""
    name = conv_name
    for prefix, short in (("core.backbone.learning_to_downsample.", "LTD."),
                          ("core.backbone.global_feature_extractor.", "GFE."),
                          ("core.backbone.feature_fusion.", "FFM."),
                          ("core.decode_head.", "HEAD.")):
        if name.startswith(prefix):
            name = short + name[len(prefix):]
            break
    if name.endswith(".conv"):
        name = name[:-5]
    return name


def block_of(conv_name: str) -> str:
    s = short_name(conv_name)
    match = re.match(r"(LTD\.(?:conv|dsconv1|dsconv2))", s)
    if match:
        return match.group(1)
    match = re.match(r"(GFE\.bottleneck\d\.\d)", s)
    if match:
        return match.group(1)
    if s.startswith("GFE.ppm"):
        return "GFE.ppm"
    if s.startswith("GFE.out"):
        return "GFE.out"
    if s.startswith("FFM"):
        return "FFM"
    match = re.match(r"HEAD\.convs\.(\d)", s)
    if match:
        return f"HEAD.convs{match.group(1)}"
    if s.startswith("HEAD.conv_seg"):
        return "HEAD.conv_seg"
    return s   # 모르는 구조면 conv 이름 그대로


def stage_of(block: str) -> str:
    if block.startswith("LTD"):
        return "LTD"
    if block.startswith("GFE.bottleneck"):
        return block.rsplit(".", 1)[0]
    if block in ("GFE.ppm", "GFE.out"):
        return "GFE.ppm_out"
    if block.startswith("HEAD"):
        return "HEAD"
    return block


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


# ---------- 모델 ----------

def build_quant_model(args, device):
    state_dict, meta = load_checkpoint(args.ckpt)
    spec = load_spec(args.preprocess) if args.preprocess else spec_from_mmseg_cfg(meta["cfg"])
    dataset_meta = meta.get("dataset_meta") or {}
    palette = [list(color) for color in dataset_meta.get("palette", DEFAULT_PALETTE_RGB)]
    classes = list(dataset_meta.get("classes", ["floor", "obstacle", "other"]))
    core = build_from_model_py(args.model_py, state_dict, spec.num_classes, strict=True)
    model = ExportWrapper(core, spec, argmax=False)
    n_fused = quant.fuse_conv_bn(model)
    n_quant = quant.quantize_convs(model)
    model.to(device).eval()
    print(f"[model] BN 접음 {n_fused}개, QuantConv2d {n_quant}개, device {device}")
    return model, spec, classes, palette


@torch.no_grad()
def load_from_qdq_onnx(model, onnx_path: Path, load_weights: bool) -> dict:
    """Q/DQ ONNX 의 활성화 scale·가중치 scale(·가중치·bias)을 같은 이름의 QuantConv2d 에 넣음. 그래프 구조로 찾으므로
    onnxsim 이 같은 값의 scale 상수를 합쳐 놓았어도 됨"""
    import onnx
    from onnx import numpy_helper
    graph = onnx.load(str(onnx_path)).graph
    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    producer = {output: node for node in graph.node for output in node.output}
    modules = dict(quant.named_quant_convs(model))
    found = set()
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        name = node_to_module(node.name)
        qconv = modules.get(name)
        if qconv is None:
            raise KeyError(f"ONNX conv {node.name} 에 해당하는 QuantConv2d 가 없음 (모듈 이름 {name})")
        act_dq, w_dq = producer.get(node.input[0]), producer.get(node.input[1])
        if not (act_dq is not None and act_dq.op_type == "DequantizeLinear" and w_dq is not None and w_dq.op_type == "DequantizeLinear"):
            raise ValueError(f"{node.name}: 입력 또는 가중치에 Q/DQ 가 없음 (모든 conv 가 양자화된 ONNX 만 지원)")
        qconv.input_quant.scale.copy_(torch.as_tensor(inits[act_dq.input[1]], dtype=torch.float32).reshape(()))
        qconv.w_scale.copy_(torch.as_tensor(inits[w_dq.input[1]], dtype=torch.float32).reshape(-1))
        qconv.w_scale_frozen = True
        if load_weights:
            w_q = producer[w_dq.input[0]]
            qconv.conv.weight.copy_(torch.as_tensor(inits[w_q.input[0]]))
            if len(node.input) > 2 and node.input[2] in inits:
                qconv.conv.bias.copy_(torch.as_tensor(inits[node.input[2]]))
        found.add(name)
    missing = sorted(set(modules) - found)
    if missing:
        raise KeyError(f"ONNX 에 없는 QuantConv2d {len(missing)}개: {missing[:5]}")
    return {"n_conv": len(found), "weights_loaded": load_weights}


def load_trt_entropy_scales(calib_path: Path, fp32_onnx: Path) -> dict[str, float]:
    """TensorRT 캘리브레이션 캐시(텐서 이름: scale 의 hex) + FP32 ONNX 의 conv 입력 텐서 이름 → 모듈 이름별 scale"""
    import onnx
    cache = {}
    for line in calib_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        if ":" in line:
            name, hexval = line.rsplit(":", 1)
            cache[name.strip()] = struct.unpack("!f", bytes.fromhex(hexval.strip()))[0]
    scales = {}
    for node in onnx.load(str(fp32_onnx)).graph.node:
        if node.op_type == "Conv" and node.input[0] in cache:
            scales[node_to_module(node.name)] = cache[node.input[0]]
    return scales


# ---------- 데이터 ----------

class ValCache:
    """val 사진·라벨을 한 번만 읽어 GPU 에 올려 둠 (평가를 수백 번 하므로)"""

    def __init__(self, args, spec: ModelSpec, palette: list, device):
        root = args.data_root
        val_list = args.val_list or root / "splits/val.txt"
        names = read_split(val_list if val_list.exists() else None, root / args.img_dir, args.img_suffix)
        dataset = RGBMaskDataset(root, names, spec, palette, train=False, img_dir=args.img_dir, ann_dir=args.ann_dir,
                                 img_suffix=args.img_suffix, ann_suffix=args.ann_suffix)
        items = [dataset[i] for i in range(len(dataset))]
        self.names = names
        self.x = torch.stack([x for x, _ in items]).to(device)   # (N,3,H,W) 0~255
        self.y = torch.stack([y for _, y in items]).to(device)   # (N,H,W) 클래스 번호
        self.ref_masks: torch.Tensor | None = None               # fp32 예측 (일치율 기준)
        print(f"[data] val {len(names)}장 → {tuple(self.x.shape)} ({self.x.numel() * 4 / 1e6:.0f} MB)")


# ---------- 지표 ----------

def metrics_from_cm(cm: np.ndarray) -> dict:
    """bench_engines.SegMetrics.iou_from_confusion 과 같은 식 (반올림만 안 함). cm[정답, 예측], 0 floor / 1 obstacle"""
    cm = cm.astype(np.float64)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
    result = {"miou": float(np.nanmean(iou)), "iou": [float(v) for v in iou],
              "pixel_acc": float(inter.sum() / max(cm.sum(), 1))}
    if cm.shape[0] >= 2:
        result["obstacle_to_floor_pct"] = float(cm[1, 0] / max(cm[1].sum(), 1) * 100)
        result["floor_to_obstacle_pct"] = float(cm[0, 1] / max(cm[0].sum(), 1) * 100)
    return result


def paired_bootstrap_ci(cms_a: np.ndarray, cms_b: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    """(a − b) 지표 차이의 95% CI. 같은 사진들을 복원추출해 두 쪽 혼동행렬을 같이 합치므로 paired. cms: (N,C,C)"""
    n, c = cms_a.shape[0], cms_a.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1)

    def stats(cms):
        m = (counts @ cms.reshape(n, -1).astype(np.float64)).reshape(n_boot, c, c)
        inter = np.einsum("bii->bi", m)
        iou = inter / np.maximum(m.sum(1) + m.sum(2) - inter, 1)
        return {"miou": iou.mean(1), "floor_iou": iou[:, 0],
                "obstacle_to_floor_pct": m[:, 1, 0] / np.maximum(m[:, 1].sum(1), 1) * 100,
                "floor_to_obstacle_pct": m[:, 0, 1] / np.maximum(m[:, 0].sum(1), 1) * 100}

    sa, sb = stats(cms_a), stats(cms_b)
    return {key: [float(np.percentile(sa[key] - sb[key], 2.5)), float(np.percentile(sa[key] - sb[key], 97.5))] for key in sa}


# ---------- 측정 ----------

class Sweep:
    """양자화 on/off 조합 하나를 val 전체로 평가. 결과(records)에 이미지별 혼동행렬까지 남김"""

    def __init__(self, model, cache: ValCache, num_classes: int, convs: dict):
        self.model = model
        self.cache = cache
        self.n = num_classes
        self.convs = convs
        self.records: list[dict] = []

    def _set(self, off_act: set, off_weight: set) -> None:
        for name, module in self.convs.items():
            module.input_quant.enabled = True
            module.act_quant_enabled = name not in off_act
            module.weight_quant_enabled = name not in off_weight

    @torch.no_grad()
    def _forward_all(self):
        n, c = len(self.cache.names), self.n
        cms = torch.zeros(n, c, c, dtype=torch.int64, device=self.cache.x.device)
        masks = torch.empty(self.cache.y.shape, dtype=torch.uint8, device=self.cache.x.device)
        for i in range(n):
            pred = self.model(self.cache.x[i:i + 1]).argmax(1)[0]
            masks[i] = pred.to(torch.uint8)
            y = self.cache.y[i]
            valid = y != IGNORE
            cms[i] = torch.bincount(y[valid] * c + pred[valid], minlength=c * c).view(c, c)
        return cms.cpu().numpy(), masks

    def run(self, tag: str, level: str, mode: str, unit: str, off_act: set, off_weight: set, as_reference=False) -> dict:
        started = time.perf_counter()
        self._set(off_act, off_weight)
        cms, masks = self._forward_all()
        record = {"tag": tag, "level": level, "mode": mode, "unit": unit,
                  "off_act": sorted(off_act), "off_weight": sorted(off_weight),
                  "metrics": metrics_from_cm(cms.sum(0)), "cms": cms.tolist()}
        if as_reference:
            self.cache.ref_masks = masks.clone()
        if self.cache.ref_masks is not None:
            equal = masks == self.cache.ref_masks
            record["agree_pct"] = float(equal.float().mean()) * 100
            record["class_agree_pct"] = [float(equal[self.cache.ref_masks == c].float().mean()) * 100 for c in range(self.n)]
        m = record["metrics"]
        print(f"  {tag:48s} mIoU {m['miou']:.4f}  바닥 {m['iou'][0]:.4f}  장→바 {m['obstacle_to_floor_pct']:.3f}%  "
              f"바→장 {m['floor_to_obstacle_pct']:.3f}%  fp32 일치 {record.get('agree_pct', 100):.2f}%  ({time.perf_counter() - started:.1f}s)")
        self.records.append(record)
        return record


def resolve_units(convs: dict, wanted: list[str]) -> list[str]:
    """짧은 이름·블록·스테이지·모듈 이름(또는 all) 목록 → QuantConv2d 모듈 이름 목록"""
    chosen = []
    for item in wanted:
        if item == "all":
            hits = list(convs)
        else:
            hits = [n for n in convs if item in (short_name(n), block_of(n), stage_of(block_of(n)), n)]
        if not hits:
            raise SystemExit(f"'{item}' 에 맞는 conv 가 없음. 예: {[short_name(n) for n in list(convs)[:3]]}, LTD, GFE.bottleneck1.0, all")
        chosen += [n for n in hits if n not in chosen]
    return chosen


def make_groups(conv_names: list[str]) -> dict[str, dict[str, list[str]]]:
    groups = {"stage": {}, "block": {}, "conv": {}}
    for name in conv_names:
        block = block_of(name)
        groups["stage"].setdefault(stage_of(block), []).append(name)
        groups["block"].setdefault(block, []).append(name)
        groups["conv"][short_name(name)] = [name]
    return groups


# ---------- 표 ----------

def _f(value, digits):
    return "-" if value is None else f"{value:+.{digits}f}" if digits < 0 else f"{value:.{digits}f}"


def _delta(value, digits):
    return "-" if value is None else f"{value:+.{digits}f}"


def _ci(ci, digits):
    return "-" if ci is None else f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(records: list[dict], groups: dict, entropy_scales: dict, act_scales: dict, onnx_name: str,
                 expected: dict, n_boot: int, out_md: Path, fp32_label: str) -> list[str]:
    """sensitivity.md 작성. 장애물→바닥 회복 순으로 정렬한 conv 이름 목록을 돌려줌"""
    by_tag = {r["tag"]: r for r in records}
    fp32, int8 = by_tag["fp32"], by_tag["int8"]
    fp32_cms, int8_cms = np.asarray(fp32["cms"]), np.asarray(int8["cms"])

    def deltas(record, base):
        m, b = record["metrics"], base["metrics"]
        return {"miou": m["miou"] - b["miou"], "floor_iou": m["iou"][0] - b["iou"][0],
                "obstacle_to_floor_pct": m["obstacle_to_floor_pct"] - b["obstacle_to_floor_pct"],
                "floor_to_obstacle_pct": m["floor_to_obstacle_pct"] - b["floor_to_obstacle_pct"]}

    for record in records:
        if record["mode"] in ("loo", "loo_act", "loo_weight") or record["level"] == "cumulative":
            base, base_cms = int8, int8_cms
        elif record["mode"] == "only":
            base, base_cms = fp32, fp32_cms
        else:
            continue
        record["delta"] = deltas(record, base)
        record["ci95"] = paired_bootstrap_ci(np.asarray(record["cms"]), base_cms, n_boot)

    lines = [f"# 블록별 양자화 민감도: {onnx_name} ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    lines += ["val 26장, PyTorch 가짜 양자화 (scale·가중치는 위 ONNX 에서 읽음). Δ: leave-one-out(off) 은 전부-INT8 대비, "
              "only-one 은 전부-float 대비. CI95 는 사진 paired bootstrap (n={}). 장→바 = 정답 장애물 중 바닥으로 찍은 %".format(n_boot), ""]

    # 기준
    lines += ["## 기준", ""]
    rows = []
    for tag, label in (("fp32", fp32_label), ("int8", "전부 INT8 (이 ONNX 와 같은 상태)")):
        m = by_tag[tag]["metrics"]
        rows.append([label, _f(m["miou"], 4), _f(m["iou"][0], 4), _f(m["iou"][1], 4), _f(m["iou"][2], 4),
                     _f(m["obstacle_to_floor_pct"], 3), _f(m["floor_to_obstacle_pct"], 3), _f(by_tag[tag].get("agree_pct"), 2)])
    lines.append(table(["상태", "mIoU", "IoU 바닥", "IoU 장애물", "IoU 그 외", "장→바%", "바→장%", "fp32 일치%"], rows))
    gap = deltas(int8, fp32)
    lines += ["", f"INT8 − fp32 차이: mIoU {gap['miou']:+.4f}, 바닥 IoU {gap['floor_iou']:+.4f}, 장→바 {gap['obstacle_to_floor_pct']:+.3f}%p, "
              f"바→장 {gap['floor_to_obstacle_pct']:+.3f}%p"]
    if expected:
        lines += ["", "ONNX 옆 json 에 기록된 값과 비교 (같아야 재현이 맞음): " + ", ".join(
            f"{k} mIoU {v:.4f}" for k, v in expected.items())]
    lines.append("")

    def loo_rows(level, key_fn):
        rows = []
        for record in sorted((r for r in records if r["level"] == level and r["mode"] == "loo"),
                             key=lambda r: (r["delta"]["obstacle_to_floor_pct"], -r["delta"]["floor_iou"])):
            d, m, ci = record["delta"], record["metrics"], record["ci95"]
            rows.append(key_fn(record) + [_delta(d["obstacle_to_floor_pct"], 3), _ci(ci["obstacle_to_floor_pct"], 3),
                                          _delta(d["floor_iou"], 4), _ci(ci["floor_iou"], 4), _delta(d["miou"], 4),
                                          _delta(d["floor_to_obstacle_pct"], 3), _f(record["agree_pct"], 2)])
        return rows

    def only_rows(level, key_fn):
        rows = []
        for record in sorted((r for r in records if r["level"] == level and r["mode"] == "only"),
                             key=lambda r: (-r["delta"]["obstacle_to_floor_pct"], r["delta"]["floor_iou"])):
            d, ci = record["delta"], record["ci95"]
            rows.append(key_fn(record) + [_delta(d["obstacle_to_floor_pct"], 3), _ci(ci["obstacle_to_floor_pct"], 3),
                                          _delta(d["floor_iou"], 4), _delta(d["miou"], 4),
                                          _delta(d["floor_to_obstacle_pct"], 3), _f(record["agree_pct"], 2)])
        return rows

    loo_header = ["Δ장→바%p", "CI95", "Δ바닥 IoU", "CI95", "ΔmIoU", "Δ바→장%p", "fp32 일치%"]
    only_header = ["Δ장→바%p", "CI95", "Δ바닥 IoU", "ΔmIoU", "Δ바→장%p", "fp32 일치%"]
    for level, label in (("stage", "스테이지"), ("block", "블록")):
        n_convs = {unit: len(members) for unit, members in groups[level].items()}
        key_fn = lambda r: [r["unit"], str(n_convs[r["unit"]])]   # noqa: E731
        lines += [f"## {label} 단위: 그 단위만 float (leave-one-out, 전부-INT8 대비. 음수 = 회복)", "",
                  table(["단위", "conv 수"] + loo_header, loo_rows(level, key_fn)), "",
                  f"## {label} 단위: 그 단위만 INT8 (only-one, 전부-float 대비. 양수 = 손상)", "",
                  table(["단위", "conv 수"] + only_header, only_rows(level, key_fn)), ""]

    # conv 단위: both / act / weight / only-one 을 한 표에
    conv_records = {(r["unit"], r["mode"]): r for r in records if r["level"] == "conv"}
    units = sorted({unit for unit, _ in conv_records},
                   key=lambda u: (conv_records[(u, "loo")]["delta"]["obstacle_to_floor_pct"], -conv_records[(u, "loo")]["delta"]["floor_iou"]))
    rows = []
    for unit in units:
        full = groups["conv"][unit][0]
        loo, act, weight, only = (conv_records.get((unit, mode)) for mode in ("loo", "loo_act", "loo_weight", "only"))
        ema, ent = act_scales.get(full), entropy_scales.get(full)
        ratio = f"{ema / ent:.2f}" if (ema and ent) else "-"
        rows.append([unit, block_of(full), ratio,
                     _delta(loo["delta"]["obstacle_to_floor_pct"], 3), _ci(loo["ci95"]["obstacle_to_floor_pct"], 3),
                     _delta(act["delta"]["obstacle_to_floor_pct"], 3) if act else "-",
                     _delta(weight["delta"]["obstacle_to_floor_pct"], 3) if weight else "-",
                     _delta(loo["delta"]["floor_iou"], 4), _ci(loo["ci95"]["floor_iou"], 4),
                     _delta(act["delta"]["floor_iou"], 4) if act else "-",
                     _delta(weight["delta"]["floor_iou"], 4) if weight else "-",
                     _f(loo["agree_pct"], 2),
                     _delta(only["delta"]["obstacle_to_floor_pct"], 3) if only else "-",
                     _delta(only["delta"]["floor_iou"], 4) if only else "-",
                     _f(only["agree_pct"], 2) if only else "-"])
    lines += ["## conv 단위 (장애물→바닥 회복 순)", "",
              "off = 그 conv 의 활성화+가중치 양자화를 끔 (전부-INT8 대비), off_act = 활성화 쪽만, off_w = 가중치 쪽만. "
              "only = 그 conv 만 INT8 (전부-float 대비). EMA/entropy = 이 ONNX 활성화 scale ÷ TensorRT PTQ 엔트로피 scale", "",
              table(["conv", "블록", "EMA/entropy", "off Δ장→바", "CI95", "off_act Δ장→바", "off_w Δ장→바",
                     "off Δ바닥IoU", "CI95", "off_act Δ바닥", "off_w Δ바닥", "off 일치%",
                     "only Δ장→바", "only Δ바닥", "only 일치%"], rows), ""]

    cumulative = [r for r in records if r["level"] == "cumulative"]
    if cumulative:
        rows = []
        for record in cumulative:
            d, m, ci = record["delta"], record["metrics"], record["ci95"]
            rows.append([record["mode"], record["unit"], str(len(record["off_act"])), ", ".join(short_name(n) for n in record["off_act"]),
                         _f(m["obstacle_to_floor_pct"], 3), _delta(d["obstacle_to_floor_pct"], 3), _ci(ci["obstacle_to_floor_pct"], 3),
                         _f(m["iou"][0], 4), _delta(d["floor_iou"], 4), _delta(d["miou"], 4), _f(record["agree_pct"], 2)])
        lines += ["## 누적: 회복 순으로 conv 를 여러 개 float 로 (전부-INT8 대비)", "",
                  f"fp32 까지의 거리: 장→바 {-gap['obstacle_to_floor_pct']:+.3f}%p, 바닥 IoU {-gap['floor_iou']:+.4f}, mIoU {-gap['miou']:+.4f}", "",
                  table(["순위 기준", "단계", "float conv 수", "float 로 둔 conv", "장→바%", "Δ", "CI95", "바닥 IoU", "Δ", "ΔmIoU", "fp32 일치%"], rows), ""]

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return [groups["conv"][unit][0] for unit in units]


# ---------- 내보내기 ----------

def export_variant(model, spec: ModelSpec, float_convs: list[str], out: Path, opset: int, source: dict) -> int:
    """float_convs 만 양자화를 끈 Q/DQ ONNX (+ .json). qat_train.export 와 같은 방식. Q 노드 수 반환"""
    import onnx
    exported = copy.deepcopy(model).cpu().eval()
    for name, module in quant.named_quant_convs(exported):
        module.input_quant.enabled = name not in float_convs   # 입력·가중치 둘 다
        module.act_quant_enabled = module.weight_quant_enabled = True
    exported.argmax = True
    freeze_adaptive_pools(exported, torch.zeros(1, 3, spec.height, spec.width))
    out.parent.mkdir(parents=True, exist_ok=True)
    export(exported, spec, out, opset)

    def count_q(path):
        return sum(1 for node in onnx.load(str(path)).graph.node if node.op_type == "QuantizeLinear")

    n_q = count_q(out)
    raw = out.read_bytes()
    simplify_and_check(out)
    if count_q(out) != n_q:   # onnxsim 이 Q/DQ 를 건드렸으면 원본으로
        out.write_bytes(raw)
    manifest_spec = dataclasses.replace(spec, normalize_in_graph=True, argmax_in_graph=True)
    save_spec(out.with_suffix(out.suffix + ".json"), manifest_spec, onnx_input="input", onnx_output="mask", opset=opset,
              quantization="qat_qdq_int8_partial", float_convs=[short_name(n) for n in float_convs],
              torch_version=torch.__version__, exported_at=time.strftime("%Y-%m-%d %H:%M:%S"), **source)
    return n_q


# ---------- main ----------

def parse_args():
    parser = argparse.ArgumentParser(description="블록별 INT8 양자화 민감도 (바닥 IoU·장애물→바닥%)")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--qdq-onnx", required=True, type=Path, help="scale·가중치를 읽을 Q/DQ ONNX (qat_train 산출물)")
    parser.add_argument("--scales-only", action="store_true", help="ONNX 에서 scale 만 읽고 가중치는 체크포인트 것(BN 접은 것) 유지")
    parser.add_argument("--model-py", type=Path, default=Path(__file__).resolve().parents[1] / "deploy/fastscnn_mmseg.py")
    parser.add_argument("--preprocess", type=Path, help="preprocess.json (생략하면 체크포인트 meta.cfg 에서)")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--img-dir", default="images")
    parser.add_argument("--ann-dir", default="labels_RGB")
    parser.add_argument("--img-suffix", default="_color.png")
    parser.add_argument("--ann-suffix", default="_label_RGB.png")
    parser.add_argument("--val-list", type=Path, default=None, help="기본: <data-root>/splits/val.txt")
    parser.add_argument("--out", type=Path, default=Path("results/quant_sensitivity"))
    parser.add_argument("--levels", nargs="*", default=["stage", "block", "conv"], choices=["stage", "block", "conv"],
                        help="측정할 단위. 비우면 기준 두 줄만 잼 (--export-convs 만 쓸 때)")
    parser.add_argument("--cumulative", type=int, default=8, help="회복 순 상위 K 개까지 누적으로 float 로 두며 측정 (0 = 안 함)")
    parser.add_argument("--export-top", type=int, default=0, help="장→바 회복 상위 K conv 를 float 로 둔 Q/DQ ONNX 내보내기")
    parser.add_argument("--export-convs", nargs="*", default=[],
                        help="이 conv 들(짧은 이름. 예: LTD.dsconv1.depthwise_conv, 또는 블록 이름 LTD)을 float 로 둔 Q/DQ ONNX 내보내기")
    parser.add_argument("--export-name", default=None,
                        help="--export-convs 결과 파일 이름 (기본: <onnx 이름>_float_<conv 수>). --export-convs 없이 주면 지금 scale 그대로 내보냄")
    parser.add_argument("--trt-scales-for", nargs="*", default=[],
                        help="이 단위들(conv·블록·스테이지 이름, 또는 all)의 활성화 scale 을 TensorRT 엔트로피 캘리브레이션 값으로 "
                             "바꾸고 측정 (= 그 층에 PTQ 식 scale 을 쓰면 얼마나 회복되나)")
    parser.add_argument("--run-name", default=None, help="결과 폴더 이름 (기본: ONNX 이름)")
    parser.add_argument("--trt-calib", type=Path, default=None, help="기본: <qdq-onnx 폴더>/fast_scnn_int8_entropy.calib")
    parser.add_argument("--fp32-onnx", type=Path, default=None, help="기본: <qdq-onnx 폴더>/fast_scnn.onnx")
    parser.add_argument("--boot", type=int, default=2000, help="bootstrap 횟수")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    print(f"[env] torch {torch.__version__}, device {device}")

    model, spec, classes, palette = build_quant_model(args, device)
    info = load_from_qdq_onnx(model, args.qdq_onnx, load_weights=not args.scales_only)
    print(f"[onnx] {args.qdq_onnx}: conv {info['n_conv']}개의 scale{'·가중치' if info['weights_loaded'] else ''} 읽음")
    convs = dict(quant.named_quant_convs(model))
    act_scales = {name: float(module.input_quant.scale) for name, module in convs.items()}

    expected = {}
    fp32_label = "전부 float (BN 접은 FP32 체크포인트)"
    onnx_json = args.qdq_onnx.with_suffix(args.qdq_onnx.suffix + ".json")
    if onnx_json.exists():
        qat_meta = json.loads(onnx_json.read_text(encoding="utf-8")).get("qat", {})
        results = qat_meta.get("results", {})
        expected = {key: results[key]["miou"] for key in ("fp32_fused", "qat") if key in results}
        if qat_meta.get("iters") and info["weights_loaded"]:   # 학습된 가중치: 양자화 끈 상태는 json 에 없음
            expected.pop("fp32_fused", None)
            fp32_label = f"전부 float (QAT {qat_meta['iters']} iter 학습된 가중치, 양자화만 끔)"
            print(f"[onnx] QAT {qat_meta['iters']} iter 학습된 ONNX → fp32 행은 학습된 가중치로 양자화만 끈 것")

    entropy_scales = {}
    calib = args.trt_calib or args.qdq_onnx.parent / "fast_scnn_int8_entropy.calib"
    fp32_onnx = args.fp32_onnx or args.qdq_onnx.parent / "fast_scnn.onnx"
    if calib.exists() and fp32_onnx.exists():
        entropy_scales = load_trt_entropy_scales(calib, fp32_onnx)
        print(f"[trt] 엔트로피 캘리브레이션 scale {len(entropy_scales)}개 읽음 ({calib})")
    else:
        print("[trt] 캘리브레이션 캐시 또는 FP32 ONNX 없음 → EMA/entropy 열 생략")

    if args.trt_scales_for:
        if not entropy_scales:
            raise SystemExit("--trt-scales-for 에는 TensorRT 캘리브레이션 캐시와 FP32 ONNX 가 필요함 (--trt-calib, --fp32-onnx)")
        targets = resolve_units(convs, args.trt_scales_for)
        with torch.no_grad():
            for name in targets:
                convs[name].input_quant.scale.fill_(entropy_scales[name])
        act_scales = {name: float(module.input_quant.scale) for name, module in convs.items()}
        expected.pop("qat", None)   # scale 이 바뀌었으니 json 의 INT8 값과 같을 이유가 없음
        print(f"[trt] conv {len(targets)}개의 활성화 scale 을 엔트로피 캘리브레이션 값으로 바꿈: "
              f"{', '.join(short_name(n) for n in targets[:6])}{' …' if len(targets) > 6 else ''}")

    cache = ValCache(args, spec, palette, device)
    sweep = Sweep(model, cache, spec.num_classes, convs)
    all_names = set(convs)
    groups = make_groups(sorted(convs))

    print("\n[기준]")
    fp32 = sweep.run("fp32", "base", "all_off", "-", all_names, all_names, as_reference=True)
    int8 = sweep.run("int8", "base", "all_on", "-", set(), set())
    for key, tag in (("fp32_fused", "fp32"), ("qat", "int8")):
        if key in expected:
            diff = sweep.records[0 if tag == "fp32" else 1]["metrics"]["miou"] - expected[key]
            print(f"  재현 확인: {tag} mIoU − json {key} = {diff:+.5f} {'(OK)' if abs(diff) < 5e-4 else '(다름! scale·가중치 로드 확인)'}")

    for level in args.levels:
        print(f"\n[{level} 단위] {len(groups[level])}개")
        for unit, members in groups[level].items():
            members = set(members)
            sweep.run(f"{level}:{unit}:off", level, "loo", unit, members, members)
            sweep.run(f"{level}:{unit}:only", level, "only", unit, all_names - members, all_names - members)
            if level == "conv":
                sweep.run(f"{level}:{unit}:off_act", level, "loo_act", unit, members, set())
                sweep.run(f"{level}:{unit}:off_weight", level, "loo_weight", unit, set(), members)

    if args.cumulative > 0 and "conv" in args.levels:
        loo = [r for r in sweep.records if r["level"] == "conv" and r["mode"] == "loo"]
        base = int8["metrics"]
        rankings = {
            "장→바 회복": sorted(loo, key=lambda r: (r["metrics"]["obstacle_to_floor_pct"] - base["obstacle_to_floor_pct"],
                                                   -(r["metrics"]["iou"][0] - base["iou"][0]))),
            "fp32 일치 회복": sorted(loo, key=lambda r: -r["agree_pct"]),
        }
        for label, ranked in rankings.items():
            print(f"\n[누적: {label} 순]")
            for k in range(1, min(args.cumulative, len(ranked)) + 1):
                off = {r["off_act"][0] for r in ranked[:k]}
                sweep.run(f"cumulative:{label}:top{k}", "cumulative", label, f"top{k}", off, off)

    run_name = args.run_name or args.qdq_onnx.stem
    out_dir = args.out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked_convs = write_report(sweep.records, groups, entropy_scales, act_scales, run_name, expected, args.boot,
                                out_dir / "sensitivity.md", fp32_label)
    payload = {"env": {"torch": torch.__version__, "device": str(device), "datetime": time.strftime("%Y-%m-%d %H:%M:%S")},
               "args": {k: str(v) for k, v in vars(args).items()}, "classes": classes, "val_names": cache.names,
               "expected_from_onnx_json": expected, "act_scales": act_scales, "trt_entropy_scales": entropy_scales,
               "trt_scales_for": args.trt_scales_for,
               "groups": groups, "ranked_convs_by_obstacle_to_floor_recovery": ranked_convs, "records": sweep.records}
    (out_dir / "sensitivity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[done] {out_dir}/sensitivity.md, sensitivity.json")

    exports = []
    if args.export_top > 0:
        exports.append((ranked_convs[:args.export_top], out_dir / f"{args.qdq_onnx.stem}_top{args.export_top}.onnx"))
    if args.export_convs or args.export_name:   # --export-name 만 주면 지금 상태(예: --trt-scales-for 적용) 그대로 내보냄
        chosen = resolve_units(convs, args.export_convs)   # 짧은 이름, 블록·스테이지 이름, 모듈 이름 전부 받음
        name = args.export_name or f"{args.qdq_onnx.stem}_float_{len(chosen)}"
        exports.append((chosen, out_dir / f"{name}.onnx"))
    for float_convs, out in exports:
        n_q = export_variant(model, spec, float_convs, out, args.opset,
                             {"source_onnx": str(args.qdq_onnx), "source_ckpt": str(args.ckpt),
                              "trt_entropy_scales_for": [short_name(n) for n in resolve_units(convs, args.trt_scales_for)] if args.trt_scales_for else []})
        print(f"[export] {out}: float 로 둔 conv {len(float_convs)}개 ({', '.join(short_name(n) for n in float_convs)}), Q 노드 {n_q}개")
        print(f"  → python -m deploy.build_engine --onnx {out} --precision qat")


if __name__ == "__main__":
    main()
