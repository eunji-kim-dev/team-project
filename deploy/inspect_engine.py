"""기존 TensorRT 엔진의 원본 레이어 정보와 초기 Conv 관련 정보를 저장한다.

python -m deploy.inspect_engine --engine models/fast_scnn_int8.engine \
    --out results/inspect_ptq

엔진 재빌드/캘리브레이션/추론 없이 Inspector로 읽는다.
출력 텐서 형식만으로 Conv의 연산 정밀도를 판정하지 않는다.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


TARGETS = (
    ("Conv 1", "일반 Conv", "conv.conv"),
    ("Conv 2", "첫 번째 depthwise", "dsconv1.depthwise_conv.conv"),
    ("Conv 3", "첫 번째 pointwise", "dsconv1.pointwise_conv.conv"),
    ("Conv 4", "두 번째 depthwise", "dsconv2.depthwise_conv.conv"),
    ("Conv 5", "두 번째 pointwise", "dsconv2.pointwise_conv.conv"),
)


def target_matches(value):
    """슬래시/점으로 표시된 모듈 경로를 매칭. 숫자 순서로 Conv를 추측하지 않는다."""
    normalized = re.sub(r"[/\\]+", ".", str(value)).casefold()
    matches = []
    for tag, _, suffix in TARGETS:
        path = "learning_to_downsample." + suffix
        if re.search(r"(?<![a-z0-9_])" + re.escape(path) + r"(?![a-z0-9_])", normalized):
            matches.append(tag)
    return matches


def layers_from_info(info):
    layers = info.get("Layers", []) if isinstance(info, dict) else info
    if not isinstance(layers, list):
        raise ValueError("Inspector JSON의 Layers가 목록 형식이 아닙니다.")
    result = []
    for row in layers:
        if isinstance(row, dict):
            result.append(row)
        elif isinstance(row, str):
            result.append({"Name": row})
        else:
            raise ValueError("Inspector 레이어 항목 형식을 확인할 수 없습니다.")
    return result


def ports(layer, key):
    values = layer.get(key, [])
    if isinstance(values, list):
        return [v for v in values if isinstance(v, dict)]
    return []


def cell(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def port_formats(layer, key):
    return "; ".join(str(p.get("Format/Datatype", p.get("Type", "정보 없음")))
                     for p in ports(layer, key)) or "정보 없음"


def matching_rows(layers):
    rows = []
    for index, layer in enumerate(layers):
        name = str(layer.get("Name", ""))
        tensor_names = " ".join(str(p.get("Name", ""))
                                for key in ("Inputs", "Outputs") for p in ports(layer, key))
        if "learning_to_downsample" not in (name + " " + tensor_names).casefold():
            continue
        rows.append(dict(index=index, layer=layer, name_targets=target_matches(name),
                         tensor_targets=target_matches(tensor_names)))
    return rows


def quant_bypass_source(path):
    """복원 검사 실패 확인용. torch를 import하지 않고 현재 해제 분기 원문만 기록."""
    if not path.is_file():
        return "quant.py가 같은 폴더에 없어 해제 분기 원문을 기록하지 않았습니다."
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    lines = [f"파일: `{path}`", f"SHA256: `{hashlib.sha256(raw).hexdigest()}`", ""]
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "quant.py 구문을 읽지 못했습니다. 엔진 정보 추출에는 영향을 주지 않습니다."
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("ActQuantizer", "QuantConv2d"):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "forward":
                    lines += [f"{node.name}.forward", "", "```python",
                              ast.get_source_segment(text, method) or "원문 추출 실패", "```", ""]
    return "\n".join(lines)


def make_report(info, meta):
    layers = layers_from_info(info)
    related = matching_rows(layers)
    lines = ["# 기존 PTQ 엔진의 초기 Conv 정보", "",
             f"- 엔진: `{meta['engine']}`",
             f"- SHA256: `{meta['sha256']}`",
             f"- TensorRT: {meta['tensorrt_version']}",
             f"- 빌드 시 ProfilingVerbosity: {meta['profiling_verbosity']}",
             f"- 추출 시각: {meta['created_at']}",
             f"- Inspector 레이어 수: {len(layers)}", "",
             "엔진을 읽어 기록했습니다. 재빌드·캘리브레이션·추론은 하지 않았습니다.",
             "입출력 텐서 형식과 가중치 형식, TacticName을 함께 확인해야 합니다.",
             "INT8 Conv도 FP16/FP32로 출력할 수 있으며, 커널 이름의 i32/f32만으로 FP32 Conv라고 판정하지 않습니다.",
             "퓨전된 레이어는 여러 Conv 이름을 포함할 수 있습니다. 아래 이름 매칭은 연관 후보이며 연산 정밀도 판정이 아닙니다.",
             "입출력 이름만 일치하면 인접 레이어일 수 있습니다. Reformat 등의 보조 연산도 함께 표시합니다.", "",
             "## 엔진 입출력", "", "| 이름 | 방향 | dtype | shape |", "|---|---|---|---|"]
    for tensor in meta["io_tensors"]:
        lines.append("| " + " | ".join(cell(tensor.get(k)) for k in ("name", "mode", "dtype", "shape")) + " |")
    if any(-1 in tensor["shape"] for tensor in meta["io_tensors"]):
        lines += ["", "동적 shape가 포함되어 있습니다. context에 실제 입력 shape를 지정하지 않은 덤프이므로 일부 텐서 형식은 생략될 수 있습니다."]
    if not layers or not any("Inputs" in layer and "Outputs" in layer for layer in layers):
        lines += ["", "상세 입출력 정보가 없습니다. 현재 덤프로 정밀도를 판정하지 않습니다. 원본 JSON을 확인하세요."]
    lines += ["", "## 초기 5개 Conv 이름 연결", "",
              "| 원본 Conv | 모듈 경로 | 레이어 이름 일치 인덱스 | 입출력 이름 일치 인덱스 |",
              "|---|---|---|---|"]
    for tag, label, suffix in TARGETS:
        named = [str(row["index"]) for row in related if tag in row["name_targets"]]
        tensors = [str(row["index"]) for row in related if tag in row["tensor_targets"]]
        lines.append(f"| {tag}: {label} | `learning_to_downsample.{suffix}` | "
                     f"{', '.join(named) or '미확인'} | {', '.join(tensors) or '미확인'} |")
    lines += ["", "## 초기 블록과 이름이 연결된 레이어", "",
              "| 인덱스 | LayerType | 레이어 이름 | 입력 텐서 형식 | 가중치 정보 | 출력 텐서 형식 |",
              "|---:|---|---|---|---|---|"]
    for row in related:
        layer = row["layer"]
        cells = [row["index"], layer.get("LayerType"), layer.get("Name"),
                 port_formats(layer, "Inputs"), layer.get("Weights"), port_formats(layer, "Outputs")]
        lines.append("| " + " | ".join(cell(value) for value in cells) + " |")
    if not related:
        lines += ["", "초기 블록 이름이 보존된 레이어를 찾지 못했습니다. 처음 5개 레이어를 초기 Conv로 간주하지 않습니다."]
    lines += ["", "## 관련 레이어 원문과 커널 정보", ""]
    for row in related:
        lines += [f"### Inspector 인덱스 {row['index']}", "", "```json",
                  json.dumps(row["layer"], ensure_ascii=False, indent=2), "```", ""]
    lines += ["## 엔진 앞부분 레이어 목록", "",
              "이 목록의 순서는 참고용이며 원본 Conv 번호와 같다는 뜻은 아닙니다.", "",
              "| 인덱스 | LayerType | 이름 |", "|---:|---|---|"]
    for index, layer in enumerate(layers[:20]):
        lines.append(f"| {index} | {cell(layer.get('LayerType'))} | {cell(layer.get('Name'))} |")
    if meta.get("quant_bypass_source"):
        lines += ["", "## 현재 가짜 양자화 해제 구현", "",
                  "별도 민감도 실험의 FP32 복원 실패를 확인하기 위한 코드 원문입니다. 엔진 정밀도 판정에는 사용하지 않습니다.",
                  "이 코드는 import하거나 실행하지 않았습니다.", "", meta["quant_bypass_source"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/inspect_ptq"))
    args = parser.parse_args()
    if not args.engine.is_file():
        parser.error(f"엔진 파일이 없습니다: {args.engine}")
    # 어떤 출력 경로도 입력 엔진을 덮어쓰지 않도록 먼저 검사한다.
    outputs = [args.out / "engine_layers.json", args.out / "engine_report.md"]
    if any(path.resolve() == args.engine.resolve() for path in outputs):
        parser.error("출력 파일 경로가 입력 엔진과 같습니다.")
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise SystemExit("TensorRT를 불러올 수 없습니다. 엔진을 만든 Jetson의 기존 가상환경에서 실행하세요.") from exc
    serialized = args.engine.read_bytes()
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise SystemExit("엔진 로드 실패: 위 TensorRT 로그와 엔진을 만든 장치·버전을 확인하세요.")
    inspector = engine.create_engine_inspector()
    raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(raw, encoding="utf-8")
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Inspector 출력이 유효한 JSON이 아닙니다. 원문은 {outputs[0]}에 저장했습니다.") from exc
    io_tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        io_tensors.append(dict(name=name, mode=str(engine.get_tensor_mode(name)),
                               dtype=str(engine.get_tensor_dtype(name)),
                               shape=list(engine.get_tensor_shape(name))))
    meta = dict(engine=str(args.engine), sha256=hashlib.sha256(serialized).hexdigest(),
                tensorrt_version=trt.__version__, profiling_verbosity=str(engine.profiling_verbosity),
                io_tensors=io_tensors, created_at=datetime.now().astimezone().isoformat(timespec="seconds"))
    try:
        meta["quant_bypass_source"] = quant_bypass_source(Path(__file__).with_name("quant.py"))
    except (OSError, UnicodeError) as exc:
        meta["quant_bypass_source"] = f"quant.py 해제 분기 원문을 읽지 못했습니다: {exc}"
    report = make_report(info, meta)
    outputs[1].write_text(report, encoding="utf-8")
    related = matching_rows(layers_from_info(info))
    print(f"[engine] {args.engine}; TensorRT {trt.__version__}; {engine.profiling_verbosity}")
    print(f"[inspect] 전체 {len(layers_from_info(info))}개 레이어, 초기 블록 관련 후보 {len(related)}개")
    for row in related:
        layer = row["layer"]
        print(f"  [{row['index']}] {layer.get('LayerType', '?')}: {layer.get('Name', '?')}")
        print(f"    입력={port_formats(layer, 'Inputs')}  가중치={layer.get('Weights', '정보 없음')}  출력={port_formats(layer, 'Outputs')}")
        print(f"    TacticName={layer.get('TacticName', '정보 없음')}")
    for output in outputs:
        print(f"[done] {output}")


if __name__ == "__main__":
    main()
