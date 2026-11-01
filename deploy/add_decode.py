"""로짓 ONNX 뒤에 softmax 신뢰도·argmax 를 붙여 카메라 앱용 '디코드 포함' ONNX 를 만든다

python -m deploy.add_decode --onnx models/calib_trt_adds/fast_scnn_qat_logits.onnx
    → models/calib_trt_adds/fast_scnn_qat_decode.onnx (+ .json: decode_in_graph=true, onnx_output=["labels","confidence"])
python -m deploy.build_engine --onnx models/calib_trt_adds/fast_scnn_qat_decode.onnx --precision qat
    → models/calib_trt_adds/fast_scnn_qat_decode_int8.engine

- 왜: 카메라 앱은 로짓 [1,3,H,W] 를 받아 CPU 로 softmax 신뢰도를 계산하는데 그 디코드가 프레임당 15 ms (엔진 실행 3 ms 의 5배)이고,
  처리 스레드가 GIL 을 잡는 시간이라 다른 스레드까지 느리게 한다. 같은 계산을 엔진 그래프 끝에 넣으면 GPU 가 한다
- 출력 labels: int32 [1,H,W] = argmax(logits, axis=1). 동점이면 앞 채널 (앱 decode_prediction 과 같은 규칙)
  출력 confidence: float32 [1,H,W] = max softmax(logits, axis=1) = 앱이 CPU 로 계산하던 신뢰도
- 앱(d435map.trt_fastscnn)은 매니페스트 decode_in_graph 를 보고 두 출력을 그대로 쓴다. 벤치 도구는 첫 출력(labels)을 마스크로 쓴다
- 정확도 확인: python -m eval.confidence_agreement --models cpu_fp32=... int8=<logits engine> int8_decode=<decode engine> --pairs int8_decode:int8 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="로짓 ONNX 에 argmax·softmax 신뢰도 출력 추가")
    parser.add_argument("--onnx", required=True, type=Path, help="--keep-logits 로 만든 로짓 ONNX (+ 옆의 .json)")
    parser.add_argument("--out", type=Path, default=None, help="기본: <이름에서 _logits 제거>_decode.onnx")
    args = parser.parse_args()

    model = onnx.load(str(args.onnx))
    manifest_path = args.onnx.with_suffix(args.onnx.suffix + ".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("argmax_in_graph"):
        raise SystemExit("argmax 가 이미 든 ONNX. export_onnx --keep-logits 로 만든 로짓 ONNX 를 줄 것")
    graph = model.graph
    if len(graph.output) != 1:
        raise SystemExit(f"출력 1개(로짓) ONNX 만 지원: {[o.name for o in graph.output]}")
    logits = graph.output[0]
    dims = [d.dim_value for d in logits.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[1] < 2:
        raise SystemExit(f"로짓 출력은 [N,C,H,W] 여야 함: {dims}")
    n, _, h, w = dims
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))

    nodes = [helper.make_node("Softmax", [logits.name], ["decode/probs"], axis=1, name="decode/softmax")]
    if opset >= 18:   # ReduceMax 의 axes 가 입력으로 바뀜
        graph.initializer.append(numpy_helper.from_array(np.array([1], np.int64), "decode/axes"))
        nodes.append(helper.make_node("ReduceMax", ["decode/probs", "decode/axes"], ["confidence"], keepdims=0, name="decode/confidence"))
    else:
        nodes.append(helper.make_node("ReduceMax", ["decode/probs"], ["confidence"], axes=[1], keepdims=0, name="decode/confidence"))
    nodes += [helper.make_node("ArgMax", [logits.name], ["decode/argmax"], axis=1, keepdims=0, select_last_index=0, name="decode/argmax"),
              helper.make_node("Cast", ["decode/argmax"], ["labels"], to=TensorProto.INT32, name="decode/labels")]
    graph.node.extend(nodes)
    del graph.output[:]
    graph.output.extend([helper.make_tensor_value_info("labels", TensorProto.INT32, [n, h, w]),
                         helper.make_tensor_value_info("confidence", TensorProto.FLOAT, [n, h, w])])
    onnx.checker.check_model(model)

    out = args.out or args.onnx.with_name(args.onnx.stem.replace("_logits", "") + "_decode.onnx")
    onnx.save(model, str(out))
    manifest.update(onnx_output=["labels", "confidence"], argmax_in_graph=False, decode_in_graph=True,
                    decode={"labels": "int32 [N,H,W] argmax(logits, axis=1); first index on ties",
                            "confidence": "float32 [N,H,W] max softmax(logits, axis=1)",
                            "source_onnx": str(args.onnx)})
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}: outputs labels int32 {[n, h, w]}, confidence float32 {[n, h, w]} (opset {opset})")


if __name__ == "__main__":
    main()
