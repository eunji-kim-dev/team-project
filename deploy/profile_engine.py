"""TensorRT 엔진 층별 실행 시간·정밀도 프로파일 (왜 느린지 볼 때). Jetson 에서 실행

python -m deploy.profile_engine ptq=models/fast_scnn_int8.engine qat=models/calib_trt/fast_scnn_qat_int8.engine \\
    --iters 300 --json results/profile.json

- 엔진마다: 전체 시간(프로파일러 없이) / 층 종류·출력 정밀도별 (개수, ms, %) / 가장 느린 층
- 엔진이 둘 이상이면 첫 엔진 기준으로 종류별 시간 차이표 (예: QAT 엔진이 PTQ 보다 느린 이유 → FP16 conv, Reformat 증가)
- IProfiler 는 층마다 동기화하므로 프로파일 합이 전체 시간보다 조금 크다. 클럭 고정(jetson_clocks) 상태에서 비교할 것
- 출력 정밀도는 engine inspector 의 Outputs[0].Format/Datatype (Int8/Half/Float). 'kgen' 은 TensorRT 가 만든 융합 커널(여기선 argmax)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy import cuda_util as cu  # noqa: E402


class _Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.ms = collections.defaultdict(float)

    def report_layer_time(self, layer_name, ms):
        self.ms[layer_name] += ms


def precision_of(layer: dict) -> str:
    fmt = (layer.get("Outputs") or [{}])[0].get("Format/Datatype", "")
    for key, label in (("Int8", "INT8"), ("Half", "FP16"), ("Float", "FP32"), ("Int32", "INT32")):
        if key in fmt:
            return label
    return fmt or "?"


def profile(path: Path, iters: int, warmup: int = 50) -> dict:
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    engine = trt.Runtime(logger).deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"엔진 로드 실패: {path}")
    layers = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
    meta = {layer["Name"]: (layer["LayerType"], precision_of(layer)) for layer in layers}

    context = engine.create_execution_context()
    stream = cu.Stream()
    buffers = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        host = cu.pinned_empty(tuple(engine.get_tensor_shape(name)), trt.nptype(engine.get_tensor_dtype(name)))
        device = cu.DeviceBuffer(host.nbytes)
        context.set_tensor_address(name, device.ptr)
        buffers.append((host, device))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:   # 사진 대신 0~255 난수 (시간은 입력값과 무관)
            host[...] = np.random.default_rng(0).random(host.shape, dtype=np.float32) * 255
            cu.memcpy_h2d_async(device.ptr, host, stream)
    stream.synchronize()

    for _ in range(warmup):
        context.execute_async_v3(stream.handle)
    stream.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        context.execute_async_v3(stream.handle)
    stream.synchronize()
    total_ms = (time.perf_counter() - started) / iters * 1e3

    profiler = _Profiler()
    context.profiler = profiler
    for _ in range(iters):
        context.execute_async_v3(stream.handle)
        stream.synchronize()
    per_layer = {name: ms / iters for name, ms in profiler.ms.items()}

    by_kind: dict[tuple, list] = collections.defaultdict(lambda: [0, 0.0])
    for name, ms in per_layer.items():
        key = meta.get(name, ("?", "?"))
        by_kind[key][0] += 1
        by_kind[key][1] += ms
    return {"path": str(path), "total_ms": total_ms, "profiled_sum_ms": sum(per_layer.values()), "n_layers": len(per_layer),
            "by_kind": {f"{kind} {prec}": {"n": n, "ms": ms} for (kind, prec), (n, ms) in by_kind.items()},
            "slowest": [{"ms": ms, "type": meta.get(name, ("?", "?"))[0], "precision": meta.get(name, ("?", "?"))[1], "name": name}
                        for name, ms in sorted(per_layer.items(), key=lambda kv: -kv[1])[:10]]}


def print_report(tag: str, result: dict, top: int) -> None:
    total = result["profiled_sum_ms"]
    print(f"\n=== {tag}: 전체 {result['total_ms']:.3f} ms (프로파일 합 {total:.3f} ms, 층 {result['n_layers']}개) ===")
    print(f"{'층 종류':24} {'정밀도':5} {'개수':>4} {'ms':>7} {'%':>5}")
    for key, value in sorted(result["by_kind"].items(), key=lambda kv: -kv[1]["ms"]):
        kind, prec = key.rsplit(" ", 1)
        print(f"{kind:24} {prec:5} {value['n']:4d} {value['ms']:7.3f} {value['ms'] / total * 100:5.1f}")
    print(f"-- 가장 느린 층 {top}개 --")
    for row in result["slowest"][:top]:
        print(f"  {row['ms']:6.3f} ms  {row['type']:18} {row['precision']:5} {row['name'][:88]}")


def print_diff(base_tag: str, base: dict, tag: str, other: dict) -> None:
    print(f"\n=== 종류별 차이 ({tag} − {base_tag}, ms) ===")
    keys = set(base["by_kind"]) | set(other["by_kind"])
    rows = []
    for key in keys:
        a = base["by_kind"].get(key, {"n": 0, "ms": 0.0})
        b = other["by_kind"].get(key, {"n": 0, "ms": 0.0})
        rows.append((b["ms"] - a["ms"], key, a, b))
    for diff, key, a, b in sorted(rows, key=lambda r: -abs(r[0])):
        if abs(diff) > 0.005:
            print(f"{key:30} {base_tag} n={a['n']:2d} {a['ms']:6.3f} | {tag} n={b['n']:2d} {b['ms']:6.3f} | {diff:+.3f}")
    print(f"전체: {base['total_ms']:.3f} → {other['total_ms']:.3f} ms ({other['total_ms'] - base['total_ms']:+.3f})")


def main():
    parser = argparse.ArgumentParser(description="TensorRT 엔진 층별 프로파일")
    parser.add_argument("engines", nargs="+", help="이름=경로 (여러 개면 첫 것이 비교 기준)")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--top", type=int, default=8, help="가장 느린 층 몇 개를 보일지")
    parser.add_argument("--json", type=Path, default=None, help="결과 저장 경로")
    args = parser.parse_args()

    results = {}
    for item in args.engines:
        tag, _, path = item.partition("=") if "=" in item else (Path(item).stem, "", item)
        results[tag] = profile(Path(path), args.iters)
        print_report(tag, results[tag], args.top)
    tags = list(results)
    for tag in tags[1:]:
        print_diff(tags[0], results[tags[0]], tag, results[tag])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[done] {args.json}")


if __name__ == "__main__":
    main()
