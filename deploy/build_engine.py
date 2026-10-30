"""ONNX → TensorRT 엔진 (GPU FP32 / FP16 / PTQ-INT8 / QAT-INT8). Jetson 에서 실행

# 일반 ONNX → fp32, fp16, int8(PTQ)
python -m deploy.build_engine --onnx models/fast_scnn.onnx --precision fp32 fp16 int8 \\
    --calib-list calib_list.txt --check-image data/images/0001_color.png
# QAT ONNX (Q/DQ 포함) → int8. 캘리브레이션 불필요
python -m deploy.build_engine --onnx models/fast_scnn_qat.onnx --precision qat

정밀도
  fp32  Orin 은 FP32 를 기본으로 TF32(가수 10비트)로 돌리므로 꺼서 진짜 FP32 로 만듦 (--allow-tf32 로 되돌림)
  fp16  반정밀도. INT8 로 못 바꾸는 층의 기본값
  int8  PTQ. calib_list.txt 사진을 흘려 층별 값 범위를 재서 TensorRT 가 scale 정함
  qat   ONNX 의 Q/DQ 가 scale 을 이미 들고 있음. TensorRT 는 그대로 따름

산출물: models/fast_scnn_{fp32,fp16,int8}.engine, fast_scnn_qat_int8.engine (+ 각 .json)
        fast_scnn_int8_entropy.calib (캘리브레이션 캐시. 사진 목록 바뀌면 --recalibrate), timing.cache
엔진은 만든 보드·TensorRT 버전 전용. PC 에서 만들어 옮기면 안 됨
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.preprocess import ModelSpec, load_spec, make_blob, read_image_list  # noqa: E402
from deploy import cuda_util as cu  # noqa: E402

TRT_MAJOR = int(trt.__version__.split(".")[0])
CHECK_LOOPS = 30   # --check-image 때 시간 재는 반복 횟수


# ---------- INT8 캘리브레이터 ----------

def make_calibrator(kind: str, images: list[Path], spec: ModelSpec, cache_file: Path):
    """kind: entropy(기본, 세그멘테이션에 보통 이걸 씀) / minmax"""
    base_class = trt.IInt8MinMaxCalibrator if kind == "minmax" else trt.IInt8EntropyCalibrator2

    class ImageCalibrator(base_class):
        """사진을 한 장씩(ONNX 가 batch 1 고정) GPU 에 올려 TensorRT 에 넘김. 전처리는 추론과 같은 make_blob"""

        def __init__(self):
            super().__init__()
            self.images = images
            self.index = 0
            nbytes = 1 * 3 * spec.height * spec.width * 4   # float32 한 장
            self.dev_buffer = cu.DeviceBuffer(nbytes)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.index >= len(self.images):
                return None   # 끝
            path = self.images[self.index]
            self.index += 1
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"캘리브레이션 사진을 못 읽음: {path}")
            cu.memcpy_h2d(self.dev_buffer.ptr, make_blob(bgr, spec))
            if self.index % 20 == 0 or self.index == len(self.images):
                print(f"  [calib] {self.index}/{len(self.images)}")
            return [int(self.dev_buffer.ptr)]

        def read_calibration_cache(self):
            if cache_file.exists():
                print(f"  [calib] 캐시 재사용: {cache_file} (다시 재려면 --recalibrate)")
                return cache_file.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            cache_file.write_bytes(cache)
            print(f"  [calib] 캐시 저장: {cache_file}")

    return ImageCalibrator()


# ---------- 빌더 ----------

def onnx_has_qdq(onnx_path: Path) -> bool:
    """QuantizeLinear 노드가 있으면 QAT(명시적 양자화) 그래프"""
    try:
        import onnx
        return any(node.op_type == "QuantizeLinear" for node in onnx.load(str(onnx_path)).graph.node)
    except ImportError:   # onnx 패키지 없으면 바이트에서 찾음
        return b"QuantizeLinear" in onnx_path.read_bytes()


class EngineBuilder:
    """ONNX 하나로 여러 정밀도의 엔진을 만듦. build(precision) 이 엔진 파일 + 빌드 정보 dict 를 돌려줌"""

    def __init__(self, onnx_path: Path, out_dir: Path, workspace_mb: int, timing_cache: Path | None,
                 allow_tf32: bool, logger):
        self.onnx_path = onnx_path
        self.out_dir = out_dir
        self.workspace_mb = workspace_mb
        self.timing_cache_path = timing_cache
        self.allow_tf32 = allow_tf32
        self.logger = logger
        self.stem = onnx_path.stem   # fast_scnn / fast_scnn_qat

    def engine_path(self, precision: str) -> Path:
        if precision == "qat":
            label = "int8" if "qat" in self.stem.lower() else "qat_int8"   # fast_scnn_qat → fast_scnn_qat_int8
        else:
            label = precision
        return self.out_dir / f"{self.stem}_{label}.engine"

    def _parse_onnx(self, network, parser):
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("  [onnx parser]", parser.get_error(i))
                raise RuntimeError("ONNX 파싱 실패")
        inp, out = network.get_input(0), network.get_output(0)
        print(f"  입력 {inp.name} {tuple(inp.shape)}  출력 {out.name} {tuple(out.shape)}")

    def _configure(self, builder, config, precision: str, calibrator):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(self.workspace_mb) << 20)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED   # 나중에 층별 정밀도 확인용

        if precision == "fp32" and not self.allow_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)   # Ampere 기본 TF32 끔 → 진짜 FP32
        if precision in ("fp16", "int8", "qat"):
            config.set_flag(trt.BuilderFlag.FP16)     # INT8 로 못 바꾸는 층은 FP16 으로
        if precision in ("int8", "qat"):
            if not builder.platform_has_fast_int8:
                print("  [warn] 이 GPU 는 INT8 가속이 없다고 나옴 (Orin 은 있어야 정상)")
            config.set_flag(trt.BuilderFlag.INT8)
            if precision == "int8":
                config.int8_calibrator = calibrator   # qat 는 Q/DQ 가 scale 을 들고 있어 캘리브레이터 없음

    def build(self, precision: str, calibrator=None) -> tuple[Path, dict]:
        builder = trt.Builder(self.logger)
        flags = 0
        if TRT_MAJOR < 10:   # 8.x 는 explicit batch 플래그 필요. 10.x 는 항상 explicit
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, self.logger)
        self._parse_onnx(network, parser)

        config = builder.create_builder_config()
        self._configure(builder, config, precision, calibrator)

        timing_cache = None
        if self.timing_cache_path is not None:
            data = self.timing_cache_path.read_bytes() if self.timing_cache_path.exists() else b""
            timing_cache = config.create_timing_cache(data)
            config.set_timing_cache(timing_cache, ignore_mismatch=False)

        started = time.time()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("엔진 빌드 실패 (위 로그 참고)")
        seconds = time.time() - started

        out_path = self.engine_path(precision)
        with open(out_path, "wb") as f:
            f.write(serialized)
        if timing_cache is not None:
            with open(self.timing_cache_path, "wb") as f:
                f.write(timing_cache.serialize())

        info = {"precision": precision, "build_seconds": round(seconds, 1),
                "engine_size_mb": round(out_path.stat().st_size / 1e6, 2),
                "trt_version": trt.__version__, "workspace_mb": self.workspace_mb,
                "tf32": self.allow_tf32 if precision == "fp32" else None}
        print(f"  빌드 {seconds:.0f}s → {out_path} ({info['engine_size_mb']} MB)")
        return out_path, info

    def layer_precision_summary(self, engine_path: Path) -> dict:
        """엔진 안의 층들이 실제로 어떤 정밀도로 도는지 셈 (INT8 로 못 바뀐 층이 얼마나 되나)"""
        try:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
            inspector = engine.create_engine_inspector()
            info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
            counts = {}
            for layer in info.get("Layers", []):
                outputs = layer.get("Outputs", [])
                fmt = outputs[0].get("Format/Datatype", "?") if outputs else "?"
                if "Int8" in fmt or "INT8" in fmt:
                    key = "INT8"
                elif "Half" in fmt or "FP16" in fmt:
                    key = "FP16"
                elif "Float" in fmt or "FP32" in fmt:
                    key = "FP32"
                elif "Int32" in fmt:
                    key = "INT32"
                else:
                    key = fmt
                counts[key] = counts.get(key, 0) + 1
            counts["total_layers"] = len(info.get("Layers", []))
            print(f"  층 정밀도: {counts}")
            return counts
        except Exception as e:   # 형식이 버전마다 달라 실패해도 빌드는 성공한 것
            print(f"  (층 정밀도 요약 실패: {e})")
            return {}


def check_engine(engine_path: Path, image: Path, onnx_path: Path) -> dict:
    """엔진으로 사진 한 장 돌려 보고, onnxruntime 있으면 ONNX 결과와 마스크 일치율 비교"""
    from deploy.segmenter import TrtSegmenter, OnnxSegmenter
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image)

    result = {}
    with TrtSegmenter(engine_path) as seg:
        mask, _ = seg.segment(bgr, 0)
        infer_times = []
        for i in range(CHECK_LOOPS):
            seg.segment(bgr, i)
            infer_times.append(seg.last_timing["infer_ms"])
        result["infer_ms_p50"] = round(float(np.percentile(infer_times, 50)), 2)
        result["class_ratio"] = [round(float((mask == c).mean()), 4) for c in range(seg.spec.num_classes)]
        print(f"  확인: infer p50 {result['infer_ms_p50']} ms, 클래스 비율 {result['class_ratio']}")

    try:
        with OnnxSegmenter(onnx_path) as ref:
            ref_mask, _ = ref.segment(bgr, 0)
        agreement = float((ref_mask == mask).mean()) * 100.0
        result["agree_vs_onnxruntime_pct"] = round(agreement, 3)
        print(f"  확인: onnxruntime 마스크와 일치 {agreement:.3f}%")
    except ImportError:
        print("  (onnxruntime 없음 → ONNX 와 비교는 건너뜀)")
    return result


# ---------- main ----------

def parse_args():
    parser = argparse.ArgumentParser(description="ONNX → TensorRT 엔진")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--precision", nargs="+", default=["fp32", "fp16", "int8"],
                        choices=["fp32", "fp16", "int8", "qat"],
                        help="qat 는 qat_train.py 가 만든 Q/DQ ONNX 에만 (int8 은 Q/DQ 없는 ONNX 에만)")
    parser.add_argument("--allow-tf32", action="store_true", help="fp32 엔진에서 TF32 허용 (기본은 끔)")
    parser.add_argument("--out-dir", type=Path, default=None, help="기본: ONNX 가 있는 폴더")
    parser.add_argument("--spec", type=Path, default=None, help="기본: <onnx>.json")
    parser.add_argument("--calib-list", type=Path, help="INT8 용 사진 목록 (calib_list.txt)")
    parser.add_argument("--calib-type", default="entropy", choices=["entropy", "minmax"])
    parser.add_argument("--calib-max", type=int, default=0, help="캘리브레이션 사진 수 제한 (0 = 전부)")
    parser.add_argument("--recalibrate", action="store_true", help="캘리브레이션 캐시 지우고 다시 잼")
    parser.add_argument("--workspace", type=int, default=2048, help="빌드 작업 메모리 상한 MB")
    parser.add_argument("--no-timing-cache", action="store_true")
    parser.add_argument("--check-image", type=Path, help="빌드 후 이 사진으로 동작 확인")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = load_spec(args.spec or args.onnx.with_suffix(args.onnx.suffix + ".json"))
    out_dir = args.out_dir or args.onnx.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    print(f"[env] TensorRT {trt.__version__}, CUDA 백엔드 {cu.BACKEND}")
    print(f"[spec] {spec.height}x{spec.width} {spec.color_order}, normalize_in_graph={spec.normalize_in_graph}")

    has_qdq = onnx_has_qdq(args.onnx)
    print(f"[onnx] Q/DQ 노드 {'있음 → qat 전용' if has_qdq else '없음 → fp32/fp16/int8 용'}")
    if has_qdq and "int8" in args.precision:
        raise SystemExit("Q/DQ 가 든 ONNX 에는 --precision int8(PTQ) 대신 qat 를 쓸 것")
    if not has_qdq and "qat" in args.precision:
        raise SystemExit("이 ONNX 에는 Q/DQ 노드가 없음 → qat_train.py 로 만든 ONNX 를 줄 것")

    timing_cache = None if args.no_timing_cache else out_dir / "timing.cache"
    builder = EngineBuilder(args.onnx, out_dir, args.workspace, timing_cache, args.allow_tf32, logger)

    for precision in args.precision:
        print(f"\n=== {precision.upper()} ===")
        calibrator = None
        calib_info = {}
        if precision == "int8":
            if not args.calib_list:
                raise SystemExit("INT8 에는 --calib-list 가 필요함")
            images = read_image_list(args.calib_list)
            if args.calib_max > 0:
                images = images[:args.calib_max]
            cache_file = out_dir / f"{builder.stem}_int8_{args.calib_type}.calib"
            if args.recalibrate and cache_file.exists():
                cache_file.unlink()
            print(f"  캘리브레이션 사진 {len(images)}장 ({args.calib_type})")
            calibrator = make_calibrator(args.calib_type, images, spec, cache_file)
            calib_info = {"calib_type": args.calib_type, "calib_images": len(images),
                          "calib_list": str(args.calib_list), "calib_cache": str(cache_file)}

        engine_path, info = builder.build(precision, calibrator)
        info["layer_precisions"] = builder.layer_precision_summary(engine_path)
        info.update(calib_info)

        # 매니페스트: ONNX 것 + 빌드 정보 → segmenter 가 읽음. check_engine 도 이걸 읽으므로 먼저 씀
        def write_manifest():
            manifest = spec.to_dict()
            manifest.update(source_onnx=str(args.onnx), built_at=time.strftime("%Y-%m-%d %H:%M:%S"), build=info)
            manifest_path = engine_path.with_suffix(engine_path.suffix + ".json")
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        write_manifest()
        if args.check_image:
            info["check"] = check_engine(engine_path, args.check_image, args.onnx)
            write_manifest()

        del calibrator   # INT8 끝나면 GPU 버퍼 해제

    print("\n[done]", out_dir)


if __name__ == "__main__":
    main()
