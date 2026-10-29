"""추론 모듈. 역할 4 가 import 해서 씀

    from deploy.segmenter import create_segmenter
    seg = create_segmenter("models/fast_scnn_fp16.engine")   # 확장자 보고 onnxruntime / TensorRT 자동 선택
    mask, frame_id = seg.segment(bgr, frame_id)              # mask: (H,W) uint8, 0 floor / 1 obstacle / 2 other
    seg.last_timing                                          # {"pre_ms", "infer_ms", "post_ms", "total_ms"}

- OnnxSegmenter: onnxruntime (PC 개발용 + Jetson 의 CPU FP32 비교 행)
- TrtSegmenter : TensorRT 엔진 (Jetson 실제 추론)
- 모델 옆의 <모델>.json 을 읽어 입력 크기·채널 순서·정규화/argmax 포함 여부를 알아냄
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.preprocess import ModelSpec, load_spec, make_blob  # noqa: E402

ENGINE_SUFFIXES = (".engine", ".trt", ".plan")   # TensorRT 엔진으로 보는 확장자


def _find_spec(model_path: Path, spec_path) -> ModelSpec:
    # spec_path 없으면 모델 옆의 <모델>.json
    if spec_path is None:
        candidate = model_path.with_suffix(model_path.suffix + ".json")
        if not candidate.exists():
            raise FileNotFoundError(f"{candidate} 가 없음. 매니페스트를 모델 옆에 두거나 spec_path 로 preprocess.json 을 줄 것")
        spec_path = candidate
    return load_spec(spec_path)


class BaseSegmenter:
    """공통: 전처리 → run() → 후처리 + 시간 기록. run() 만 자식이 구현"""

    def __init__(self, model_path, spec_path=None):
        self.model_path = Path(model_path)
        self.spec = _find_spec(self.model_path, spec_path)
        self.last_timing = {"pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0, "total_ms": 0.0}
        self.out_is_mask = None   # 출력이 (1,H,W) 마스크인지 (1,C,H,W) 로짓인지. 첫 프레임에 판단

    def run(self, blob: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        if self.out_is_mask is None:
            self.out_is_mask = output.ndim == 3 or (output.ndim == 4 and output.shape[1] == 1)

        if self.out_is_mask:
            mask = output.reshape(output.shape[-2], output.shape[-1]).astype(np.uint8)   # int32 → uint8
        else:
            mask = np.argmax(output[0], axis=0).astype(np.uint8)                          # 로짓이면 여기서 argmax

        if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def segment(self, bgr: np.ndarray, frame_id: int):
        t_start = time.perf_counter()
        blob = make_blob(bgr, self.spec)
        t_pre = time.perf_counter()
        output = self.run(blob)
        t_infer = time.perf_counter()
        mask = self._postprocess(output, bgr.shape[0], bgr.shape[1])
        t_post = time.perf_counter()

        self.last_timing = {
            "pre_ms": (t_pre - t_start) * 1e3,
            "infer_ms": (t_infer - t_pre) * 1e3,
            "post_ms": (t_post - t_infer) * 1e3,
            "total_ms": (t_post - t_start) * 1e3,
        }
        return mask, frame_id

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class OnnxSegmenter(BaseSegmenter):
    """onnxruntime. providers 안 주면 CUDA 있으면 CUDA, 없으면 CPU"""

    def __init__(self, model_path, spec_path=None, providers=None, threads: int | None = None):
        super().__init__(model_path, spec_path)
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]

        self.session = ort.InferenceSession(str(self.model_path), options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        if tuple(input_shape[-2:]) != (self.spec.height, self.spec.width):
            raise ValueError(f"ONNX 입력 {input_shape} ≠ spec {self.spec.height}x{self.spec.width}")
        self.providers = self.session.get_providers()

    def run(self, blob):
        return self.session.run(None, {self.input_name: blob})[0]


class TrtSegmenter(BaseSegmenter):
    """TensorRT. 엔진은 같은 보드·같은 TensorRT 버전에서 만든 것이어야 함"""

    def __init__(self, model_path, spec_path=None, verbose: bool = False):
        super().__init__(model_path, spec_path)
        import tensorrt as trt
        from deploy import cuda_util as cu
        self.trt, self.cu = trt, cu

        # 엔진 로드
        self.logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        with open(self.model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"엔진 로드 실패: {self.model_path} (다른 보드/TensorRT 버전에서 만든 엔진?)")
        self.context = self.engine.create_execution_context()

        # 입출력 텐서 (TensorRT 8.5+ 이름 기반 API)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"입력 1개·출력 1개 엔진만 지원함 (입력 {inputs}, 출력 {outputs})")
        self.in_name, self.out_name = inputs[0], outputs[0]
        self.in_shape = tuple(int(v) for v in self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(int(v) for v in self.engine.get_tensor_shape(self.out_name))
        self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out_name))
        if self.in_shape[-2:] != (self.spec.height, self.spec.width):
            raise ValueError(f"엔진 입력 {self.in_shape} ≠ spec {self.spec.height}x{self.spec.width}")

        # 버퍼는 한 번만 만들고 매 프레임 재사용
        self.host_in = cu.pinned_empty(self.in_shape, self.in_dtype)
        self.host_out = cu.pinned_empty(self.out_shape, out_dtype)
        self.dev_in = cu.DeviceBuffer(self.host_in.nbytes)
        self.dev_out = cu.DeviceBuffer(self.host_out.nbytes)
        self.stream = cu.Stream()
        self.context.set_tensor_address(self.in_name, self.dev_in.ptr)
        self.context.set_tensor_address(self.out_name, self.dev_out.ptr)

    @property
    def device_memory_bytes(self) -> int:
        """엔진 활성화 메모리(바이트). 성능표의 '활성화 MB'"""
        engine = self.engine
        return int(getattr(engine, "device_memory_size_v2", None) or getattr(engine, "device_memory_size", 0))

    def run(self, blob):
        if blob.dtype != self.in_dtype:
            blob = blob.astype(self.in_dtype)
        np.copyto(self.host_in, blob.reshape(self.in_shape))
        self.cu.memcpy_h2d_async(self.dev_in.ptr, self.host_in, self.stream)      # 호스트 → GPU
        if not self.context.execute_async_v3(self.stream.handle):                 # 추론
            raise RuntimeError("TensorRT execute_async_v3 실패")
        self.cu.memcpy_d2h_async(self.host_out, self.dev_out.ptr, self.stream)    # GPU → 호스트
        self.stream.synchronize()
        return self.host_out   # 다음 run() 때 덮어써지는 버퍼. 보관하려면 copy()

    def close(self):
        self.context = None   # 버퍼 지우기 전에 컨텍스트부터 놓음
        for name in ("dev_in", "dev_out"):
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer.free()
        self.engine = None


def create_segmenter(model_path, spec_path=None, **kwargs) -> BaseSegmenter:
    """.onnx → OnnxSegmenter, .engine/.trt/.plan → TrtSegmenter"""
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return OnnxSegmenter(model_path, spec_path, **kwargs)
    if suffix in ENGINE_SUFFIXES:
        return TrtSegmenter(model_path, spec_path, **kwargs)
    raise ValueError(f"모르는 모델 확장자: {suffix}")


# ---------- 단독 실행: 사진 한 장 테스트 ----------

PALETTE_BGR = np.array([[0, 200, 0], [0, 0, 220], [128, 128, 128]], dtype=np.uint8)   # floor 초록 / obstacle 빨강 / other 회색


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="사진 한 장으로 segmenter 동작 확인")
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("--spec", default=None)
    parser.add_argument("--out", default=None, help="색칠한 결과 png 저장 경로")
    parser.add_argument("--loops", type=int, default=20, help="시간 측정 반복 횟수")
    args = parser.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"이미지를 못 읽음: {args.image}")

    with create_segmenter(args.model, args.spec) as seg:
        mask, _ = seg.segment(bgr, 0)
        total_times = []
        for i in range(args.loops):
            seg.segment(bgr, i)
            total_times.append(seg.last_timing["total_ms"])

        ratios = ", ".join(f"{c}: {(mask == c).mean() * 100:.1f}%" for c in range(seg.spec.num_classes))
        timing = seg.last_timing
        print(f"mask {mask.shape} {mask.dtype} 값 {np.unique(mask).tolist()}  클래스별 픽셀 비율: {ratios}")
        print(f"마지막 프레임  pre {timing['pre_ms']:.2f} ms / infer {timing['infer_ms']:.2f} ms / post {timing['post_ms']:.2f} ms")
        print(f"{args.loops}회 total p50 {np.percentile(total_times, 50):.2f} ms, p95 {np.percentile(total_times, 95):.2f} ms")

        if args.out:
            color = PALETTE_BGR[np.clip(mask, 0, 2)]
            cv2.imwrite(args.out, cv2.addWeighted(bgr, 0.5, color, 0.5, 0))
            print(f"저장: {args.out}")


if __name__ == "__main__":
    _main()
