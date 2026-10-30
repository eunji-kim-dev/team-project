"""CUDA 메모리·스트림 얇은 래퍼 (segmenter, build_engine 이 씀)
- cuda-python 먼저 찾고 없으면 pycuda 씀. 둘 중 하나만 설치하면 됨
- DeviceBuffer: GPU 메모리 / pinned_empty: 고정 호스트 배열 / Stream: CUDA 스트림 / memcpy_*: 복사
"""
from __future__ import annotations

import ctypes
import weakref

import numpy as np

BACKEND = None   # "cuda-python" 또는 "pycuda"
try:
    try:
        from cuda.bindings import runtime as cudart   # cuda-python 12.6.1 이상
    except ImportError:
        from cuda import cudart                       # 그 이전 버전
    BACKEND = "cuda-python"
except ImportError:
    try:
        import pycuda.autoinit  # noqa: F401  # import 자체가 CUDA 컨텍스트 만듦
        import pycuda.driver as pycu
        BACKEND = "pycuda"
    except ImportError as e:
        raise ImportError("cuda-python 또는 pycuda 필요: pip install cuda-python") from e


def _check(err):
    # cuda-python 은 에러 코드를 반환값으로 줌. pycuda 는 예외를 던지므로 할 일 없음
    if BACKEND != "cuda-python":
        return
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA 오류: {err}")


class DeviceBuffer:
    """GPU 메모리 한 덩어리. free() 안 불러도 객체 사라질 때 해제됨"""

    def __init__(self, nbytes: int):
        self.nbytes = int(nbytes)
        if BACKEND == "cuda-python":
            err, ptr = cudart.cudaMalloc(self.nbytes)
            _check(err)
            self.ptr = int(ptr)          # GPU 주소. TensorRT 에 넘기는 값
            self._alloc = None
        else:
            self._alloc = pycu.mem_alloc(self.nbytes)
            self.ptr = int(self._alloc)
        self._freed = False

    def free(self):
        if self._freed:
            return
        self._freed = True
        if BACKEND == "cuda-python":
            cudart.cudaFree(self.ptr)
        else:
            self._alloc.free()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


def pinned_empty(shape, dtype) -> np.ndarray:
    """페이지 고정 호스트 배열. GPU 와 복사가 빠름. 쓰는 법은 np.empty 와 같음"""
    shape = tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    if BACKEND == "pycuda":
        return pycu.pagelocked_empty(shape, dtype)

    nbytes = int(np.prod(shape)) * dtype.itemsize
    err, ptr = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)
    _check(err)
    raw = (ctypes.c_byte * nbytes).from_address(int(ptr))
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    finalizer = weakref.finalize(array, cudart.cudaFreeHost, int(ptr))   # 배열 사라질 때 같이 해제
    finalizer.atexit = False
    return array


class Stream:
    def __init__(self):
        if BACKEND == "cuda-python":
            err, stream = cudart.cudaStreamCreate()
            _check(err)
            self._stream = stream
            self.handle = int(stream)    # execute_async_v3 에 넘기는 값
        else:
            self._stream = pycu.Stream()
            self.handle = int(self._stream.handle)

    def synchronize(self):
        if BACKEND == "cuda-python":
            _check(cudart.cudaStreamSynchronize(self._stream))
        else:
            self._stream.synchronize()


def memcpy_h2d_async(dst_ptr: int, src: np.ndarray, stream: Stream):
    """호스트 → GPU (비동기). src 는 C-contiguous 여야 함"""
    if BACKEND == "cuda-python":
        _check(cudart.cudaMemcpyAsync(int(dst_ptr), src.ctypes.data, src.nbytes,
                                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream._stream))
    else:
        pycu.memcpy_htod_async(int(dst_ptr), src, stream._stream)


def memcpy_d2h_async(dst: np.ndarray, src_ptr: int, stream: Stream):
    """GPU → 호스트 (비동기). stream.synchronize() 뒤에 dst 읽을 것"""
    if BACKEND == "cuda-python":
        _check(cudart.cudaMemcpyAsync(dst.ctypes.data, int(src_ptr), dst.nbytes,
                                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream._stream))
    else:
        pycu.memcpy_dtoh_async(dst, int(src_ptr), stream._stream)


def memcpy_h2d(dst_ptr: int, src: np.ndarray):
    """호스트 → GPU (동기). 캘리브레이션처럼 속도 안 중요한 곳에 씀"""
    if BACKEND == "cuda-python":
        _check(cudart.cudaMemcpy(int(dst_ptr), src.ctypes.data, src.nbytes,
                                 cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    else:
        pycu.memcpy_htod(int(dst_ptr), src)
