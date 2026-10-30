"""Jetson TensorRT 10+ 추론. PyTorch 불필요, cuda-python 런타임으로 버퍼/스트림 재사용."""
import ctypes
from pathlib import Path
import threading
import warnings
import numpy as np


def cuda_runtime():
    # [추가] CUDA Python의 신규/구버전 namespace 지원. pycuda.autoinit는 사용하지 않는다.
    try:
        from cuda.bindings import runtime
    except ImportError:
        try:
            from cuda import cudart as runtime
        except ImportError as exc:
            raise RuntimeError("CUDA Python missing: install cuda-python matching Jetson CUDA (13.2 for this board)") from exc
    return runtime


def checked(result):
    error,*values=result
    if int(error)!=0:
        raise RuntimeError(f"CUDA failed: {error}")
    return values[0] if len(values)==1 else tuple(values)


class TensorRTBackend:
    def __init__(self,path,spec):
        # [추가] 생성은 UI 스레드, 추론은 worker여도 같은 CUDA primary context를 사용.
        # cudaSetDevice를 호출 스레드마다 수행하고 객체 동시 사용은 lock으로 금지한다.
        self.lock=threading.Lock();self.cuda=None;self.stream=None
        self.context=None;self.engine=None;self.runtime=None
        self.host_ptrs=[];self.device_ptrs=[];self.arrays=[];self.closed=False
        try:
            try:
                import tensorrt as trt
            except ImportError as exc:
                raise RuntimeError("TensorRT missing: use Jetson's TensorRT Python environment, not the Windows CPU Python") from exc
            if int(trt.__version__.split('.')[0])<10:
                raise RuntimeError("TensorRT 10+ required; provided engines were built with 10.16.2.10")
            self.cuda=cuda_runtime()
            checked(self.cuda.cudaSetDevice(0))
            errors=[]

            class Logger(trt.ILogger):
                def __init__(self):super().__init__()
                def log(self,severity,message):
                    if severity<=trt.ILogger.Severity.WARNING:
                        errors.append(str(message));del errors[:-8]

            self.logger=Logger()  # TensorRT 객체보다 오래 유지해야 한다.
            trt.init_libnvinfer_plugins(self.logger,"")
            self.runtime=trt.Runtime(self.logger)
            self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if self.engine is None:
                raise RuntimeError("Engine load failed (GPU/platform/TensorRT compatibility). "
                    f"Runtime={trt.__version__}; rebuild on this Jetson if incompatible. "+" | ".join(errors)[-1800:])
            self.context=self.engine.create_execution_context()
            if self.context is None:raise RuntimeError("TensorRT execution context creation failed (GPU memory?)")
            names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs=[n for n in names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
            outputs=[n for n in names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT]
            if len(inputs)!=1 or len(outputs)!=1:
                raise ValueError(f"Expected one input and one output; got {inputs}/{outputs}")
            self.input_name,self.output_name=inputs[0],outputs[0]
            h,w=spec["input"]["height"],spec["input"]["width"]
            expected=(1,3,h,w)
            if -1 in self.engine.get_tensor_shape(self.input_name):
                if not self.context.set_input_shape(self.input_name,expected):
                    raise ValueError(f"Engine optimization profile does not accept {expected}")
            self.input_shape=tuple(self.context.get_tensor_shape(self.input_name))
            self.output_shape=tuple(self.context.get_tensor_shape(self.output_name))
            self.input_dtype=np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
            self.output_dtype=np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
            if self.input_shape!=expected or self.input_dtype not in (np.dtype('float32'),np.dtype('float16')):
                raise ValueError(f"Expected float NCHW {expected}; engine input={self.input_shape}/{self.input_dtype}")
            kind=spec['engine']['output_kind']
            if kind=='labels':
                valid=self.output_shape in ((1,h,w),(1,1,h,w)) and self.output_dtype in (np.dtype('int32'),np.dtype('int64'))
            else:
                valid=self.output_shape==expected and self.output_dtype in (np.dtype('float32'),np.dtype('float16'))
            if not valid:
                raise ValueError(f"Engine output {self.output_shape}/{self.output_dtype} disagrees with standard JSON output_kind={kind}")
            # 벡터 패킹/host I/O/비연속 텐서는 단순 복사를 적용하지 않고 명확히 거부한다.
            for name,shape in ((self.input_name,self.input_shape),(self.output_name,self.output_shape)):
                if self.engine.get_tensor_location(name)!=trt.TensorLocation.DEVICE or self.engine.get_tensor_format(name)!=trt.TensorFormat.LINEAR:
                    raise ValueError(f"Only device LINEAR tensor I/O is supported: {name}")
                stride=1;expected_strides=[]
                for size in reversed(shape):expected_strides.insert(0,stride);stride*=size
                if tuple(self.context.get_tensor_strides(name))!=tuple(expected_strides):
                    raise ValueError(f"Noncontiguous engine tensor: {name}")
            self.stream=checked(self.cuda.cudaStreamCreate())
            # 고정 호스트 메모리 + GPU 메모리를 시작할 때 한 번 할당. 매 프레임 재할당 없음.
            for name,shape,dtype in ((self.input_name,self.input_shape,self.input_dtype),
                                     (self.output_name,self.output_shape,self.output_dtype)):
                size=int(np.prod(shape))*dtype.itemsize
                host=int(checked(self.cuda.cudaHostAlloc(size,0)));self.host_ptrs.append(host)
                raw=(ctypes.c_byte*size).from_address(host)
                array=np.frombuffer(raw,dtype=dtype).reshape(shape);self.arrays.append(array)
                device=int(checked(self.cuda.cudaMalloc(size)));self.device_ptrs.append(device)
                if not self.context.set_tensor_address(name,device):
                    raise RuntimeError(f"Failed to bind engine tensor: {name}")
            self.info={"backend":"tensorrt","device":0,"tensorrt_version":trt.__version__,
                "cuda_runtime_version":int(checked(self.cuda.cudaRuntimeGetVersion())),
                "input_name":self.input_name,"input_shape":self.input_shape,"input_dtype":str(self.input_dtype),
                "output_name":self.output_name,"output_shape":self.output_shape,"output_dtype":str(self.output_dtype),
                "normalize_in_graph":spec['engine']['normalize_in_graph'],"output_kind":kind}
        except Exception:
            self.close()
            raise

    def __call__(self,array):
        with self.lock:
            if self.closed:raise RuntimeError("TensorRT backend is closed")
            if array.shape!=self.input_shape:raise ValueError(f"Input shape changed: {array.shape}")
            checked(self.cuda.cudaSetDevice(0))
            host_in,host_out=self.arrays
            np.copyto(host_in,array,casting='same_kind')
            checked(self.cuda.cudaMemcpyAsync(self.device_ptrs[0],host_in.ctypes.data,host_in.nbytes,
                self.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,self.stream))
            if not self.context.execute_async_v3(stream_handle=int(self.stream)):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            checked(self.cuda.cudaMemcpyAsync(host_out.ctypes.data,self.device_ptrs[1],host_out.nbytes,
                self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,self.stream))
            checked(self.cuda.cudaStreamSynchronize(self.stream))
            # 다음 프레임과 독립적인 결과를 반환. FP16 logits는 기존 후처리용 FP32로 변환.
            dtype=np.float32 if np.issubdtype(self.output_dtype,np.floating) else self.output_dtype
            return host_out.astype(dtype,copy=True)

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
            failures=[]
            def release(fn,*args):
                try:checked(fn(*args))
                except Exception as exc:failures.append(str(exc))
            if self.cuda is not None:
                release(self.cuda.cudaSetDevice,0)
                if self.stream is not None:release(self.cuda.cudaStreamSynchronize,self.stream)
            self.context=None;self.engine=None;self.runtime=None
            self.arrays.clear()
            for ptr in self.device_ptrs:release(self.cuda.cudaFree,ptr)
            for ptr in self.host_ptrs:release(self.cuda.cudaFreeHost,ptr)
            self.device_ptrs.clear();self.host_ptrs.clear()
            if self.stream is not None:release(self.cuda.cudaStreamDestroy,self.stream)
            self.stream=None
            if failures:warnings.warn("CUDA cleanup: "+"; ".join(failures),RuntimeWarning)

    def __del__(self):
        try:self.close()
        except Exception:pass
