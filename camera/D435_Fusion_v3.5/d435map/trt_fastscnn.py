"""[Jetson] TensorRT 엔진 추론 백엔드. 저장소의 deploy/segmenter.py(TrtSegmenter)를 그대로 재사용한다.
- models/ 에 <이름>.engine 과 deploy 가 만든 <이름>.engine.json(매니페스트)을 함께 둔다.
- 앱은 softmax 신뢰도를 fusion 판정에 쓰므로 엔진 출력은 float 로짓 [1,3,H,W] 이어야 한다.
  argmax 가 그래프에 든 엔진(출력 (1,H,W) int32 mask)은 거부한다:
      python -m deploy.export_onnx --keep-logits ... && python -m deploy.build_engine ...
- normalize_in_graph=true 엔진은 0~255 입력을 받으므로 semantics.prepare_input 의 CPU 정규화를 건너뛴다.
- decode_in_graph=true 엔진(deploy.add_decode 로 만든 것)은 labels int32 [1,H,W] + confidence float32 [1,H,W] 를 내고,
  앱은 CPU softmax 디코드(프레임당 약 15 ms)를 건너뛴다. __call__ 이 (labels uint8, confidence float32) 튜플을 돌려준다.
- 엔진은 주 스레드에서 만들고 처리 스레드에서 실행한다. cuda-python 런타임 API는 프로세스 공용
  primary context 를 쓰므로 스레드가 달라도 된다(pycuda.autoinit 로 떨어지면 스레드 종속이라 실패할 수 있음).
"""
from pathlib import Path
import json
import sys
import numpy as np

def _import_trt_segmenter():
    # 저장소 루트(team-project)에서 실행하지 않아도 되도록 camera/D435_Fusion_v3.5 기준으로 deploy 를 찾는다.
    try:
        from deploy.segmenter import TrtSegmenter
        return TrtSegmenter
    except ModuleNotFoundError:
        repo=Path(__file__).resolve().parents[3]
        if not (repo/"deploy"/"segmenter.py").exists():
            raise ModuleNotFoundError("deploy/segmenter.py not found; run inside the team-project checkout or add it to PYTHONPATH")
        if str(repo) not in sys.path:sys.path.insert(0,str(repo))
        from deploy.segmenter import TrtSegmenter
        return TrtSegmenter

def _check_manifest(manifest,spec,path):
    # 앱 표준 JSON(standard_fastscnn.json)과 엔진 매니페스트의 입력 사양이 다르면 조용히 틀린 마스크가 나오므로 막는다.
    inp=spec["input"]
    size=manifest.get("input_size")
    if isinstance(size,dict):size=[size.get("height"),size.get("width")]
    if list(size or [])!=[inp["height"],inp["width"]]:
        raise ValueError(f"engine input_size {size} != standard {inp['height']}x{inp['width']} ({path.name})")
    order=str(manifest.get("color_order","RGB")).upper()
    if order!=inp["color_order"]:
        raise ValueError(f"engine color_order {order} != standard {inp['color_order']}")
    for key in ("mean","std"):
        if not np.allclose(np.asarray(manifest.get(key,[]),np.float64),np.asarray(inp[key],np.float64),atol=1e-3):
            raise ValueError(f"engine {key} {manifest.get(key)} != standard {inp[key]}")
    if int(manifest.get("num_classes",3))!=3:
        raise ValueError(f"engine num_classes {manifest.get('num_classes')} != 3")
    if manifest.get("argmax_in_graph"):
        raise ValueError("engine outputs a class mask (argmax_in_graph); the app needs float logits [1,3,H,W] "
                         "for softmax confidence. Re-export with `deploy.export_onnx --keep-logits` and rebuild")

class TrtBackend:
    def __init__(self,path,spec):
        path=Path(path)
        manifest_path=path.with_suffix(path.suffix+".json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"engine manifest missing: {manifest_path.name} (deploy.build_engine writes it next to the engine)")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        _check_manifest(manifest,spec,path)
        self.normalize_in_graph=bool(manifest.get("normalize_in_graph",False))
        build=manifest.get("build",{}) or {}
        self.precision=str(build.get("precision","?")).upper()
        self.trt_version=str(build.get("trt_version","?"))
        # [Jetson 속도] decode_in_graph 엔진(deploy.add_decode): argmax 와 softmax 신뢰도를 GPU 그래프가 계산해 labels/confidence 로 낸다.
        self.decode_in_graph=bool(manifest.get("decode_in_graph",False))
        TrtSegmenter=_import_trt_segmenter()
        self.segmenter=TrtSegmenter(path)
        inp=spec["input"];expected=(1,3,inp["height"],inp["width"])
        if self.decode_in_graph:
            names=list(self.segmenter.out_names);plane=(1,inp["height"],inp["width"])
            shapes={n:tuple(h.shape) for n,h in zip(names,self.segmenter.host_outs)}
            dtypes={n:np.dtype(h.dtype) for n,h in zip(names,self.segmenter.host_outs)}
            if ("labels" not in names or "confidence" not in names or shapes["labels"]!=plane or shapes["confidence"]!=plane
                    or not np.issubdtype(dtypes["labels"],np.integer) or not np.issubdtype(dtypes["confidence"],np.floating)):
                self.close()
                raise ValueError(f"decode_in_graph engine must output labels int [1,H,W] and confidence float [1,H,W]; got {shapes} {dtypes}")
            self._labels_index=names.index("labels");self._confidence_index=names.index("confidence")
        else:
            out_shape=tuple(self.segmenter.out_shape);out_dtype=np.dtype(self.segmenter.host_out.dtype)
            if len(self.segmenter.out_names)!=1 or out_shape!=expected or not np.issubdtype(out_dtype,np.floating):
                self.close()
                raise ValueError(f"engine output {out_shape}/{out_dtype} is not float logits {expected}; rebuild from a --keep-logits ONNX")
        if tuple(self.segmenter.in_shape)!=expected:
            self.close()
            raise ValueError(f"engine input {self.segmenter.in_shape} != {expected}")
        # 첫 프레임 지연을 줄이도록 카메라 시작 전에 커널/메모리를 준비한다.
        self.segmenter.run(np.zeros(expected,np.float32))

    def describe(self):
        return f"TensorRT {self.trt_version} {self.precision} GPU"+(" +decode" if self.decode_in_graph else "")

    def __call__(self,array):
        # run() 은 다음 호출 때 덮어쓰는 고정 호스트 버퍼를 돌려주므로 복사해서 넘긴다(로짓 약 3.7MB).
        out=self.segmenter.run(array)
        if self.decode_in_graph:
            labels=out[self._labels_index][0].astype(np.uint8)            # int32 → uint8 복사
            confidence=np.array(out[self._confidence_index][0],dtype=np.float32,copy=True)
            return labels,confidence
        return np.array(out,dtype=np.float32,copy=True)

    def close(self):
        segmenter=getattr(self,"segmenter",None)
        if segmenter is not None:
            segmenter.close();self.segmenter=None
