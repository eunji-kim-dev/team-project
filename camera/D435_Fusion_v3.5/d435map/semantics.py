"""고정 표준 설정 + models 폴더의 단일 모델 파일(.pth 또는 .engine)로 동일 RGB 프레임을 추론. 오류 시 마스크를 만들지 않는다.
- .pth    : torch_fastscnn.TorchBackend (PyTorch CPU)
- .engine : TensorRT GPU (Jetson). 옆에 deploy 가 만든 <이름>.engine.json 매니페스트가 있는지로 백엔드를 고른다.
  - 있음 → trt_fastscnn.TrtBackend. float 로짓 [1,3,H,W] (softmax 신뢰도는 여기서 계산) 또는
           decode_in_graph 엔진의 (labels, confidence) 튜플 (GPU 가 디코드, 여기서는 검사만).
  - 없음 → tensorrt_backend.TensorRTBackend. standard_fastscnn.json 의 "engine" 계약(normalize_in_graph,
           output_kind labels|logits, hard_label_fusion)을 따른다. labels 엔진은 확률이 없으므로 evidence_kind 로
           구분하고 fusion 에는 fusion_scores() 가 1.0 을 넘긴다.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STANDARD_SPEC_NAME = "standard_fastscnn.json"
# [Jetson] 모델 파일로 인정하는 확장자. 폴더에 이 중 정확히 하나만 있어야 한다.
MODEL_SUFFIXES = (".pth", ".engine")

@dataclass
class SemanticResult:
    labels: object
    frame_id: str
    status: str
    message: str
    inference_ms: float = 0.0
    model_sha256: str = ""
    confidence: object = None
    model_filename: str = ""
    timing: object = None   # [Jetson 측정] {prepare_ms, infer_ms, decode_ms}
    evidence_kind: str = "softmax"
    backend_info: object = None

    def record(self):
        return dict(frame_id=self.frame_id, status=self.status, message=self.message,
                    inference_ms=self.inference_ms, model_sha256=self.model_sha256,
                    model_filename=self.model_filename,timing=self.timing,
                    evidence_kind=self.evidence_kind, backend_info=self.backend_info,
                    confidence_available=self.confidence is not None,
                    classes={"0": "floor", "1": "obstacle", "2": "ceiling/other"})


def fusion_scores(semantic):
    # [engine 추가] 라벨만 출력하는 엔진에는 확률이 없다. 1은 내부 분류 채택 gate이며
    # 100% 확률이 아니다. confidence 파일을 합성하지 않고 evidence_kind를 별도 기록한다.
    # 기존 Depth/바닥 평면/가림/경계/시간 검증은 그대로 적용한다.
    if semantic.evidence_kind == "hard_labels":
        return np.ones(semantic.labels.shape, np.float32)
    return semantic.confidence

def validate_spec(spec):
    # [v3.5 추가] RGB/BGR, 정규화, 클래스 순서를 모델과 함께 명시한다.
    if spec.get("schema") != "fastscnn-rgb-v1":
        raise ValueError("Model JSON schema must be fastscnn-rgb-v1")
    inp, out = spec["input"], spec["output"]
    for key in ("width", "height"):
        if type(inp[key]) is not int or inp[key] < 1:
            raise ValueError("Model input width/height must be positive integers")
    if inp["color_order"] not in ("RGB", "BGR"):
        raise ValueError("input.color_order must be RGB or BGR")
    if not np.isfinite(inp["scale"]) or inp["scale"] <= 0:
        raise ValueError("Model scale must be finite and positive")
    for key in ("mean", "std"):
        v = np.asarray(inp[key], dtype=np.float32)
        if v.shape != (3,) or not np.isfinite(v).all() or (key == "std" and (v <= 0).any()):
            raise ValueError("Model mean/std need three finite values; std must be positive")
    if out["kind"] != "logits" or out["layout"] != "NCHW":
        raise ValueError("Expected one float32 [1,3,H,W] logits output")
    ids = out["class_ids"]
    if len(ids) != 3 or any(type(x) is not int for x in ids) or sorted(ids) != [0, 1, 2]:
        raise ValueError("Output channel class_ids must be a permutation of [0,1,2]")
    if "engine" in spec:
        e=spec["engine"]
        if type(e.get("normalize_in_graph")) is not bool or e.get("output_kind") not in ("labels","logits"):
            raise ValueError("engine requires normalize_in_graph (bool), output_kind (labels/logits)")
        if type(e.get("hard_label_fusion")) is not bool:
            raise ValueError("engine.hard_label_fusion must be bool")
    return spec

def prepare_input(bgr, spec, normalize=True):
    inp = spec["input"]
    resized = cv2.resize(bgr, (inp["width"], inp["height"]), interpolation=cv2.INTER_LINEAR)
    if inp["color_order"] == "RGB":
        resized = resized[:, :, ::-1]
    # [v3.5] 정규화 연산 순서는 유지하고 3채널 임시 배열의 반복 할당을 줄인다.
    x = resized.astype(np.float32)
    # [Jetson] normalize_in_graph 엔진은 normalize=False: 0~255 그대로 넘기고 scale/mean/std 는 GPU 그래프가 적용한다.
    if normalize:
        x *= np.float32(inp["scale"])
        x -= np.asarray(inp["mean"],np.float32)
        x /= np.asarray(inp["std"],np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

def decode_prediction(output,spec,shape):
    # [v3.5] 3개 채널 최대값/argmax/softmax를 함께 계산. 학습 모델/정밀도 변경 없음.
    a=np.asarray(output);inp=spec["input"]
    if a.shape!=(1,3,inp["height"],inp["width"]) or not np.issubdtype(a.dtype,np.floating):
        raise ValueError(f"Expected full-input-resolution float logits, got {a.shape}/{a.dtype}")
    if not np.isfinite(a).all():raise ValueError("SCNN returned NaN/Inf")
    channels=a[0]
    if channels.shape[1:]!=tuple(shape):
        channels=cv2.resize(channels.transpose(1,2,0),(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
    first,second,third=channels
    top=np.maximum(first,second)
    winner=(second>first).astype(np.uint8)  # 동점이면 기존 np.argmax처럼 앞 채널
    winner[third>top]=2
    top=np.maximum(top,third)
    denominator=first-top;np.exp(denominator,out=denominator)
    scratch=second-top;np.exp(scratch,out=scratch);denominator+=scratch
    np.subtract(third,top,out=scratch);np.exp(scratch,out=scratch);denominator+=scratch
    confidence=(1.0/denominator).astype(np.float32)
    labels=np.asarray(spec["output"]["class_ids"],np.uint8)[winner]
    return labels,confidence

def decode_in_graph_result(output,spec,shape):
    # [Jetson 속도] 엔진이 argmax 와 softmax 신뢰도를 그래프 안에서 계산한 경우(decode_in_graph). 값 검사만 하고 그대로 쓴다.
    labels,confidence=output;inp=spec["input"]
    if not isinstance(labels,np.ndarray) or not isinstance(confidence,np.ndarray) or labels.shape!=(inp["height"],inp["width"]) or confidence.shape!=labels.shape:
        raise ValueError(f"Expected decoded labels/confidence at input resolution, got {getattr(labels,'shape',None)}/{getattr(confidence,'shape',None)}")
    if labels.dtype!=np.uint8 or labels.max(initial=0)>2:raise ValueError("decoded labels must be uint8 channel indices 0..2")
    if not np.issubdtype(confidence.dtype,np.floating) or not np.isfinite(confidence).all():raise ValueError("SCNN returned NaN/Inf confidence")
    if labels.shape!=tuple(shape):
        labels=cv2.resize(labels,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
        confidence=cv2.resize(confidence,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
    return np.asarray(spec["output"]["class_ids"],np.uint8)[labels],confidence.astype(np.float32,copy=False)

def decode_logits(output,spec,shape):
    return decode_prediction(output,spec,shape)[0]


def decode_mask(output,spec,shape):
    # [engine 추가] ArgMax가 포함된 int32/int64 마스크: 보간으로 클래스 번호를 섞지 않는다.
    a=np.asarray(output);h,w=spec["input"]["height"],spec["input"]["width"]
    if a.shape not in ((1,h,w),(1,1,h,w)) or not np.issubdtype(a.dtype,np.integer):
        raise ValueError(f"Expected integer class mask [1,H,W], got {a.shape}/{a.dtype}")
    if np.any((a<0)|(a>2)):
        raise ValueError("Engine returned class IDs outside 0,1,2")
    mask=np.asarray(spec["output"]["class_ids"],np.uint8)[a.reshape(h,w)]
    if mask.shape!=tuple(shape):
        mask=cv2.resize(mask,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
    return mask,None

class FastSCNN:
    def __init__(self, cfg):
        self.net = None
        self.backend = None
        self.spec = None
        self.model_sha256 = ""
        self.model_filename = ""
        self.normalize_on_cpu = True
        self.is_engine=False
        self.evidence_kind="softmax"
        self.backend_info=None
        self.decoder=decode_prediction   # 로짓 → (labels, softmax 신뢰도). labels 전용 엔진은 decode_mask
        self.status, self.message = "DISABLED", "SCNN disabled in config"
        if not cfg["enabled"]:
            return
        try:
            directory=Path(cfg["model_dir"]).expanduser()
            if not directory.is_absolute():directory=ROOT/directory
            # [수정] 시작할 때 바로 아래의 실제 모델 파일(.pth/.engine)만 센다. 하위 폴더/다른 확장자(.json 등)는 제외. 혼합 2개도 임의 선택하지 않는다.
            if not directory.exists():
                candidates=[]
            else:
                candidates=sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES),
                                  key=lambda p:p.name.lower())
            kinds="/".join(MODEL_SUFFIXES)
            if not candidates:
                self.status,self.message="MODEL MISSING",f"missing model: keep exactly one compatible {kinds} file in "+str(directory)
                return
            if len(candidates)>1:
                self.status,self.message="TOO MUCH MODEL",f"too much model: {len(candidates)} model files ({kinds}) found; keep exactly one in "+str(directory)
                return
            path=candidates[0]
            self.model_filename=path.name
            # [수정] 모델 이름과 관계없이 같은 표준 JSON을 사용한다.
            self.spec=validate_spec(json.loads((directory/STANDARD_SPEC_NAME).read_text(encoding="utf-8-sig")))
            # 고정 JSON의 예전 모델 해시와 비교하지 않음. 실제 사용 파일의 해시는 촬영 이력에 기록.
            self.model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix.lower()==".engine":
                self.is_engine=True
                manifest=path.with_suffix(path.suffix+".json")
                if manifest.exists():
                    # [Jetson] deploy 가 만든 <이름>.engine.json 이 있으면 저장소 deploy/segmenter 기반 백엔드.
                    # 로짓 또는 decode_in_graph(labels+confidence) 엔진. 매니페스트 검증(입력 사양 일치/로짓 출력)은 백엔드가 한다.
                    from .trt_fastscnn import TrtBackend
                    self.backend=TrtBackend(path,self.spec)
                    self.normalize_on_cpu=not self.backend.normalize_in_graph
                    self.backend_info={"backend":"tensorrt","manifest":manifest.name,"tensorrt_version":self.backend.trt_version,
                                       "precision":self.backend.precision,"decode_in_graph":self.backend.decode_in_graph}
                    label="Team FastSCNN | "+self.backend.describe()
                else:
                    # [engine 추가] 매니페스트 없는 엔진: standard_fastscnn.json 의 engine 계약을 따른다 (cuda-python 런타임).
                    if "engine" not in self.spec:
                        raise ValueError("standard_fastscnn.json needs the engine input/output contract (or put the deploy-made .engine.json next to the engine)")
                    from .tensorrt_backend import TensorRTBackend
                    self.backend=TensorRTBackend(path,self.spec)
                    self.backend_info=self.backend.info
                    self.normalize_on_cpu=not self.spec["engine"]["normalize_in_graph"]
                    if self.spec["engine"]["output_kind"]=="labels":
                        self.decoder=decode_mask
                        self.evidence_kind="hard_labels" if self.spec["engine"]["hard_label_fusion"] else "labels_only"
                    label="TensorRT "+self.backend.info["tensorrt_version"]+" | "+self.spec["engine"]["output_kind"]
            else:
                from .torch_fastscnn import TorchBackend
                self.backend=TorchBackend(path,cfg.get("torch_threads",2))
                self.normalize_on_cpu=True
                self.backend_info={"backend":"pytorch","device":"cpu"}
                label="Team FastSCNN | PyTorch CPU"
            self.net=self.backend
            self.status,self.message="READY",path.name+" | "+label
        except Exception as exc:
            self.net=None;self.backend=None
            self.status,self.message="MODEL ERROR",str(exc)

    def predict(self, bgr, frame_id):
        if self.net is None:
            return SemanticResult(None, frame_id, self.status, self.message,
                                  model_sha256=self.model_sha256,model_filename=self.model_filename)
        start = time.monotonic()
        try:
            tensor=prepare_input(bgr,self.spec,self.normalize_on_cpu)
            prepared=time.monotonic()
            output=self.backend(tensor)
            inferred=time.monotonic()
            # decode_in_graph 엔진은 (labels, confidence) 튜플 → 검사만. 그 외는 백엔드에 맞춘 디코더 (로짓 softmax 또는 argmax 마스크).
            if isinstance(output,tuple):labels,confidence=decode_in_graph_result(output,self.spec,bgr.shape[:2])
            else:labels,confidence=self.decoder(output,self.spec,bgr.shape[:2])
            done=time.monotonic()
            # [Jetson 측정] 단계별 시간. 입력 준비(CPU) / 모델 실행(백엔드) / 디코드(CPU). 합이 inference_ms.
            timing={"prepare_ms":(prepared-start)*1000,"infer_ms":(inferred-prepared)*1000,"decode_ms":(done-inferred)*1000}
            return SemanticResult(labels, frame_id, "OK", self.message,
                                  (done-start)*1000, self.model_sha256, confidence,self.model_filename,
                                  timing=timing,evidence_kind=self.evidence_kind,backend_info=self.backend_info)
        except Exception as exc:
            # [v3.5 추가] 예전 마스크를 재사용하지 않는다. 오류는 표시하고 Depth 처리는 계속한다.
            self.net = None
            self.status, self.message = "INFERENCE ERROR", str(exc)
            self.close()
            return SemanticResult(None, frame_id, self.status, self.message,
                                  (time.monotonic()-start)*1000, self.model_sha256,model_filename=self.model_filename)

    def close(self):
        # [engine 추가] 추론 종료 후 CUDA 자원을 회수. PTH 경로에는 CUDA 의존성을 만들지 않는다.
        backend=self.backend
        self.net=None;self.backend=None
        if backend is not None and hasattr(backend,"close"):backend.close()
