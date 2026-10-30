"""고정 표준 설정 + models 폴더의 단일 PTH로 동일 RGB 프레임을 추론. 오류 시 마스크를 만들지 않는다."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STANDARD_SPEC_NAME = "standard_fastscnn.json"

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

    def record(self):
        return dict(frame_id=self.frame_id, status=self.status, message=self.message,
                    inference_ms=self.inference_ms, model_sha256=self.model_sha256,
                    model_filename=self.model_filename,
                    classes={"0": "floor", "1": "obstacle", "2": "ceiling/other"})

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
    return spec

def prepare_input(bgr, spec):
    inp = spec["input"]
    resized = cv2.resize(bgr, (inp["width"], inp["height"]), interpolation=cv2.INTER_LINEAR)
    if inp["color_order"] == "RGB":
        resized = resized[:, :, ::-1]
    # [v3.5] 정규화 연산 순서는 유지하고 3채널 임시 배열의 반복 할당을 줄인다.
    x = resized.astype(np.float32)
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

def decode_logits(output,spec,shape):
    return decode_prediction(output,spec,shape)[0]

class FastSCNN:
    def __init__(self, cfg):
        self.net = None
        self.backend = None
        self.spec = None
        self.model_sha256 = ""
        self.model_filename = ""
        self.status, self.message = "DISABLED", "SCNN disabled in config"
        if not cfg["enabled"]:
            return
        try:
            directory=Path(cfg["model_dir"]).expanduser()
            if not directory.is_absolute():directory=ROOT/directory
            # [수정] 시작할 때 바로 아래의 실제 .pth 파일만 센다. 하위 폴더/다른 확장자는 제외.
            if not directory.exists():
                candidates=[]
            else:
                candidates=sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower()==".pth"),
                                  key=lambda p:p.name.lower())
            if not candidates:
                self.status,self.message="MODEL MISSING","missing model: keep exactly one compatible .pth file in "+str(directory)
                return
            if len(candidates)>1:
                self.status,self.message="TOO MUCH MODEL",f"too much model: {len(candidates)} .pth files found; keep exactly one in "+str(directory)
                return
            path=candidates[0]
            self.model_filename=path.name
            # [수정] PTH 이름과 관계없이 같은 표준 JSON을 사용한다.
            self.spec=validate_spec(json.loads((directory/STANDARD_SPEC_NAME).read_text(encoding="utf-8-sig")))
            # 고정 JSON의 예전 모델 해시와 비교하지 않음. 실제 사용 파일의 해시는 촬영 이력에 기록.
            self.model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            from .torch_fastscnn import TorchBackend
            self.backend=TorchBackend(path,cfg.get("torch_threads",2))
            self.net=self.backend
            self.status,self.message="READY",path.name+" | Team FastSCNN | PyTorch CPU"
        except Exception as exc:
            self.net=None;self.backend=None
            self.status,self.message="MODEL ERROR",str(exc)

    def predict(self, bgr, frame_id):
        if self.net is None:
            return SemanticResult(None, frame_id, self.status, self.message,
                                  model_sha256=self.model_sha256,model_filename=self.model_filename)
        start = time.monotonic()
        try:
            tensor=prepare_input(bgr,self.spec)
            output=self.backend(tensor)
            labels,confidence=decode_prediction(output,self.spec,bgr.shape[:2])
            return SemanticResult(labels, frame_id, "OK", self.message,
                                  (time.monotonic()-start)*1000, self.model_sha256, confidence,self.model_filename)
        except Exception as exc:
            # [v3.5 추가] 예전 마스크를 재사용하지 않는다. 오류는 표시하고 Depth 처리는 계속한다.
            self.net = None
            self.status, self.message = "INFERENCE ERROR", str(exc)
            return SemanticResult(None, frame_id, self.status, self.message,
                                  (time.monotonic()-start)*1000, self.model_sha256,model_filename=self.model_filename)
