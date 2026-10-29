"""입력 전처리 공통 모듈
- 리사이즈, 채널 순서, 정규화를 여기서만 함 (변환/캘리브레이션/추론/벤치 전부 make_blob 하나 씀)
- preprocess.json 예시
    {"input_size": [480, 640], "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375],
     "color_order": "RGB", "num_classes": 3}
- export_onnx 가 만드는 <모델>.json 에는 normalize_in_graph / argmax_in_graph 가 추가됨
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ModelSpec:
    input_size: tuple                      # (높이, 너비)
    mean: list                             # 채널별 평균
    std: list                              # 채널별 표준편차
    color_order: str = "RGB"               # 모델이 받는 채널 순서. RGB / BGR
    num_classes: int = 3                   # 0 floor / 1 obstacle / 2 other
    input_range: float = 255.0             # 모델이 받는 픽셀 범위. 255 또는 1
    normalize_in_graph: bool = False       # True 면 정규화를 모델 그래프가 함 (CPU 는 0~255 그대로 줌)
    argmax_in_graph: bool | None = None    # True 면 출력이 로짓이 아니라 클래스 번호 마스크
    extra: dict = field(default_factory=dict)   # 모르는 키 보관용 (저장할 때 다시 합침)

    @property
    def height(self) -> int:
        return int(self.input_size[0])

    @property
    def width(self) -> int:
        return int(self.input_size[1])

    @property
    def pixel_scale(self) -> float:
        return float(self.input_range) / 255.0   # 0~255 픽셀에 곱할 값. 255 → 1.0, 1 → 1/255

    def to_dict(self) -> dict:
        data = asdict(self)
        data["input_size"] = [self.height, self.width]
        extra = data.pop("extra")
        data.update(extra)
        return data


def _parse_input_size(raw_size) -> tuple:
    # [H, W] 리스트나 {"height":..,"width":..} 둘 다 받음
    if isinstance(raw_size, dict):
        return int(raw_size["height"]), int(raw_size["width"])
    if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
        return int(raw_size[0]), int(raw_size[1])
    raise ValueError(f"input_size 는 [H, W] 형식이어야 함 (지금: {raw_size!r})")


def _parse_color_order(raw: dict) -> str:
    # color_order / to_rgb / bgr_to_rgb 어느 키로 적혀 있어도 RGB 또는 BGR 로 통일함
    if "color_order" in raw:
        order = str(raw["color_order"]).upper()
    elif "to_rgb" in raw:
        order = "RGB" if raw["to_rgb"] else "BGR"
    elif "bgr_to_rgb" in raw:
        order = "RGB" if raw["bgr_to_rgb"] else "BGR"
    else:
        order = "RGB"   # 표시 없으면 mmseg 기본값
    if order not in ("RGB", "BGR"):
        raise ValueError(f"color_order 는 RGB 또는 BGR 이어야 함 (지금: {order})")
    return order


def load_spec(path: str | Path) -> ModelSpec:
    """preprocess.json 또는 <모델>.json 읽음"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mean = [float(v) for v in raw.get("mean", [0.0, 0.0, 0.0])]
    std = [float(v) for v in raw.get("std", [1.0, 1.0, 1.0])]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"{path}: mean/std 는 길이 3 이어야 함")

    if "input_range" in raw:
        input_range = float(raw["input_range"])
    else:
        input_range = 1.0 if max(mean) <= 1.0 else 255.0   # mean 이 전부 1 이하면 0~1 스케일 모델로 봄

    known_keys = {"input_size", "mean", "std", "color_order", "to_rgb", "bgr_to_rgb",
                  "num_classes", "input_range", "normalize_in_graph", "argmax_in_graph"}
    extra = {key: value for key, value in raw.items() if key not in known_keys}

    return ModelSpec(
        input_size=_parse_input_size(raw.get("input_size")),
        mean=mean,
        std=std,
        color_order=_parse_color_order(raw),
        num_classes=int(raw.get("num_classes", 3)),
        input_range=input_range,
        normalize_in_graph=bool(raw.get("normalize_in_graph", False)),
        argmax_in_graph=raw.get("argmax_in_graph", None),
        extra=extra,
    )


def save_spec(path: str | Path, spec: ModelSpec, **more) -> None:
    """spec + 추가 정보(more)를 json 으로 저장함"""
    data = spec.to_dict()
    data.update(more)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_blob(bgr: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """BGR 영상(H×W×3 uint8) → 모델 입력 (1×3×H×W float32)
    - 크기 다르면 input_size 로 리사이즈 (비율 무시)
    - normalize_in_graph 면 0~255 그대로, 아니면 여기서 정규화까지 함
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"BGR 3채널 영상이어야 함 (지금 모양: {bgr.shape})")

    height, width = spec.height, spec.width
    swap_rb = spec.color_order == "RGB"

    if spec.normalize_in_graph:
        # 리사이즈 + 채널 교환 + NCHW + float32 를 OpenCV 한 번 호출로 끝냄 (제일 빠름)
        blob = cv2.dnn.blobFromImage(bgr, scalefactor=1.0, size=(width, height),
                                     mean=(0.0, 0.0, 0.0), swapRB=swap_rb, crop=False)
        return np.ascontiguousarray(blob, dtype=np.float32)

    image = bgr
    if image.shape[0] != height or image.shape[1] != width:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if swap_rb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixels = image.astype(np.float32)
    if spec.pixel_scale != 1.0:
        pixels *= spec.pixel_scale
    pixels = (pixels - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(spec.std, dtype=np.float32)

    blob = pixels.transpose(2, 0, 1)[None]   # H×W×C → 1×C×H×W
    return np.ascontiguousarray(blob, dtype=np.float32)


def read_image_list(list_path: str | Path) -> list[Path]:
    """한 줄에 경로 하나인 목록(calib_list.txt) 읽음. 상대경로는 목록 파일 위치 기준"""
    list_path = Path(list_path)
    base_dir = list_path.parent
    paths = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = base_dir / path
            paths.append(path)
    return paths
