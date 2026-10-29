"""체크포인트(.pth) → ONNX. PC 에서 실행 (torch 만 있으면 됨, mmseg 불필요)

python -m deploy.export_onnx --ckpt checkpoints/best_mIoU_iter_1600.pth \\
    --model-py deploy/fastscnn_mmseg.py --out models/fast_scnn.onnx --check-image data/images/0001_color.png

- 정규화·업샘플·argmax 를 그래프 안에 넣음 → 입력은 0~255 RGB, 출력은 (1,H,W) int32 마스크
- PPM 의 AdaptiveAvgPool2d(2·3·6) 는 480x640 에서 export 가 안 되므로 행렬곱으로 바꿈
- 끝나면 onnxruntime 으로 torch 결과와 비교하고, 모델 옆에 <out>.json / preprocess.json / classes.json 저장함
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.preprocess import ModelSpec, load_spec, save_spec, make_blob  # noqa: E402

DEFAULT_OPSET = 17   # TensorRT 8.5 이상이면 17 로 충분함


# ---------- 체크포인트 읽기 ----------

class _Inert:
    """체크포인트 안의 mmengine/numpy 객체(학습 로그) 대신 들어가는 빈 껍데기. pickle 로 코드가 실행될 여지 없앰"""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return self


def _inert_fn(*args, **kwargs):
    return _Inert()


_STUB_GLOBALS = [   # weights_only 로더가 모르는 이름 → 빈 껍데기
    (_Inert, "mmengine.logging.history_buffer.HistoryBuffer"),
    (_inert_fn, "numpy.core.multiarray._reconstruct"),
    (_Inert, "numpy.ndarray"),
    (_Inert, "numpy.dtype"),
    (_inert_fn, "numpy.core.multiarray.scalar"),
    (_inert_fn, "builtins.getattr"),
]


class CheckpointLoader:
    """체크포인트에서 state_dict 와 meta(학습 설정 등) 꺼냄"""

    @staticmethod
    def load(path: Path) -> tuple[dict, dict]:
        try:
            torch.serialization.add_safe_globals(_STUB_GLOBALS)
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:   # 옛 torch 거나 처음 보는 객체
            print(f"[load] weights_only 로드 실패({type(e).__name__}) → 일반 로드로 다시 시도")
            obj = torch.load(path, map_location="cpu", weights_only=False)

        if not isinstance(obj, dict):
            raise ValueError(f"{path}: 알 수 없는 형식 (type={type(obj)})")

        only_tensors = "state_dict" not in obj and all(torch.is_tensor(v) for v in obj.values())
        if only_tensors:
            state_dict, meta = obj, {}   # state_dict 만 저장된 파일
        else:
            meta = obj.get("meta", {}) or {}
            state_dict = None
            for key in ("state_dict", "model", "model_state_dict", "net"):
                if key in obj and isinstance(obj[key], dict):
                    state_dict = obj[key]
                    break
            if state_dict is None:
                raise ValueError(f"{path}: state_dict 를 찾지 못함 (keys={list(obj)[:8]})")

        state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}   # DataParallel 접두사 제거
        return state_dict, meta

    @staticmethod
    def _dict_block(cfg_text: str, key: str) -> str | None:
        # cfg 문자열에서 `key=dict(` 의 괄호 안쪽만 잘라냄 (괄호 짝 세서)
        match = re.search(key + r"\s*=\s*dict\(", cfg_text)
        if not match:
            return None
        depth = 1
        pos = match.end()
        while pos < len(cfg_text) and depth > 0:
            depth += {"(": 1, ")": -1}.get(cfg_text[pos], 0)
            pos += 1
        return cfg_text[match.end():pos - 1]

    @classmethod
    def spec_from_cfg(cls, cfg_text: str) -> ModelSpec:
        """mmseg 학습 설정의 data_preprocessor / decode_head 에서 입력 사양 뽑음"""
        block = cls._dict_block(cfg_text, "data_preprocessor")
        if block is None:
            raise ValueError("cfg 에 data_preprocessor 가 없음 → --preprocess 로 직접 줄 것")

        def read_numbers(key: str) -> list[float]:
            match = re.search(key + r"\s*=\s*\[([^\]]*)\]", block, re.S)
            if not match:
                raise ValueError(f"cfg 의 data_preprocessor 에 {key} 가 없음 → --preprocess 로 직접 줄 것")
            return [float(v) for v in re.findall(r"[-\d.]+", match.group(1))]

        mean = read_numbers("mean")
        std = read_numbers("std")
        to_rgb = re.search(r"bgr_to_rgb\s*=\s*(True|False)", block)
        size = re.search(r"size\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)", block, re.S)
        if not size:
            raise ValueError("cfg 의 data_preprocessor 에 size=(H, W) 가 없음 → --preprocess 로 직접 줄 것")

        head_block = cls._dict_block(cfg_text, "decode_head")
        num_classes = re.search(r"num_classes\s*=\s*(\d+)", head_block if head_block else cfg_text)

        return ModelSpec(
            input_size=(int(size.group(1)), int(size.group(2))),
            mean=mean,
            std=std,
            color_order="RGB" if (to_rgb and to_rgb.group(1) == "True") else "BGR",
            num_classes=int(num_classes.group(1)) if num_classes else 3,
            input_range=1.0 if max(mean) <= 1.0 else 255.0,
        )


# 다른 파일에서 짧게 부르는 이름
def load_checkpoint(path: Path) -> tuple[dict, dict]:
    return CheckpointLoader.load(Path(path))


def load_state_dict(path) -> dict:
    return CheckpointLoader.load(Path(path))[0]


def spec_from_mmseg_cfg(cfg_text: str) -> ModelSpec:
    return CheckpointLoader.spec_from_cfg(cfg_text)


# ---------- 모델 만들기 ----------

def build_from_model_py(model_py: Path, state_dict: dict, num_classes: int, strict: bool) -> nn.Module:
    """model.py 의 build_model(num_classes) 로 모델 만들고 가중치 넣음"""
    module_spec = importlib.util.spec_from_file_location("role1_model", str(model_py))
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(model_py.parent))   # model.py 가 옆 파일 import 할 수 있게
    module_spec.loader.exec_module(module)

    if not hasattr(module, "build_model"):
        raise AttributeError(f"{model_py} 에 build_model(num_classes) 함수가 없음")
    model = module.build_model(num_classes)

    if hasattr(module, "adapt_state_dict"):
        state_dict = module.adapt_state_dict(state_dict)
    result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"[load] missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
        for key in list(result.missing_keys)[:5]:
            print("   missing:", key)
        for key in list(result.unexpected_keys)[:5]:
            print("   unexpected:", key)
    return model


def build_from_mmseg(config: Path, state_dict: dict) -> nn.Module:
    """mmseg 설치된 환경(Colab 등)용. mmengine 로더는 torch 2.6+ 에서 깨져서 가중치는 우리 로더 결과를 넣음"""
    from mmseg.apis import init_model
    model = init_model(str(config), None, device="cpu")
    print("[mmseg]", model.load_state_dict(state_dict, strict=False))
    return model


# ---------- adaptive pooling 교체 ----------

class StaticAdaptiveAvgPool2d(nn.Module):
    """입력 크기가 고정일 때 AdaptiveAvgPool2d 와 같은 결과를 행렬곱 2번으로 냄 (out = P_h · X · P_wᵀ)
    15x20 은 2·3·6 으로 안 나눠져서 ONNX export 가 실패함 → 평균 가중치 행렬을 미리 만들어 두면 MatMul 로 나감
    i 번째 출력 = 입력 [floor(i·n/o), ceil((i+1)·n/o)) 구간 평균 (PyTorch 공식 그대로)"""

    def __init__(self, in_hw: tuple, out_hw: tuple):
        super().__init__()
        self.register_buffer("ph", self._average_matrix(in_hw[0], out_hw[0]))                     # (out_h, in_h)
        self.register_buffer("pw_t", self._average_matrix(in_hw[1], out_hw[1]).t().contiguous())  # (in_w, out_w)

    @staticmethod
    def _average_matrix(n_in: int, n_out: int) -> torch.Tensor:
        matrix = torch.zeros(n_out, n_in)
        for i in range(n_out):
            start = math.floor(i * n_in / n_out)
            end = math.ceil((i + 1) * n_in / n_out)
            matrix[i, start:end] = 1.0 / (end - start)
        return matrix

    def forward(self, x):
        return torch.matmul(torch.matmul(self.ph, x), self.pw_t)


def freeze_adaptive_pools(model: nn.Module, example: torch.Tensor) -> list[str]:
    """example 을 한 번 흘려서 각 AdaptiveAvgPool2d 의 입력 크기를 재고, 안 나눠지는 것만 Static 으로 바꿈
    1x1 출력은 GlobalAveragePool 로 잘 나가므로 그대로 둠. 바꾼 모듈 이름 목록 반환"""
    input_sizes: dict[str, tuple] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            def record(mod, inputs, name=name):
                input_sizes[name] = tuple(int(v) for v in inputs[0].shape[2:])
            hooks.append(module.register_forward_pre_hook(record))
    with torch.no_grad():
        model(example)
    for hook in hooks:
        hook.remove()

    replaced = []
    for name, in_hw in input_sizes.items():
        module = model.get_submodule(name)
        out_size = module.output_size
        if not isinstance(out_size, (tuple, list)):
            out_size = (out_size, out_size)
        out_hw = tuple(in_hw[i] if out_size[i] is None else int(out_size[i]) for i in range(2))

        divisible = in_hw[0] % out_hw[0] == 0 and in_hw[1] % out_hw[1] == 0
        if out_hw == (1, 1) or divisible:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, StaticAdaptiveAvgPool2d(in_hw, out_hw))
        replaced.append(f"{name} {in_hw}→{out_hw}")
    return replaced


# ---------- 내보내기용 래퍼 ----------

class ExportWrapper(nn.Module):
    """모델 앞뒤에 전처리·후처리 붙임
    입력: (1,3,H,W) float32, 0~255, color_order 순서
    출력: argmax=True → (1,H,W) int32 마스크 / False → (1,C,H,W) 로짓 (입력 크기로 업샘플된 것)"""

    def __init__(self, core: nn.Module, spec: ModelSpec, argmax: bool):
        super().__init__()
        self.core = core
        self.argmax = argmax
        self.need_upsample = True                  # prepare() 에서 로짓 크기 보고 정함
        self.out_hw = (spec.height, spec.width)    # 상수로 둬야 ONNX 에 Shape/Slice 노드가 안 생김
        self.is_mmseg = hasattr(core, "extract_feat") and hasattr(core, "decode_head")
        self.register_buffer("mean", torch.tensor(spec.mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(spec.std, dtype=torch.float32).view(1, 3, 1, 1))
        self.pixel_scale = float(spec.pixel_scale)

    def raw_logits(self, x):
        if self.is_mmseg:
            out = self.core.decode_head(self.core.extract_feat(x))
        else:
            out = self.core(x)
        if isinstance(out, (tuple, list)):   # (main, aux...) 면 main 만
            out = out[0]
        if isinstance(out, dict):            # {"out": ...} 형태
            out = out.get("out", next(iter(out.values())))
        return out

    def forward(self, x):
        if self.pixel_scale != 1.0:
            x = x * self.pixel_scale
        x = (x - self.mean) / self.std
        logits = self.raw_logits(x)
        if self.need_upsample:
            logits = F.interpolate(logits, size=self.out_hw, mode="bilinear", align_corners=False)
        if self.argmax:
            return logits.argmax(dim=1).to(torch.int32)
        return logits


# ---------- 내보내기 + 검증 ----------

def export(wrapper: nn.Module, spec: ModelSpec, out: Path, opset: int) -> None:
    dummy = torch.rand(1, 3, spec.height, spec.width) * 255.0
    output_name = "mask" if wrapper.argmax else "logits"
    kwargs = dict(input_names=["input"], output_names=[output_name], opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(wrapper, dummy, str(out), dynamo=False, **kwargs)   # torch 2.9+ 는 기본이 dynamo 라 옛 방식 명시
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(out), **kwargs)                 # dynamo 인자 없는 옛 torch


def simplify_and_check(out: Path, simplify: bool = True) -> None:
    """onnxsim 으로 정리(선택) + onnx.checker"""
    import onnx
    model = onnx.load(str(out))
    if simplify:
        try:
            import onnxsim
            model, ok = onnxsim.simplify(model)
            print(f"[onnxsim] {'성공' if ok else '실패(원본 유지)'}")
            if ok:
                onnx.save(model, str(out))
        except ImportError:
            print("[onnxsim] 미설치 → 건너뜀")
    onnx.checker.check_model(str(out))
    op_types = sorted({node.op_type for node in onnx.load(str(out)).graph.node})
    print(f"[onnx] checker 통과: {out} ({out.stat().st_size / 1e6:.1f} MB), ops={op_types}")


def verify_with_ort(wrapper: nn.Module, spec: ModelSpec, out: Path, image: Path | None) -> dict:
    """같은 입력을 torch 와 onnxruntime 에 넣어 비교"""
    import onnxruntime as ort
    if image is not None:
        import cv2
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image)
        blob = make_blob(bgr, spec)
    else:
        rng = np.random.default_rng(0)
        blob = rng.random((1, 3, spec.height, spec.width), dtype=np.float32) * 255.0

    with torch.no_grad():
        expected = wrapper(torch.from_numpy(blob)).cpu().numpy()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: blob})[0]

    if wrapper.argmax:
        agreement = float((expected == actual).mean()) * 100.0
        print(f"[verify] torch vs onnxruntime 마스크 일치율 {agreement:.3f}%  (100% 에 가까워야 함)")
        return {"mask_agreement_pct": agreement}

    max_diff = float(np.abs(expected - actual).max())
    agreement = float((expected.argmax(1) == actual.argmax(1)).mean()) * 100.0
    print(f"[verify] 로짓 최대 차이 {max_diff:.2e}, argmax 일치율 {agreement:.3f}%")
    return {"logit_max_abs_diff": max_diff, "mask_agreement_pct": agreement}


class OnnxExporter:
    """prepare() → run() → save_manifests() 순서로 씀"""

    def __init__(self, core: nn.Module, spec: ModelSpec, keep_logits: bool = False):
        self.spec = spec
        self.keep_logits = keep_logits
        self.wrapper = ExportWrapper(core.eval(), spec, argmax=not keep_logits).eval()
        self.replaced_pools: list[str] = []

    def prepare(self) -> None:
        """출력 채널 확인, 업샘플 필요 여부, adaptive pooling 교체"""
        dummy = torch.zeros(1, 3, self.spec.height, self.spec.width)
        with torch.no_grad():
            logits = self.wrapper.raw_logits(dummy)
        if logits.shape[1] != self.spec.num_classes:
            raise ValueError(f"모델 출력 채널 {logits.shape[1]} ≠ num_classes {self.spec.num_classes}")

        self.wrapper.need_upsample = tuple(logits.shape[-2:]) != (self.spec.height, self.spec.width)
        note = " → 그래프 안에서 입력 크기로 업샘플함" if self.wrapper.need_upsample else ""
        print(f"[model] 로짓 모양 {tuple(logits.shape)} (입력 대비 1/{self.spec.height // logits.shape[-2]}){note}")

        self.replaced_pools = freeze_adaptive_pools(self.wrapper, dummy)
        if self.replaced_pools:
            print(f"[model] AdaptiveAvgPool2d → StaticAdaptiveAvgPool2d 교체 {len(self.replaced_pools)}개: {self.replaced_pools}")

    def run(self, out: Path, opset: int, check_image: Path | None) -> dict:
        out.parent.mkdir(parents=True, exist_ok=True)
        export(self.wrapper, self.spec, out, opset)
        simplify_and_check(out)
        return verify_with_ort(self.wrapper, self.spec, out, check_image)

    def save_manifests(self, out: Path, opset: int, meta: dict, ckpt: Path, verify: dict, classes: dict | None) -> None:
        """<out>.json (segmenter 가 읽음) + preprocess.json + classes.json"""
        spec = self.spec
        spec.normalize_in_graph = True
        spec.argmax_in_graph = not self.keep_logits
        save_spec(out.with_suffix(out.suffix + ".json"), spec,
                  onnx_input="input",
                  onnx_output="logits" if self.keep_logits else "mask",
                  opset=opset,
                  torch_version=torch.__version__,
                  source_ckpt=str(ckpt),
                  ckpt_iter=meta.get("iter"),
                  exported_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  verify=verify,
                  classes=classes["classes"] if classes else None)

        plain = ModelSpec(spec.input_size, spec.mean, spec.std, spec.color_order, spec.num_classes, spec.input_range)
        save_spec(out.parent / "preprocess.json", plain)
        if classes:
            (out.parent / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- main ----------

def parse_args():
    parser = argparse.ArgumentParser(description="best.pth → ONNX")
    parser.add_argument("--ckpt", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-py", type=Path, help="build_model(num_classes) 가 있는 .py (deploy/fastscnn_mmseg.py)")
    source.add_argument("--mmseg-config", type=Path, help="mmseg config.py (mmseg 설치 환경에서만)")
    parser.add_argument("--preprocess", type=Path, help="preprocess.json. 생략하면 체크포인트 meta.cfg 에서 읽음")
    parser.add_argument("--out", required=True, type=Path, help="예: models/fast_scnn.onnx")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--keep-logits", action="store_true", help="argmax 없이 로짓으로 내보냄")
    parser.add_argument("--non-strict", action="store_true", help="키 안 맞아도 넣을 수 있는 것만 넣음")
    parser.add_argument("--check-image", type=Path, help="검증용 실제 사진 (없으면 난수)")
    return parser.parse_args()


def main():
    args = parse_args()

    state_dict, meta = CheckpointLoader.load(args.ckpt)
    print(f"[ckpt] 텐서 {len(state_dict)}개, meta keys={list(meta)[:8]}")
    if meta.get("iter") is not None:
        print(f"[ckpt] iter {meta.get('iter')}, seed {meta.get('seed')}, 실험 {meta.get('experiment_name')}")

    # 입력 사양: 파일로 받았으면 그걸, 아니면 체크포인트 안의 학습 설정에서
    if args.preprocess:
        spec = load_spec(args.preprocess)
    elif isinstance(meta.get("cfg"), str):
        spec = CheckpointLoader.spec_from_cfg(meta["cfg"])
        print("[spec] 체크포인트 meta.cfg 의 data_preprocessor 에서 읽음")
    else:
        raise SystemExit("--preprocess 가 필요함 (체크포인트에 cfg 가 없음)")
    print(f"[spec] input {spec.height}x{spec.width}, {spec.color_order}, range {spec.input_range:g}, classes {spec.num_classes}")

    # 클래스 이름·팔레트 (mmseg 체크포인트에 들어 있음)
    classes = None
    dataset_meta = meta.get("dataset_meta") or {}
    if dataset_meta.get("classes"):
        classes = {"classes": list(dataset_meta["classes"]),
                   "palette_rgb": [list(color) for color in dataset_meta.get("palette", [])]}
        print(f"[ckpt] classes {classes['classes']}, palette(RGB) {classes['palette_rgb']}")

    if args.mmseg_config:
        core = build_from_mmseg(args.mmseg_config, state_dict)
    else:
        core = build_from_model_py(args.model_py, state_dict, spec.num_classes, strict=not args.non_strict)

    started = time.time()
    exporter = OnnxExporter(core, spec, keep_logits=args.keep_logits)
    exporter.prepare()
    verify = exporter.run(args.out, args.opset, args.check_image)
    exporter.save_manifests(args.out, args.opset, meta, args.ckpt, verify, classes)

    produced = f"{args.out} + {args.out.name}.json + preprocess.json" + (" + classes.json" if classes else "")
    print(f"[done] {produced}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
