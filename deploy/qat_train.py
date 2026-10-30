"""QAT INT8: FP32 체크포인트를 가짜 양자화 상태로 조금 더 학습 → Q/DQ 든 ONNX 로 내보냄 (PC/Colab)

python -m deploy.qat_train --ckpt checkpoints/best_mIoU_iter_1600.pth --data-root data \\
    --out models/fast_scnn_qat.onnx --iters 300 --lr 1e-4 --batch 4

순서
  [1] FP32 val mIoU 측정 (기준)
  [2] BN 을 conv 에 접음 → [1] 과 같아야 정상
  [3] conv 전부 QuantConv2d 로 바꾸고 학습 사진 몇 배치로 scale 캘리브레이션 → val mIoU (PTQ 흉내)
  [4] 가짜 양자화 켠 채 낮은 lr 로 학습, 주기적으로 val, 제일 좋은 가중치 보관
  [5] Q/DQ ONNX 로 내보내고 onnxruntime 과 비교 → models/fast_scnn_qat.onnx (+ .json)
  → Jetson: python -m deploy.build_engine --onnx models/fast_scnn_qat.onnx --precision qat

데이터 형식 (역할 1 mmseg 설정과 같음)
  <data_root>/images/<이름>_color.png, labels_RGB/<이름>_label_RGB.png, splits/train.txt, val.txt
  라벨은 색으로 칠한 png. 팔레트에 없는 중간색은 역할 1 로더와 똑같이 가장 가까운 색으로 붙임
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy import quant  # noqa: E402
from deploy.export_onnx import (ExportWrapper, build_from_model_py, export, freeze_adaptive_pools,  # noqa: E402
                                load_checkpoint, simplify_and_check, spec_from_mmseg_cfg)
from deploy.preprocess import ModelSpec, load_spec, save_spec  # noqa: E402

DEFAULT_PALETTE_RGB = [[0, 255, 0], [255, 0, 0], [0, 0, 255]]   # floor, obstacle, other
IGNORE = 255   # loss 에서 빼는 라벨 값


# ---------- 데이터 ----------

def color_mask_to_index(mask_bgr: np.ndarray, palette_rgb: list) -> np.ndarray:
    """색 라벨(BGR 로 읽은 것) → 클래스 번호. 팔레트에 없는 색은 가장 가까운 색으로 (역할 1 rgb_mask_loader 와 같은 규칙)"""
    rgb = mask_bgr[:, :, ::-1].astype(np.int32)
    palette = np.asarray(palette_rgb, dtype=np.int32)
    dist = ((rgb[:, :, None, :] - palette) ** 2).sum(-1)   # (H, W, 클래스수)
    return dist.argmin(-1).astype(np.uint8)


def read_label(path: Path, palette_rgb: list, num_classes: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:                                   # 색 라벨
        return color_mask_to_index(mask[:, :, :3], palette_rgb)
    bad_values = [int(v) for v in np.unique(mask) if v != IGNORE and v >= num_classes]   # 0/1/2 라벨
    if bad_values:
        raise ValueError(f"{path}: 클래스 번호가 아닌 값 {bad_values}")
    return mask.astype(np.uint8)


def read_split(list_path: Path | None, img_dir: Path, img_suffix: str) -> list[str]:
    """splits/train.txt 처럼 한 줄에 이름 하나. 목록 없으면 폴더의 전체 사진"""
    if list_path is not None:
        lines = list_path.read_text(encoding="utf-8").splitlines()
        return [line.split()[0] for line in lines if line.strip() and not line.startswith("#")]
    return sorted(p.name[: -len(img_suffix)] for p in img_dir.iterdir() if p.name.endswith(img_suffix))


class RGBMaskDataset(Dataset):
    """사진 + 색 라벨. train=True 면 간단한 증강(리사이즈 0.9~1.1, 크롭, 좌우반전, 밝기·대비)"""

    def __init__(self, root: Path, names: list[str], spec: ModelSpec, palette_rgb: list, train: bool,
                 img_dir="images", ann_dir="labels_RGB", img_suffix="_color.png", ann_suffix="_label_RGB.png"):
        self.items = [(root / img_dir / f"{n}{img_suffix}", root / ann_dir / f"{n}{ann_suffix}") for n in names]
        missing = [str(p) for pair in self.items for p in pair if not p.exists()]
        if missing:
            raise FileNotFoundError(f"파일 {len(missing)}개 없음. 예: {missing[:3]}")
        self.spec = spec
        self.palette = palette_rgb
        self.train = train
        self.height, self.width = spec.height, spec.width

    def __len__(self):
        return len(self.items)

    def _augment(self, img, label):
        # 리사이즈 0.9~1.1 (비율 유지) → 모자라면 패딩 → 랜덤 크롭 → 좌우반전 → 밝기·대비
        ratio = random.uniform(0.9, 1.1)
        new_h = max(1, int(round(img.shape[0] * ratio)))
        new_w = max(1, int(round(img.shape[1] * ratio)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pad_h, pad_w = max(0, self.height - new_h), max(0, self.width - new_w)
        if pad_h or pad_w:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE)

        y = random.randint(0, img.shape[0] - self.height)
        x = random.randint(0, img.shape[1] - self.width)
        img = img[y:y + self.height, x:x + self.width]
        label = label[y:y + self.height, x:x + self.width]

        if random.random() < 0.5:
            img, label = img[:, ::-1], label[:, ::-1]
        if random.random() < 0.5:
            alpha, beta = random.uniform(0.8, 1.2), random.uniform(-16, 16)   # 대비, 밝기
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img), np.ascontiguousarray(label)

    def __getitem__(self, index):
        img_path, label_path = self.items[index]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        label = read_label(label_path, self.palette, self.spec.num_classes)
        if label.shape != img.shape[:2]:
            label = cv2.resize(label, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.train:
            img, label = self._augment(img, label)
        if img.shape[0] != self.height or img.shape[1] != self.width:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if self.spec.color_order == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))   # 0~255. 정규화는 모델 안에서
        return x, torch.from_numpy(label.astype(np.int64))


def make_calibration_loader(train_loader: DataLoader, no_aug: bool) -> DataLoader:
    """기존 샘플링은 유지하고, 요청한 경우 캘리브레이션용 복사본의 증강만 끈다."""
    if not no_aug:
        return train_loader
    calib_set = copy.copy(train_loader.dataset)
    calib_set.train = False   # 원본 학습 데이터셋의 증강 설정은 건드리지 않음
    # 별도 generator / 미리보기 iterator를 만들지 않아 기존 RNG와 샘플 순서를 유지.
    return DataLoader(calib_set, batch_size=train_loader.batch_size, shuffle=True,
                      num_workers=train_loader.num_workers, drop_last=train_loader.drop_last)


# ---------- 평가 ----------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device) -> dict:
    """val 전체의 혼동행렬 → mIoU, 클래스별 IoU, 픽셀 정확도"""
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        valid = y != IGNORE
        index = y[valid] * num_classes + pred[valid]
        confusion += torch.bincount(index, minlength=num_classes ** 2).view(num_classes, num_classes)

    inter = confusion.diag().double()
    union = confusion.sum(0).double() + confusion.sum(1).double() - inter
    iou = torch.where(union > 0, inter / union, torch.full_like(inter, float("nan")))
    return {
        "miou": float(torch.nanmean(iou)),
        "iou": [None if torch.isnan(v) else round(float(v), 4) for v in iou],
        "pixel_acc": float(inter.sum() / confusion.sum().clamp_min(1)),
    }


def fmt(result: dict) -> str:
    return f"mIoU {result['miou']:.4f}  IoU {result['iou']}  acc {result['pixel_acc']:.4f}"


# ---------- QAT 흐름 ----------

class QatTrainer:
    """[1]~[5] 단계를 순서대로 실행. 결과(mIoU)는 self.results 에 모음"""

    def __init__(self, args, spec: ModelSpec, classes: list, palette: list, device):
        self.args = args
        self.spec = spec
        self.classes = classes
        self.palette = palette
        self.device = device
        self.results: dict[str, dict] = {}
        self.n_fused = 0       # 접은 BN 개수
        self.n_quant = 0       # QuantConv2d 개수
        self.model: nn.Module | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None

    def load_data(self):
        args, root = self.args, self.args.data_root

        def pick(explicit, default):
            if explicit:
                return explicit
            return default if default.exists() else None

        train_list = pick(args.train_list, root / "splits/train.txt")
        val_list = pick(args.val_list, root / "splits/val.txt")
        train_names = read_split(train_list, root / args.img_dir, args.img_suffix)
        val_names = read_split(val_list, root / args.img_dir, args.img_suffix) if val_list else []
        print(f"[data] train {len(train_names)}장, val {len(val_names)}장")

        kwargs = dict(img_dir=args.img_dir, ann_dir=args.ann_dir, img_suffix=args.img_suffix, ann_suffix=args.ann_suffix)
        train_set = RGBMaskDataset(root, train_names, self.spec, self.palette, train=True, **kwargs)
        self.train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
        if val_names:
            val_set = RGBMaskDataset(root, val_names, self.spec, self.palette, train=False, **kwargs)
            self.val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers)

    def _eval(self, tag: str, label: str):
        if self.val_loader is None:
            return
        self.results[tag] = evaluate(self.model, self.val_loader, self.spec.num_classes, self.device)
        print(f"{label} {fmt(self.results[tag])}")

    def build_model(self, state_dict: dict):
        """[1] FP32 기준, [2] BN 접기, [3] 가짜 양자화 + 캘리브레이션"""
        core = build_from_model_py(self.args.model_py, state_dict, self.spec.num_classes, strict=True)
        self.model = ExportWrapper(core, self.spec, argmax=False).to(self.device)   # 정규화 + 업샘플 포함, 출력은 로짓
        self._eval("fp32", "[1] FP32           ")

        self.n_fused = quant.fuse_conv_bn(self.model)
        self._eval("fp32_fused", f"[2] BN 접음 {self.n_fused}개 ")

        self.n_quant = quant.quantize_convs(self.model, calib_method=self.args.calib_method)
        calib_loader = make_calibration_loader(self.train_loader, self.args.calib_no_aug)
        n_calib_batches = min(self.args.calib_batches, len(calib_loader))
        print(f"[calib] method={self.args.calib_method}, augmentation={'off' if self.args.calib_no_aug else 'on'}, "
              f"{n_calib_batches} batches x {self.args.batch} = {n_calib_batches * self.args.batch} images, "
              f"seed={self.args.seed}")
        calib_batches = (x for i, (x, _) in zip(range(self.args.calib_batches), calib_loader))
        quant.calibrate(self.model, calib_batches, self.device)
        print(f"[3] {quant.quant_summary(self.model)}")
        self._eval("ptq_sim", "[3] PTQ 흉내        ")

    def train(self):
        """[4] 가짜 양자화 켠 채 학습. val 이 제일 좋았던 가중치를 마지막에 되돌려 놓음"""
        args, model = self.args, self.model
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: (1 - it / max(args.iters, 1)) ** 0.9)   # PolyLR

        best_miou = self.results.get("ptq_sim", {}).get("miou", -1)
        best_state = copy.deepcopy(model.state_dict())
        iteration = 0
        started = time.time()
        model.train()

        while iteration < args.iters:
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = F.cross_entropy(model(x), y, ignore_index=IGNORE)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()
                scheduler.step()
                iteration += 1

                if iteration % 10 == 0 or iteration == 1:
                    print(f"  iter {iteration}/{args.iters} loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} ({time.time() - started:.0f}s)")
                if self.val_loader and (iteration % args.eval_every == 0 or iteration == args.iters):
                    result = evaluate(model, self.val_loader, self.spec.num_classes, self.device)
                    print(f"  [val] iter {iteration}: {fmt(result)}")
                    if result["miou"] > best_miou:
                        best_miou, best_state = result["miou"], copy.deepcopy(model.state_dict())
                    model.train()
                if iteration >= args.iters:
                    break

        if self.val_loader:
            model.load_state_dict(best_state)
            self._eval("qat", "[4] QAT 최고       ")

    def export(self):
        """[5] Q/DQ ONNX 내보내기 + onnxruntime 비교 + json 저장"""
        import onnx
        import onnxruntime as ort
        args, spec = self.args, self.spec

        model = self.model.cpu().eval()
        quant.freeze_for_export(model)
        model.argmax = True
        dummy = torch.zeros(1, 3, spec.height, spec.width)
        replaced = freeze_adaptive_pools(model, dummy)
        if replaced:
            print(f"[5] AdaptiveAvgPool2d 교체 {len(replaced)}개")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        export(model, spec, args.out, args.opset)

        def count_weight_q(path) -> int:   # per-channel Q 노드 수 = 가중치 Q/DQ 수
            nodes = onnx.load(str(path)).graph.node
            return sum(1 for n in nodes if n.op_type == "QuantizeLinear" and any(a.name == "axis" for a in n.attribute))

        n_weight_q = count_weight_q(args.out)
        raw_bytes = args.out.read_bytes()
        simplify_and_check(args.out)
        if count_weight_q(args.out) != n_weight_q:   # onnxsim 이 가중치 Q/DQ 를 상수로 접었으면 원본으로 되돌림
            print("[5] onnxsim 이 Q/DQ 를 바꿔서 단순화 전 그래프를 씀")
            args.out.write_bytes(raw_bytes)
        op_types = {n.op_type for n in onnx.load(str(args.out)).graph.node}
        assert "QuantizeLinear" in op_types and "DequantizeLinear" in op_types, "Q/DQ 노드가 없음 — export 실패"
        print(f"[5] 가중치 Q/DQ {n_weight_q}개 (QuantConv2d {self.n_quant}개)")

        # ORT 의 int8 커널 최적화를 끄고 (Q/DQ 를 float 로 그대로 계산) torch 가짜 양자화와 비교
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(args.out), options, providers=["CPUExecutionProvider"])
        check_loader = DataLoader(self.val_loader.dataset if self.val_loader else self.train_loader.dataset, batch_size=1)
        agreements = []
        for x, _ in check_loader:
            with torch.no_grad():
                expected = model(x).numpy()
            actual = session.run(None, {"input": x.numpy()})[0]
            agreements.append(float((expected == actual).mean()) * 100)
            if len(agreements) >= 10:
                break
        print(f"[5] torch(가짜 양자화) vs onnxruntime(Q/DQ) 마스크 일치 평균 {np.mean(agreements):.3f}%  "
              "(반올림 경계에서 커널별 float 오차에 민감해 100% 는 안 나옴. 99% 이상 정상, 90% 아래면 export 문제)")

        spec.normalize_in_graph, spec.argmax_in_graph = True, True
        save_spec(args.out.with_suffix(args.out.suffix + ".json"), spec,
                  onnx_input="input", onnx_output="mask", opset=args.opset,
                  quantization="qat_qdq_int8", source_ckpt=str(args.ckpt), classes=self.classes,
                  qat=dict(iters=args.iters, lr=args.lr, batch=args.batch, calib_batches=args.calib_batches,
                           calib_method=args.calib_method,
                           calib_augmentation=not args.calib_no_aug, seed=args.seed, workers=args.workers,
                           conv_quantized=self.n_quant, bn_fused=self.n_fused, results=self.results),
                  torch_version=torch.__version__, exported_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        (args.out.parent / "qat_results.json").write_text(json.dumps(self.results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] {args.out} (+ .json)  → Jetson: python -m deploy.build_engine --onnx {args.out} --precision qat")


# ---------- main ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Fast-SCNN QAT INT8 → Q/DQ ONNX")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--model-py", type=Path, default=Path(__file__).with_name("fastscnn_mmseg.py"))
    parser.add_argument("--preprocess", type=Path, help="preprocess.json (생략하면 체크포인트 meta.cfg 에서)")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--img-dir", default="images")
    parser.add_argument("--ann-dir", default="labels_RGB")
    parser.add_argument("--img-suffix", default="_color.png")
    parser.add_argument("--ann-suffix", default="_label_RGB.png")
    parser.add_argument("--train-list", type=Path, default=None, help="기본: <data-root>/splits/train.txt")
    parser.add_argument("--val-list", type=Path, default=None, help="기본: <data-root>/splits/val.txt")
    parser.add_argument("--out", required=True, type=Path, help="예: models/fast_scnn_qat.onnx")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="원 학습(5e-4)보다 낮게. 1e-5~1e-4")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--calib-method", choices=("ema", "max"), default="ema",
                        help="활성화 scale 집계: ema=기존 지수이동평균, max=전체 관측 최대 절댓값")
    parser.add_argument("--calib-no-aug", action="store_true",
                        help="캘리브레이션 증강만 끔. QAT 학습 증강과 샘플링 설정은 유지")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2, help="Windows 면 0 권장")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[env] device {device}, torch {torch.__version__}")

    state_dict, meta = load_checkpoint(args.ckpt)
    spec = load_spec(args.preprocess) if args.preprocess else spec_from_mmseg_cfg(meta["cfg"])
    dataset_meta = meta.get("dataset_meta") or {}
    palette = [list(color) for color in dataset_meta.get("palette", DEFAULT_PALETTE_RGB)]
    classes = list(dataset_meta.get("classes", ["floor", "obstacle", "other"]))
    print(f"[spec] {spec.height}x{spec.width} {spec.color_order}, classes {classes}, palette(RGB) {palette}")

    trainer = QatTrainer(args, spec, classes, palette, device)
    trainer.load_data()
    trainer.build_model(state_dict)
    trainer.train()
    trainer.export()


if __name__ == "__main__":
    main()
