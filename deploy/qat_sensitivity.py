"""EMA 캘리브레이션을 한 번 수행한 뒤 블록/층별 양자화 민감도를 비교한다.

python -m deploy.qat_sensitivity --ckpt checkpoints/best_mIoU_iter_1600.pth \
    --data-root data --out results/qat_sensitivity_ema64 --device cpu

초기 5개 Conv와 특징 결합부 조합 진단: --scope downsample

학습/재캘리브레이션/ONNX export 없이, 지정 블록의 Conv 입력 및 가중치
가짜 양자화만 해제한다. 진단의 float 블록은 PyTorch FP32이며 TensorRT FP16이 아니다.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from deploy import quant
from deploy.qat_train import (DEFAULT_PALETTE_RGB, IGNORE, QatTrainer,
                              load_checkpoint, load_spec, spec_from_mmseg_cfg)


BLOCKS = {
    "float_downsample": ("초기 특징 추출만 FP32", "core.backbone.learning_to_downsample"),
    "float_global": ("전체 특징 추출만 FP32", "core.backbone.global_feature_extractor"),
    "float_fusion": ("특징 결합만 FP32", "core.backbone.feature_fusion"),
    "float_head": ("최종 분류부만 FP32", "core.decode_head"),
}


def metrics_from_confusion(cm, floor_id=0, obstacle_id=1):
    """행=정답, 열=예측. 오분류율은 전체 GT 장애물 픽셀을 분모로 사용."""
    cm = torch.as_tensor(cm, dtype=torch.int64, device="cpu")
    inter = cm.diag().double()
    union = cm.sum(0).double() + cm.sum(1).double() - inter
    present = union > 0
    iou = [float(inter[i] / union[i]) if present[i] else None for i in range(len(inter))]
    observed = [value for value in iou if value is not None]
    total, obstacle_pixels = int(cm.sum()), int(cm[obstacle_id].sum())
    return dict(miou=sum(observed) / len(observed) if observed else None,
                iou=iou, pixel_acc=float(inter.sum()) / total if total else None,
                obstacle_to_floor_pct=(100.0 * int(cm[obstacle_id, floor_id]) / obstacle_pixels
                                       if obstacle_pixels else None),
                valid_pixels=total, obstacle_pixels=obstacle_pixels, confusion=cm.tolist())


@torch.no_grad()
def evaluate_detailed(model, loader, num_classes, device, floor_id=0, obstacle_id=1):
    """기존 evaluate처럼 로더를 한 번 순회하며 방향별 오분류와 이미지별 결과도 기록."""
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    per_image = []
    items = getattr(loader.dataset, "items", None)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for expected, actual in zip(y, pred):
            valid = expected != IGNORE
            index = expected[valid] * num_classes + actual[valid]
            cm = torch.bincount(index, minlength=num_classes ** 2).view(num_classes, num_classes)
            confusion += cm
            image_index = len(per_image)
            name = str(items[image_index][0]) if items is not None else str(image_index)
            per_image.append(dict(image=name, **metrics_from_confusion(cm, floor_id, obstacle_id),
                                  mask_sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest()))
    if not per_image or int(confusion.sum()) == 0:
        raise ValueError("검증 이미지 또는 유효한 정답 픽셀이 없습니다.")
    return dict(**metrics_from_confusion(confusion, floor_id, obstacle_id), per_image=per_image)


def discover_blocks(model):
    names = [name for name, module in model.named_modules() if isinstance(module, quant.QuantConv2d)]
    groups = {}
    assigned = []
    for tag, (label, prefix) in BLOCKS.items():
        selected = [name for name in names if name == prefix or name.startswith(prefix + ".")]
        if not selected:
            raise ValueError(f"블록의 QuantConv2d를 찾지 못했습니다: {prefix}")
        groups[tag] = dict(label=label, prefix=prefix, names=selected)
        assigned.extend(selected)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(names):
        raise ValueError(f"블록에 속하지 않거나 중복된 QuantConv2d가 있습니다: {set(names) - set(assigned)}")
    return groups


def make_trials(groups, scope):
    """각 조건은 독립적 해제 목록이다. 앞 조건의 해제를 누적하지 않는다."""
    if scope == "blocks":
        return [(tag, group["label"], group["names"]) for tag, group in groups.items()]
    if scope != "downsample":
        raise ValueError(f"지원하지 않는 진단 범위: {scope}")
    downsample = groups["float_downsample"]
    fusion = groups["float_fusion"]
    # Conv 번호가 모델 변경으로 조용히 다른 층을 가리키지 않도록 이름까지 확인한다.
    suffixes = ("conv.conv", "dsconv1.depthwise_conv.conv", "dsconv1.pointwise_conv.conv",
                "dsconv2.depthwise_conv.conv", "dsconv2.pointwise_conv.conv")
    expected = [downsample["prefix"] + "." + suffix for suffix in suffixes]
    if downsample["names"] != expected or len(fusion["names"]) != 3:
        raise ValueError("초기 Conv 5개/특징 결합 Conv 3개 구성이 예상과 다릅니다. 모델 구조를 확인하세요.")
    trials = [(tag, groups[tag]["label"], groups[tag]["names"])
              for tag in ("float_downsample", "float_fusion")]
    for index, (name, suffix) in enumerate(zip(expected, suffixes), 1):
        trials.append((f"float_downsample_conv{index}", f"초기 Conv {index}만 FP32 ({suffix})", [name]))
    trials.append(("float_downsample_fusion", "초기 특징 추출 + 특징 결합 FP32",
                   downsample["names"] + fusion["names"]))
    return trials


@contextmanager
def unquantized(model, names):
    """선택 Conv의 입력+가중치 양자화만 해제하고 예외 발생 시에도 원복."""
    modules = dict(model.named_modules())
    selected = [modules[name] for name in names]
    if not all(isinstance(module, quant.QuantConv2d) for module in selected):
        raise ValueError("QuantConv2d만 해제할 수 있습니다.")
    previous = [module.input_quant.enabled for module in selected]
    try:
        for module in selected:
            module.input_quant.enabled = False
        yield
    finally:
        for module, enabled in zip(selected, previous):
            module.input_quant.enabled = enabled


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def quant_flags(model):
    return [(name, module.input_quant.enabled, module.input_quant.calibrating, module.w_scale_frozen)
            for name, module in model.named_modules() if isinstance(module, quant.QuantConv2d)]


def _number(value, digits=6):
    return "-" if value is None else f"{value:.{digits}f}"


def _print_metrics(label, result):
    print(f"{label} mIoU {_number(result['miou'])}  "
          f"IoU {[None if v is None else round(v, 6) for v in result['iou']]}  "
          f"장애물→바닥 {_number(result['obstacle_to_floor_pct'], 4)}%", flush=True)


class SensitivityTrainer(QatTrainer):
    def _eval(self, tag, label):
        # build_model의 FP32/BN-fused 검증 횟수와 시점을 그대로 유지해야
        # 뒤의 RandomSampler가 기존 실험과 같은 캘리브레이션 64장을 선택한다.
        if self.val_loader is None:
            raise ValueError("이 진단에는 비어 있지 않은 val 목록이 필요합니다.")
        result = evaluate_detailed(self.model, self.val_loader, self.spec.num_classes,
                                   self.device, self.classes.index("floor"), self.classes.index("obstacle"))
        self.results[tag] = result
        _print_metrics(label, result)


def run_sensitivity(trainer):
    """캘리브레이션 완료 모델에서 비교. 가중치/scale/관측 횟수는 바꾸지 않는다."""
    model = trainer.model.eval()
    groups = discover_blocks(model)
    scope = getattr(trainer.args, "scope", "blocks")
    trials = make_trials(groups, scope)
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, quant.QuantConv2d)}
    for name, module in modules.items():
        observer = module.input_quant
        if (int(observer.n_obs) != trainer.args.calib_batches or observer.calibrating
                or not observer.enabled or not bool(torch.isfinite(observer.scale).all())
                or not bool((observer.scale > 0).all())):
            raise RuntimeError(f"캘리브레이션 상태가 예상과 다릅니다: {name}")
    # 가중치가 바뀌지 않으므로 기존 동적 계산과 동일한 값을 한 번 저장한다.
    quant.freeze_for_export(model)
    initial_hash, initial_flags = state_fingerprint(model), quant_flags(model)
    floor_id, obstacle_id = trainer.classes.index("floor"), trainer.classes.index("obstacle")
    baseline = trainer.results["ptq_sim"]
    cases = [dict(tag="fp32_fused", label="BN 접은 FP32 기준", disabled_convs=len(modules),
                  disabled_names=list(modules), metrics=trainer.results["fp32_fused"]),
             dict(tag="all_quant", label="전체 가짜 양자화 (EMA 기준)", disabled_convs=0,
                  disabled_names=[], metrics=baseline)]
    trials.append(("all_float_control", "전체 양자화 해제 (복원 확인)", list(modules)))
    for index, (tag, label, names) in enumerate(trials, 1):
        print(f"[진단 {index}/{len(trials)}] {label}: Conv {len(names)}/{len(modules)}개 해제", flush=True)
        with unquantized(model, names):
            metrics = evaluate_detailed(model, trainer.val_loader, trainer.spec.num_classes,
                                        trainer.device, floor_id, obstacle_id)
        if state_fingerprint(model) != initial_hash or quant_flags(model) != initial_flags:
            raise RuntimeError(f"진단 후 모델 가중치/scale/양자화 설정이 달라졌습니다: {tag}")
        cases.append(dict(tag=tag, label=label, disabled_convs=len(names), disabled_names=names, metrics=metrics))
        _print_metrics(f"  [{tag}]", metrics)
    reference_masks = [row["mask_sha256"] for row in trainer.results["fp32_fused"]["per_image"]]
    control_masks = [row["mask_sha256"] for row in cases[-1]["metrics"]["per_image"]]
    control_ok = reference_masks == control_masks
    for case in cases:
        result = case["metrics"]
        def delta(a, b, factor=1.0):
            return None if a is None or b is None else (a - b) * factor
        case["delta_vs_all_quant"] = dict(
            miou_pp=delta(result["miou"], baseline["miou"], 100),
            floor_iou_pp=delta(result["iou"][floor_id], baseline["iou"][floor_id], 100),
            obstacle_to_floor_pp=delta(result["obstacle_to_floor_pct"], baseline["obstacle_to_floor_pct"]))
    scales = {name: dict(activation_scale=float(module.input_quant.scale),
                         n_obs=int(module.input_quant.n_obs)) for name, module in modules.items()}
    return dict(cases=cases, blocks=groups, scales=scales, scope=scope, state_sha256=initial_hash,
                invariance_passed=True, all_float_matches_fused_masks=control_ok)


def write_report(report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    classes = report["meta"]["classes"]
    floor_id, obstacle_id = classes.index("floor"), classes.index("obstacle")
    other_id = classes.index("other")
    scope = report.get("scope", "blocks")
    title = "초기 Conv별·블록 조합 양자화 민감도" if scope == "downsample" else "블록별 양자화 민감도"
    lines = ["# " + title, "", "PyTorch CPU 가짜 양자화 진단. 해제한 Conv는 FP32로 계산합니다.",
             "같은 EMA scale과 가중치를 유지하고 각 조건의 Conv 입력 및 가중치 양자화만 해제했습니다.",
             "각 조건은 전체 양자화 기준에서 독립적으로 시작하며, 해제 설정은 누적되지 않습니다.",
             "Δ는 전체 양자화 기준 대비 %p입니다. mIoU/바닥 IoU Δ는 양수가 개선, 오분류 Δ는 음수가 개선입니다.",
             "이 표는 TensorRT FP16/INT8 엔진 성능표가 아닙니다.", "",
             "| 조건 | 해제 Conv | mIoU | 바닥 IoU | 장애물 IoU | 그 외 IoU | 장애물→바닥 % | ΔmIoU %p | Δ바닥 IoU %p | Δ오분류 %p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in report["cases"]:
        r, d = case["metrics"], case["delta_vs_all_quant"]
        cells = [f"{case['tag']} ({case['label']})", str(case["disabled_convs"]), _number(r["miou"]),
                 _number(r["iou"][floor_id]), _number(r["iou"][obstacle_id]), _number(r["iou"][other_id]),
                 _number(r["obstacle_to_floor_pct"], 4), _number(d["miou_pp"], 4),
                 _number(d["floor_iou_pp"], 4), _number(d["obstacle_to_floor_pp"], 4)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"가중치·scale·양자화 상태 유지: {report['invariance_passed']}",
              f"전체 양자화 해제 시 BN-fused FP32 마스크와 완전 일치: {report['all_float_matches_fused_masks']}",
              "", "이미지별 지표, 혼동행렬, 마스크 해시, 블록의 Conv 목록과 scale은 JSON에 기록했습니다."]
    if scope == "downsample":
        lines += ["", "## 개별 해제 Conv 이름", "", "| 조건 | 해제한 Conv |", "|---|---|"]
        for case in report["cases"]:
            if case["tag"].startswith("float_downsample_conv"):
                lines.append(f"| {case['tag']} | `{case['disabled_names'][0]}` |")
    (out / "sensitivity_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Fast-SCNN 블록/층별 양자화 민감도 (EMA, 증강 ON, 학습 없음)")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="결과 폴더 (생략 시 scope별 폴더 사용)")
    parser.add_argument("--scope", choices=("blocks", "downsample"), default="blocks",
                        help="blocks: 기존 블록 비교, downsample: 초기 5개 Conv 개별 비교 + 특징 결합부 조합")
    parser.add_argument("--model-py", type=Path, default=Path(__file__).with_name("fastscnn_mmseg.py"))
    parser.add_argument("--preprocess", type=Path)
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--val-list", type=Path)
    parser.add_argument("--img-dir", default="images")
    parser.add_argument("--ann-dir", default="labels_RGB")
    parser.add_argument("--img-suffix", default="_color.png")
    parser.add_argument("--ann-suffix", default="_label_RGB.png")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    if args.out is None:
        args.out = Path("results/qat_sensitivity_downsample_ema64" if args.scope == "downsample"
                        else "results/qat_sensitivity_ema64")
    if args.batch < 1 or args.calib_batches < 1 or args.workers < 0:
        parser.error("batch/calib-batches는 양수, workers는 0 이상이어야 합니다.")
    args.calib_method, args.calib_no_aug, args.iters = "ema", False, 0
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[env] device {device}, torch {torch.__version__}", flush=True)
    print(f"[scope] {args.scope}; EMA, augmentation=on, iters=0", flush=True)
    state_dict, meta = load_checkpoint(args.ckpt)
    spec = load_spec(args.preprocess) if args.preprocess else spec_from_mmseg_cfg(meta["cfg"])
    dataset_meta = meta.get("dataset_meta") or {}
    classes = list(dataset_meta.get("classes", ["floor", "obstacle", "other"]))
    if spec.num_classes != 3 or len(classes) != 3 or set(classes) != {"floor", "obstacle", "other"}:
        raise ValueError("이 진단은 floor/obstacle/other 세 클래스를 대상으로 합니다.")
    palette = [list(color) for color in dataset_meta.get("palette", DEFAULT_PALETTE_RGB)]
    trainer = SensitivityTrainer(args, spec, classes, palette, device)
    trainer.load_data()
    if trainer.val_loader is None or len(trainer.val_loader.dataset) == 0:
        raise ValueError("비어 있지 않은 검증 목록이 필요합니다: data/splits/val.txt 또는 --val-list")
    if len(trainer.train_loader) < args.calib_batches:
        raise ValueError(f"캘리브레이션에 {args.calib_batches}배치가 필요하지만 {len(trainer.train_loader)}배치만 있습니다.")
    trainer.build_model(state_dict)
    report = run_sensitivity(trainer)
    report["meta"] = dict(created_at=time.strftime("%Y-%m-%d %H:%M:%S"), torch_version=torch.__version__,
                          checkpoint=str(args.ckpt), device=str(device), classes=classes, spec=spec.to_dict(),
                          scope=args.scope,
                          batch=args.batch, workers=args.workers, seed=args.seed, calib_method="ema",
                          augmentation=True, calib_batches=args.calib_batches,
                          calib_images=args.calib_batches * args.batch, iters=0,
                          train_images=len(trainer.train_loader.dataset), val_images=len(trainer.val_loader.dataset),
                          quantized_convs=trainer.n_quant, bn_fused=trainer.n_fused)
    write_report(report, args.out)
    if not report["all_float_matches_fused_masks"]:
        raise SystemExit(f"[check] FP32 복원 결과가 다릅니다. {args.out}의 결과를 확인하세요.")
    print("[check] 가중치/scale 유지 및 FP32 마스크 복원 확인 통과", flush=True)
    print(f"[done] {args.out / 'sensitivity_table.md'}", flush=True)
    print(f"[done] {args.out / 'sensitivity_results.json'}", flush=True)


if __name__ == "__main__":
    main()
