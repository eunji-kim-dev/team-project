"""카메라 앱의 판정 기준(클래스 + softmax 신뢰도 ≥ 기준값)으로 본 모델 간 일치율. Jetson 에서 실행

python -m eval.confidence_agreement \\
    --models cpu_fp32=models/logits/fast_scnn_logits.onnx gpu_fp32=models/logits/fast_scnn_logits_fp32.engine \\
             int8=models/calib_trt_adds/fast_scnn_qat_logits_int8.engine \\
    --pairs int8:gpu_fp32 --images data/images --gt data/labels_RGB --list data/splits/val.txt \\
    --out results/confidence_agreement_1101

- 성능표(bench_engines)의 일치율은 픽셀 argmax 클래스만 비교한다. 카메라 앱(camera/D435_Fusion_v3.5)은
  "클래스가 c 이고 softmax 신뢰도가 기준값 이상"인 점만 쓰므로, 클래스가 같아도 신뢰도가 기준 근처에서
  흔들리면 판정이 뒤집힌다. 이 도구는 그 뒤집힘을 앱이 쓰는 기준값마다 센다
- 전처리·softmax 디코드는 앱 코드(d435map.semantics.prepare_input / decode_prediction)를 그대로 import 해서 쓰고,
  경계 1픽셀 제외는 fusion.sample_semantics 와 같은 규칙으로 적용한다. 기준값은 앱 config.json 의 fusion 블록에서 읽는다
- 모델: .engine → 앱의 TrtBackend (로짓 엔진 또는 decode_in_graph 엔진), .onnx → onnxruntime CPU (cpu_fp32 기준행),
  .pth → 앱의 TorchBackend (PyTorch CPU, 앱이 PC 에서 쓰던 경로)
- 첫 모델이 기준. 기준 vs 나머지 모델을 전부 비교하고, --pairs a:b 로 다른 쌍을 더 비교한다
- 지표 (쌍마다, 판정마다): 전체 픽셀 일치%, 판정 마스크 IoU, 뒤집힘% (= 다른 픽셀 / 기준 모델이 통과시킨 픽셀),
  잃음(기준 통과→후보 탈락)·얻음(기준 탈락→후보 통과), 기준 통과 픽셀에서의 Δ신뢰도 부호 평균(후보−기준).
  쌍 전체로는 argmax 일치%, |Δ신뢰도| 평균/p99/최대와 부호 평균
- 정답지(--gt)가 있으면 모델·판정마다 재현율(정답 c 중 통과%), 정밀도(통과 중 정답 c%), 위험 누출%
  (바닥 판정: 정답 장애물 중 바닥으로 통과한 비율. 장애물/그 외 판정: 정답 바닥 중 장애물로 통과한 비율)
- 시간은 재지 않는다 (클럭 고정 불필요). 픽셀 단위 비교라 깊이 점 샘플링·평면 검증은 반영하지 않는다
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.bench_engines import DEFAULT_PALETTE_RGB, GroundTruth, IGNORE, collect_env  # noqa: E402

DEFAULT_APP_DIR = REPO / "camera" / "D435_Fusion_v3.5"
CLASS_KO = {0: "바닥", 1: "장애물", 2: "그 외"}
# (키, 이름, 클래스, config fusion 키, 앱 코드 위치)
DECISIONS = [
    ("floor", "바닥 채택", 0, "floor_confidence", "fusion.py classify_points / fit_fused_floor"),
    ("obstacle", "장애물 채택", 1, "obstacle_confidence", "fusion.py classify_points, candidate_review.py"),
    ("other", "그 외 채택", 2, "obstacle_confidence", "fusion.py classify_points"),
    ("held", "깊이 장애물·SCNN 바닥 충돌 보류", 0, "floor_obstacle_confidence", "fusion.py classify_points"),
    ("free", "의미적 자유공간 내부", 0, "semantic_free_confidence", "semantic_free.py projected_floor_free"),
    ("continuity", "자유공간 연속성", 0, "semantic_free_continuity_confidence", "semantic_free.py projected_floor_free"),
]
NEAR = 0.02   # 기준값 ± 이 폭 안의 픽셀 비율 (뒤집힘이 왜 생기는지 보려고)


def import_app(app_dir: Path):
    if not (app_dir / "d435map" / "semantics.py").exists():
        raise SystemExit(f"카메라 앱이 없음: {app_dir}")
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    from d435map import semantics, trt_fastscnn   # noqa: F401
    return semantics, trt_fastscnn


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------- 모델 (앱 백엔드 재사용) ----------

class OnnxCpu:
    """cpu_fp32 기준행. 매니페스트 검사는 앱의 TrtBackend 와 같은 함수"""

    def __init__(self, path: Path, spec: dict, threads: int, check_manifest):
        manifest_path = path.with_suffix(path.suffix + ".json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"매니페스트 없음: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        check_manifest(manifest, spec, path)
        self.normalize_in_graph = bool(manifest.get("normalize_in_graph", False))
        import onnxruntime as ort
        from deploy.segmenter import OnnxSegmenter
        self.segmenter = OnnxSegmenter(path, providers=["CPUExecutionProvider"], threads=threads)
        self.label = f"onnxruntime {ort.__version__} CPU FP32 ({threads} threads)"

    def describe(self):
        return self.label

    def __call__(self, array):
        return np.asarray(self.segmenter.run(array), dtype=np.float32)

    def close(self):
        self.segmenter = None


class TorchCpu:
    def __init__(self, path: Path, threads: int):
        import torch
        from d435map.torch_fastscnn import TorchBackend
        self.backend = TorchBackend(path, threads)
        self.normalize_in_graph = False
        self.label = f"PyTorch {torch.__version__} CPU FP32 ({threads} threads)"

    def describe(self):
        return self.label

    def __call__(self, array):
        return np.asarray(self.backend(array), dtype=np.float32)

    def close(self):
        self.backend = None


def open_model(path: Path, spec: dict, threads: int, trt_fastscnn):
    suffix = path.suffix.lower()
    if suffix == ".engine":
        return trt_fastscnn.TrtBackend(path, spec)
    if suffix == ".onnx":
        return OnnxCpu(path, spec, threads, trt_fastscnn._check_manifest)
    if suffix == ".pth":
        return TorchCpu(path, threads)
    raise SystemExit(f"모르는 모델 확장자: {path}")


def interior_mask(labels: np.ndarray, radius: int) -> np.ndarray:
    """fusion.sample_semantics 와 같은 규칙: 클래스별로 radius 픽셀 침식한 영역의 합. 경계 픽셀은 앱이 쓰지 않는다"""
    if not radius:
        return np.ones(labels.shape, bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    interior = np.zeros(labels.shape, bool)
    for ident in (0, 1, 2):
        interior |= cv2.erode((labels == ident).astype(np.uint8), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return interior


# ---------- 비교 ----------

def decision_mask(labels, conf, interior, class_id, threshold):
    return (labels == class_id) & (conf >= np.float32(threshold)) & interior


def compare_pair(ref: dict, cand: dict, decisions: list, thresholds: dict, names: list) -> tuple[dict, list]:
    n = len(names)
    total = same = 0
    signed_sum = 0.0
    dconf_all = []
    acc = {key: dict(inter=0, union=0, ref_pos=0, cand_pos=0, lost=0, gained=0, near_ref=0, signed_on_ref=0.0) for key, *_ in decisions}
    rows = []
    for i in range(n):
        rl, rc, ri = ref["labels"][i], ref["conf"][i], ref["interior"][i]
        cl, cc, ci = cand["labels"][i], cand["conf"][i], cand["interior"][i]
        equal = rl == cl
        same += int(equal.sum())
        total += equal.size
        signed = cc - rc   # 후보 − 기준. 음수면 후보(INT8) 신뢰도가 낮다
        dconf = np.abs(signed)
        signed_sum += float(signed.sum())
        dconf_all.append(dconf.ravel())
        row = {"image": names[i], "argmax_agree_pct": round(float(equal.mean()) * 100, 3),
               "dconf_mean": round(float(dconf.mean()), 5), "dconf_p99": round(float(np.percentile(dconf, 99)), 5),
               "dconf_max": round(float(dconf.max()), 5)}
        for key, _, class_id, cfg_key, _ in decisions:
            t = thresholds[cfg_key]
            a = decision_mask(rl, rc, ri, class_id, t)
            b = decision_mask(cl, cc, ci, class_id, t)
            inter, union = int((a & b).sum()), int((a | b).sum())
            lost, gained = int((a & ~b).sum()), int((~a & b).sum())
            near = int(((rl == class_id) & ri & (np.abs(rc - t) < NEAR)).sum())
            d = acc[key]
            d["inter"] += inter; d["union"] += union; d["ref_pos"] += int(a.sum()); d["cand_pos"] += int(b.sum())
            d["lost"] += lost; d["gained"] += gained; d["near_ref"] += near
            d["signed_on_ref"] += float(signed[a].sum())
            row[f"{key}_ref_pos"] = int(a.sum()); row[f"{key}_lost"] = lost; row[f"{key}_gained"] = gained
        rows.append(row)
    dconf_all = np.concatenate(dconf_all)
    result = {"ref": ref["tag"], "cand": cand["tag"], "pixels": total,
              "argmax_agree_pct": round(same / max(total, 1) * 100, 3),
              "dconf": {"mean": round(float(dconf_all.mean()), 5), "p99": round(float(np.percentile(dconf_all, 99)), 5),
                        "max": round(float(dconf_all.max()), 5), "signed_mean": round(signed_sum / max(total, 1), 5)},
              "decisions": {}}
    for key, name, class_id, cfg_key, where in decisions:
        d = acc[key]
        changed = d["lost"] + d["gained"]
        result["decisions"][key] = {
            "name": name, "class": class_id, "threshold": thresholds[cfg_key], "config_key": cfg_key, "where": where,
            "pixel_agree_pct": round((total - changed) / max(total, 1) * 100, 3),
            "mask_iou": round(d["inter"] / d["union"], 4) if d["union"] else None,
            "flip_pct_of_ref": round(changed / d["ref_pos"] * 100, 3) if d["ref_pos"] else None,
            "lost_pct_of_ref": round(d["lost"] / d["ref_pos"] * 100, 3) if d["ref_pos"] else None,
            "gained_pct_of_ref": round(d["gained"] / d["ref_pos"] * 100, 3) if d["ref_pos"] else None,
            "near_threshold_pct_of_ref": round(d["near_ref"] / d["ref_pos"] * 100, 3) if d["ref_pos"] else None,
            "dconf_signed_mean_on_ref_pass": round(d["signed_on_ref"] / d["ref_pos"], 5) if d["ref_pos"] else None,
            **d}
    return result, rows


def gt_metrics(model: dict, gts: list, decisions: list, thresholds: dict) -> dict:
    out = {}
    for key, name, class_id, cfg_key, _ in decisions:
        t = thresholds[cfg_key]
        tp = passed = gt_c = leak = leak_base = 0
        for labels, conf, interior, gt in zip(model["labels"], model["conf"], model["interior"], gts):
            if gt is None:
                continue
            valid = gt != IGNORE
            m = decision_mask(labels, conf, interior, class_id, t) & valid
            is_c = (gt == class_id) & valid
            tp += int((m & is_c).sum()); passed += int(m.sum()); gt_c += int(is_c.sum())
            danger = (gt == 1) if class_id == 0 else (gt == 0)   # 바닥 판정엔 정답 장애물이, 장애물 판정엔 정답 바닥이 위험
            danger &= valid
            leak += int((m & danger).sum()); leak_base += int(danger.sum())
        out[key] = {"name": name, "class": class_id, "threshold": t,
                    "recall_pct": round(tp / gt_c * 100, 3) if gt_c else None,
                    "precision_pct": round(tp / passed * 100, 3) if passed else None,
                    "leak_pct": round(leak / leak_base * 100, 3) if leak_base else None,
                    "leak_meaning": "정답 장애물 중 바닥으로 통과%" if class_id == 0 else "정답 바닥 중 장애물/그 외로 통과%",
                    "passed": passed, "gt_pixels": gt_c}
    return out


# ---------- 출력 ----------

def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def pair_table(result: dict) -> str:
    lines = [f"**{result['cand']} vs {result['ref']}** — argmax 일치 {result['argmax_agree_pct']:.2f}%, "
             f"|Δ신뢰도| 평균 {result['dconf']['mean']:.4f} / p99 {result['dconf']['p99']:.3f} / 최대 {result['dconf']['max']:.3f}, "
             f"부호 평균(후보−기준) {result['dconf']['signed_mean']:+.4f}",
             "",
             "| 판정 | 클래스 | 기준 | 전체 픽셀 일치% | 판정 IoU | 뒤집힘% | 잃음% | 얻음% | 기준 ±0.02 안% | 기준 통과 픽셀(만) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for d in result["decisions"].values():
        lines.append(f"| {d['name']} | {CLASS_KO[d['class']]} | {d['threshold']:.2f} | {fmt(d['pixel_agree_pct'])} | "
                     f"{fmt(d['mask_iou'], 4)} | {fmt(d['flip_pct_of_ref'])} | {fmt(d['lost_pct_of_ref'])} | "
                     f"{fmt(d['gained_pct_of_ref'])} | {fmt(d['near_threshold_pct_of_ref'])} | {d['ref_pos'] / 1e4:.1f} |")
    return "\n".join(lines)


def gt_table(models: list, decisions: list) -> str:
    tags = [m["tag"] for m in models]
    head = "| 판정 | 기준 | " + " | ".join(f"{t} 재현율% / 정밀도% / 위험 누출%" for t in tags) + " |"
    lines = [head, "|---|---|" + "---|" * len(tags)]
    for key, name, class_id, cfg_key, _ in decisions:
        cells = []
        for m in models:
            g = m["gt"][key]
            cells.append(f"{fmt(g['recall_pct'])} / {fmt(g['precision_pct'])} / {fmt(g['leak_pct'])}")
        lines.append(f"| {name} ({CLASS_KO[class_id]}) | {m['gt'][key]['threshold']:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("위험 누출: 바닥 판정은 정답 장애물 픽셀 중 바닥으로 통과한 비율(부딪힘), 장애물/그 외 판정은 정답 바닥 픽셀 중 통과한 비율(못 지나감)")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="카메라 앱 신뢰도 기준으로 본 모델 간 일치율")
    parser.add_argument("--models", nargs="+", required=True, help="이름=경로 (.onnx/.engine/.pth). 첫 모델이 기준")
    parser.add_argument("--pairs", nargs="*", default=[], help="추가 비교 쌍 후보:기준 (예: int8:gpu_fp32)")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--list", type=Path, default=None)
    parser.add_argument("--img-suffix", default="_color.png")
    parser.add_argument("--gt-suffix", default="_label_RGB.png")
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR, help="카메라 앱 폴더 (config.json, models/standard_fastscnn.json)")
    parser.add_argument("--config", type=Path, default=None, help="기준값을 읽을 config.json (기본: 앱 폴더의 config.json)")
    parser.add_argument("--threads", type=int, default=4, help="CPU 모델 스레드 수")
    parser.add_argument("--limit", type=int, default=0, help="앞 N장만 (동작 확인용)")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    semantics, trt_fastscnn = import_app(args.app_dir)
    spec_path = args.app_dir / "models" / semantics.STANDARD_SPEC_NAME
    spec = semantics.validate_spec(json.loads(spec_path.read_text(encoding="utf-8-sig")))
    config_path = args.config or (args.app_dir / "config.json")
    fusion_cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))["fusion"]
    thresholds = {cfg_key: float(fusion_cfg[cfg_key]) for *_, cfg_key, _ in DECISIONS}
    erode = int(fusion_cfg.get("boundary_erode_px", 0))

    models = []
    for item in args.models:
        tag, _, path = item.partition("=") if "=" in item else (Path(item).stem, "", item)
        models.append({"tag": tag, "path": Path(path)})
    tags = [m["tag"] for m in models]
    if len(set(tags)) != len(tags):
        raise SystemExit(f"모델 이름 중복: {tags}")
    pairs = [(tags[0], t) for t in tags[1:]]
    for item in args.pairs:
        cand, _, ref = item.partition(":")
        if cand not in tags or ref not in tags:
            raise SystemExit(f"--pairs {item}: 모델 이름이 아님 {tags}")
        if (ref, cand) not in pairs:
            pairs.append((ref, cand))

    names = None
    if args.list:
        names = [line.split()[0] for line in args.list.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
    data = GroundTruth(args.images, args.gt, args.img_suffix, args.gt_suffix, 3, DEFAULT_PALETTE_RGB)
    images = data.list_images(names)
    if args.limit:
        images = images[: args.limit]
    gts = [data.load(p) for p in images]
    sample_names = [data.sample_name(p) for p in images]
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in images]
    if any(f is None for f in frames):
        raise SystemExit("읽지 못한 사진이 있음")
    print(f"[data] {len(images)}장, 정답 {sum(g is not None for g in gts)}장, 경계 제외 {erode}px, 기준값 {thresholds}")

    env = collect_env()
    for m in models:
        started = time.perf_counter()
        backend = open_model(m["path"], spec, args.threads, trt_fastscnn)
        normalize = not backend.normalize_in_graph
        m["labels"], m["conf"], m["interior"] = [], [], []
        for bgr in frames:   # 앱의 predict() 와 같은 순서
            tensor = semantics.prepare_input(bgr, spec, normalize)
            output = backend(tensor)
            if isinstance(output, tuple):   # decode_in_graph 엔진: GPU 가 argmax·신뢰도를 냄
                labels, conf = semantics.decode_in_graph_result(output, spec, bgr.shape[:2])
            else:
                labels, conf = semantics.decode_prediction(output, spec, bgr.shape[:2])
            m["labels"].append(labels); m["conf"].append(conf); m["interior"].append(interior_mask(labels, erode))
        m["backend"] = backend.describe()
        m["normalize_on_cpu"] = normalize
        m["sha256_16"] = sha16(m["path"])
        backend.close()
        if args.gt:
            m["gt"] = gt_metrics(m, gts, DECISIONS, thresholds)
        print(f"[model] {m['tag']}: {m['path']} ({m['backend']}, sha {m['sha256_16']}) {time.perf_counter() - started:.1f}s")

    by_tag = {m["tag"]: m for m in models}
    results, per_image = [], []
    for ref, cand in pairs:
        result, rows = compare_pair(by_tag[ref], by_tag[cand], DECISIONS, thresholds, sample_names)
        results.append(result)
        for row in rows:
            per_image.append({"pair": f"{cand} vs {ref}", **row})
        print("\n" + pair_table(result))

    args.out.mkdir(parents=True, exist_ok=True)
    md = [f"# 카메라 앱 신뢰도 기준 일치율 ({env['datetime']})", "",
          f"- 사진 {len(images)}장 ({args.list or args.images}), 기준값 {config_path} fusion 블록, 경계 제외 {erode}px, 표준 사양 {spec_path}",
          "- 모델: " + "; ".join(f"{m['tag']} = `{m['path']}` ({m['backend']}, sha {m['sha256_16']})" for m in models), ""]
    for result in results:
        md += [pair_table(result), ""]
    if args.gt:
        md += ["## 정답지 기준 (판정 마스크 자체의 정확도)", "", gt_table(models, DECISIONS), ""]
        print("\n" + gt_table(models, DECISIONS))
    (args.out / "confidence_agreement.md").write_text("\n".join(md), encoding="utf-8")

    payload = {"env": env, "args": vars(args) | {"out": str(args.out), "images": str(args.images), "gt": str(args.gt),
                                                   "list": str(args.list), "app_dir": str(args.app_dir), "config": str(config_path)},
               "spec": spec, "thresholds": thresholds, "boundary_erode_px": erode, "near_band": NEAR,
               "decisions": [dict(key=k, name=n, class_id=c, config_key=ck, where=w) for k, n, c, ck, w in DECISIONS],
               "images": sample_names,
               "models": [{k: m[k] for k in ("tag", "backend", "normalize_on_cpu", "sha256_16")} | {"path": str(m["path"])} | ({"gt": m["gt"]} if "gt" in m else {}) for m in models],
               "pairs": results}
    (args.out / "confidence_agreement.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if per_image:
        with open(args.out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            writer.writeheader(); writer.writerows(per_image)
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
