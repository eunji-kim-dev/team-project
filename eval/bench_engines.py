"""모델별 속도·정확도·메모리 비교 → 성능표. Jetson 에서 실행

python -m eval.bench_engines \\
    --models cpu_fp32=models/fast_scnn.onnx gpu_fp32=models/fast_scnn_fp32.engine fp16=models/fast_scnn_fp16.engine \\
             ptq_int8=models/fast_scnn_int8.engine qat_int8=models/fast_scnn_qat_int8.engine \\
    --images data/images --gt data/labels_RGB --list data/splits/val.txt \\
    --out results/bench_1101 --runs 5 --cooldown 30 --save-masks

재는 것
  속도    프레임당 전체(pre+infer+post) p50/p95 ms, infer 만 p50/p95, FPS(=1000/p50), 33.3ms 넘긴 비율
  정확도  정답 있으면 클래스별 IoU·mIoU·픽셀 정확도, 기준 모델(기본 cpu_fp32)과 마스크 일치율
  메모리  파일 MB, 엔진 활성화 메모리 MB(TensorRT), 로드 후 RSS 증가 MB
          (보드 전체 RAM·전력·온도는 tegrastats_log.sh + parse_tegrastats.py 로 따로)

- "이름=" 생략 가능. 파일명에서 추정 (.onnx→cpu_fp32, _fp32→gpu_fp32, _int8→ptq_int8, qat→qat_int8)
- .onnx 는 기본으로 onnxruntime CPU 로 돎 (= cpu_fp32 행). --onnx-cuda 주면 CUDA
- 사진 <이름>_color.png, 정답 <이름>_label_RGB.png (색 라벨은 classes.json 팔레트로 번호 변환, 중간색은 가장 가까운 색)
- 정확도엔 "장애물→바닥%" (정답 장애물 중 바닥으로 찍은 비율 = 코스트맵에서 부딪히는 오류) 와 최악 5장도 같이 냄
- --target-fps 30 을 주면 카메라 속도로 맞춰 돎 → 전력·온도를 "같은 처리 속도" 조건에서 비교할 때 씀
- 표의 "반복 변동%" 는 5회 p50 의 (최대−최소)/중간값. 1~2% 안이면 믿을 만 (팀 문서 기준)
- 모델별 raw_frame_log_<모델>.csv 에 프레임마다 pre/infer/post/total ms 를 남김 (역할 4 의 전체 로그와 합치는 용)
- 장시간 안정성(스로틀링·지연 증가)은 eval/soak_test.py 로 따로
- 출력: <out>/bench_table.md (성능표), bench_results.json (전부), masks/<모델>/*.png (--save-masks)
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.segmenter import create_segmenter, TrtSegmenter  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_CLASS_NAMES = ["floor", "obstacle", "other"]
DEFAULT_PALETTE_RGB = [[0, 255, 0], [255, 0, 0], [0, 0, 255]]
CLASS_KO = {"floor": "바닥", "obstacle": "장애물", "other": "그 외"}
IGNORE = 255


# ---------- 환경 기록 ----------

def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def collect_env() -> dict:
    """결과 json 에 같이 남길 환경 정보. 나중에 "그때 전력 모드가 뭐였지" 안 헷갈리게"""
    env = {
        "hostname": platform.node(),
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "git_commit": run_cmd(["git", "rev-parse", "--short", "HEAD"]) or "unknown",
        "git_dirty": bool(run_cmd(["git", "status", "--porcelain"])),
    }
    try:
        import tensorrt as trt
        env["tensorrt"] = trt.__version__
    except ImportError:
        env["tensorrt"] = None
    try:
        import onnxruntime as ort
        env["onnxruntime"] = ort.__version__
    except ImportError:
        env["onnxruntime"] = None

    release_file = Path("/etc/nv_tegra_release")
    env["l4t"] = release_file.read_text().splitlines()[0].strip() if release_file.exists() else None
    env["nvpmodel"] = run_cmd(["nvpmodel", "-q"]).replace("\n", " ") or "unknown"

    # jetson_clocks --show 의 GPU 줄에서 MinFreq == MaxFreq 면 클럭 고정 상태 (root 필요)
    env["gpu_clock_locked"] = "unknown"
    for line in run_cmd(["jetson_clocks", "--show"]).splitlines():
        if line.strip().startswith("GPU"):
            fields = dict(part.split("=") for part in line.split() if "=" in part)
            if "MinFreq" in fields and "MaxFreq" in fields:
                env["gpu_clock_locked"] = fields["MinFreq"] == fields["MaxFreq"]
                env["gpu_max_freq_mhz"] = int(fields["MaxFreq"]) // 1_000_000
    return env


def rss_mb() -> float:
    """이 프로세스의 실제 메모리 사용량 MB"""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def prime_runtimes(models: list) -> None:
    """라이브러리 로드·CUDA 컨텍스트 메모리가 첫 모델의 RSS 증가에 섞이지 않게 미리 한 번 띄움"""
    suffixes = {path.suffix.lower() for _, path in models}
    if suffixes & {".engine", ".trt", ".plan"}:
        import tensorrt as trt
        from deploy import cuda_util as cu
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
        buffer = cu.DeviceBuffer(1 << 20)   # 첫 cudaMalloc 이 CUDA 컨텍스트를 만듦
        cu.pinned_empty((4,), np.float32)
        cu.Stream().synchronize()
        buffer.free()
    if ".onnx" in suffixes:
        import onnxruntime  # noqa: F401


# ---------- 데이터 ----------

class GroundTruth:
    """사진 목록과 정답 마스크 읽기. 역할 1 데이터 형식(_color.png / _label_RGB.png) 기본값"""

    def __init__(self, img_dir: Path, gt_dir: Path | None, img_suffix: str, gt_suffix: str,
                 num_classes: int, palette_rgb: list):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_suffix = img_suffix
        self.gt_suffix = gt_suffix
        self.num_classes = num_classes
        self.palette = palette_rgb

    def list_images(self, names: list[str] | None) -> list[Path]:
        if names:   # --list 로 받은 이름 목록
            files = [self.img_dir / f"{name}{self.img_suffix}" for name in names]
            missing = [str(f) for f in files if not f.exists()]
            if missing:
                raise SystemExit(f"목록에 있는데 없는 사진 {len(missing)}장. 예: {missing[:3]}")
            return files
        files = sorted(p for p in self.img_dir.iterdir() if p.name.endswith(self.img_suffix))
        if not files:   # 접미사 안 맞으면 폴더의 모든 이미지
            files = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise SystemExit(f"{self.img_dir} 에 이미지가 없음")
        return files

    def sample_name(self, img_path: Path) -> str:
        if img_path.name.endswith(self.img_suffix):
            return img_path.name[: -len(self.img_suffix)]
        return img_path.stem

    def color_to_index(self, mask_bgr: np.ndarray) -> np.ndarray:
        """색 라벨 → 클래스 번호. 중간색은 가장 가까운 팔레트 색 (역할 1 rgb_mask_loader 와 같은 규칙)"""
        rgb = mask_bgr[:, :, ::-1].astype(np.int32)
        palette = np.asarray(self.palette, dtype=np.int32)
        dist = ((rgb[:, :, None, :] - palette) ** 2).sum(-1)
        return dist.argmin(-1).astype(np.uint8)

    def load(self, img_path: Path):
        """정답 마스크 (없으면 None)"""
        if self.gt_dir is None:
            return None
        name = self.sample_name(img_path)
        candidates = (self.gt_dir / f"{name}{self.gt_suffix}", self.gt_dir / f"{name}.png", self.gt_dir / f"{img_path.stem}.png")
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None
        if mask.ndim == 3:
            return self.color_to_index(mask[:, :, :3])
        bad_values = [int(v) for v in np.unique(mask) if v != IGNORE and v >= self.num_classes]
        if bad_values:
            raise SystemExit(f"{path}: 클래스 번호가 아닌 값 {bad_values} 이 있음")
        return mask.astype(np.uint8)


# ---------- 지표 ----------

class SegMetrics:
    """혼동행렬 기반 IoU 와 기준 모델 대비 일치율"""

    def __init__(self, num_classes: int):
        self.n = num_classes

    def confusion(self, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        valid = gt != IGNORE
        index = gt[valid].astype(np.int64) * self.n + pred[valid].astype(np.int64)
        return np.bincount(index, minlength=self.n * self.n).reshape(self.n, self.n)

    def iou_from_confusion(self, cm: np.ndarray) -> dict:
        inter = np.diag(cm).astype(np.float64)
        union = cm.sum(0) + cm.sum(1) - np.diag(cm)
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(union > 0, inter / union, np.nan)
        result = {"iou": [None if np.isnan(v) else round(float(v), 4) for v in iou],
                  "miou": round(float(np.nanmean(iou)), 4),
                  "pixel_acc": round(float(inter.sum() / max(cm.sum(), 1)), 4)}
        if self.n >= 2:   # 코스트맵에서 위험한 오류 두 가지. cm[정답, 예측]. 0 floor, 1 obstacle
            gt_obstacle = max(cm[1].sum(), 1)
            gt_floor = max(cm[0].sum(), 1)
            result["obstacle_to_floor_pct"] = round(float(cm[1, 0] / gt_obstacle) * 100, 3)   # 장애물을 바닥으로 (미탐. 부딪힘)
            result["floor_to_obstacle_pct"] = round(float(cm[0, 1] / gt_floor) * 100, 3)      # 바닥을 장애물로 (오탐. 못 지나감)
        return result

    def per_image(self, names: list[str], gts: list, masks: list) -> dict:
        """이미지별 mIoU → 최악 5장 (어떤 사진에서 틀리는지 보려고)"""
        scores = []
        for name, gt, mask in zip(names, gts, masks):
            if gt is None:
                continue
            r = self.iou_from_confusion(self.confusion(gt, mask))
            scores.append({"name": name, "miou": r["miou"], "obstacle_to_floor_pct": r.get("obstacle_to_floor_pct")})
        scores.sort(key=lambda x: x["miou"])
        return {"worst5": scores[:5], "all": scores}

    def agreement(self, ref_masks: list, masks: list) -> dict:
        """기준 모델 마스크와 비교: 전체 픽셀 일치율, 클래스별(기준이 c 라고 한 픽셀 중 같은 비율), 클래스별 IoU"""
        same = total = 0
        per_class_same = np.zeros(self.n)
        per_class_total = np.zeros(self.n)
        inter = np.zeros(self.n)
        union = np.zeros(self.n)
        for ref, mask in zip(ref_masks, masks):
            equal = ref == mask
            same += int(equal.sum())
            total += equal.size
            for c in range(self.n):
                ref_c, mask_c = ref == c, mask == c
                per_class_total[c] += ref_c.sum()
                per_class_same[c] += (ref_c & equal).sum()
                inter[c] += (ref_c & mask_c).sum()
                union[c] += (ref_c | mask_c).sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            per_class = np.where(per_class_total > 0, per_class_same / per_class_total, np.nan)
            class_iou = np.where(union > 0, inter / union, np.nan)
        return {"pixel_agree_pct": round(same / max(total, 1) * 100, 3),
                "class_agree_pct": [None if np.isnan(v) else round(float(v) * 100, 3) for v in per_class],
                "class_iou_vs_ref": [None if np.isnan(v) else round(float(v), 4) for v in class_iou]}


def percentile(values, q):
    return round(float(np.percentile(values, q)), 3) if len(values) else None


# ---------- 모델 하나 측정 ----------

class ModelBenchmark:
    """모델 하나를 로드해서 워밍업 → runs 회 반복 측정 → 지표 dict 와 마스크 목록 반환"""

    def __init__(self, images: list[Path], gts: list, metrics: SegMetrics, args, onnx_kwargs: dict):
        self.images = images
        self.gts = gts
        self.metrics = metrics
        self.args = args
        self.onnx_kwargs = onnx_kwargs
        self.frames = None   # 디스크 읽기 시간이 안 섞이게 사진을 메모리에 올려 둠

    def _load_frames(self):
        if self.frames is None:
            self.frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in self.images]
            for path, frame in zip(self.images, self.frames):
                if frame is None:
                    raise SystemExit(f"이미지를 못 읽음: {path}")

    def run(self, tag: str, path: Path, save_dir: Path | None) -> tuple[dict, list]:
        args = self.args
        print(f"\n=== {tag}: {path} ===")
        self._load_frames()

        gc.collect()
        rss_before = rss_mb()
        t_load = time.perf_counter()
        kwargs = self.onnx_kwargs if path.suffix.lower() == ".onnx" else {}
        seg = create_segmenter(path, **kwargs)
        load_seconds = time.perf_counter() - t_load
        rss_after = rss_mb()
        if hasattr(seg, "providers"):
            print(f"  onnxruntime providers: {seg.providers}")

        for i in range(args.warmup):
            seg.segment(self.frames[0], i)

        total_ms, infer_ms, pre_ms, post_ms = [], [], [], []
        run_p50 = []        # 반복(run)마다의 p50 → 반복 간 편차 확인용
        frame_rows = []     # 프레임별 시간 (raw_frame_log 용)
        masks = []
        frame_period = 1.0 / args.target_fps if args.target_fps > 0 else 0.0   # 카메라 속도 흉내. 0 이면 최대 속도
        for run_index in range(args.runs):
            if run_index > 0 and args.cooldown > 0:
                print(f"  {args.cooldown:.0f}s 식히는 중...")
                time.sleep(args.cooldown)
            this_run = []
            masks = []
            next_due = time.perf_counter()
            for i, frame in enumerate(self.frames):
                if frame_period:   # 카메라가 다음 프레임을 주는 시각까지 기다림
                    now = time.perf_counter()
                    if now < next_due:
                        time.sleep(next_due - now)
                    next_due = max(next_due, now) + frame_period
                t_start = time.perf_counter()
                mask, _ = seg.segment(frame, i)
                elapsed = (time.perf_counter() - t_start) * 1e3
                masks.append(mask.copy())
                timing = seg.last_timing
                total_ms.append(elapsed)
                this_run.append(elapsed)
                infer_ms.append(timing["infer_ms"])
                pre_ms.append(timing["pre_ms"])
                post_ms.append(timing["post_ms"])
                frame_rows.append({"run": run_index + 1, "frame_id": i, "image": self.images[i].name,
                                   "pre_ms": round(timing["pre_ms"], 3), "infer_ms": round(timing["infer_ms"], 3),
                                   "post_ms": round(timing["post_ms"], 3), "total_ms": round(elapsed, 3),
                                   "deadline_miss": int(elapsed > args.budget_ms)})
            run_p50.append(percentile(this_run, 50))
            print(f"  run {run_index + 1}/{args.runs}: {len(self.frames)}장, total p50 {run_p50[-1]} ms")

        result = {
            "tag": tag, "path": str(path),
            "file_mb": round(path.stat().st_size / 1e6, 2),
            "load_seconds": round(load_seconds, 2),
            "rss_delta_mb": round(rss_after - rss_before, 1),
            "device_memory_mb": round(seg.device_memory_bytes / 1e6, 1) if isinstance(seg, TrtSegmenter) else None,
            "backend": type(seg).__name__,
            "frames": len(self.frames), "runs": args.runs, "warmup": args.warmup,
            "total_ms": {"p50": percentile(total_ms, 50), "p95": percentile(total_ms, 95),
                         "mean": round(float(np.mean(total_ms)), 3), "max": round(float(np.max(total_ms)), 3)},
            "infer_ms": {"p50": percentile(infer_ms, 50), "p95": percentile(infer_ms, 95)},
            "pre_ms_p50": percentile(pre_ms, 50),
            "post_ms_p50": percentile(post_ms, 50),
            "fps_from_p50": round(1000.0 / percentile(total_ms, 50), 1),
            "over_budget_pct": round(float(np.mean(np.array(total_ms) > args.budget_ms)) * 100, 2),
            "run_p50_list": run_p50,
            "target_fps": args.target_fps,   # 0 이면 최대 속도로 잰 것
        }
        if args.runs > 1:
            spread = (max(run_p50) - min(run_p50)) / max(np.median(run_p50), 1e-9) * 100
            result["run_variation_pct"] = round(spread, 2)   # 반복 간 p50 편차 %. 크면 발열·백그라운드 의심
        if hasattr(seg, "providers"):
            result["providers"] = seg.providers

        if any(gt is not None for gt in self.gts):
            cm = np.zeros((self.metrics.n, self.metrics.n), dtype=np.int64)
            used = 0
            for gt, mask in zip(self.gts, masks):
                if gt is None:
                    continue
                if gt.shape != mask.shape:
                    raise SystemExit(f"정답 마스크 크기 {gt.shape} ≠ 마스크 {mask.shape}")
                cm += self.metrics.confusion(gt, mask)
                used += 1
            result["accuracy"] = self.metrics.iou_from_confusion(cm)
            result["accuracy"]["gt_frames"] = used
            names = [p.stem for p in self.images]
            result["per_image"] = self.metrics.per_image(names, self.gts, masks)
            worst = ", ".join(f"{w['name']}({w['miou']:.3f})" for w in result["per_image"]["worst5"])
            print(f"  mIoU {result['accuracy']['miou']}  장애물→바닥 {result['accuracy'].get('obstacle_to_floor_pct')}%  최악 5장: {worst}")

        if save_dir is not None:
            out_dir = save_dir / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            for image_path, mask in zip(self.images, masks):
                cv2.imwrite(str(out_dir / (image_path.stem + ".png")), mask)   # 값 0/1/2 그대로
            result["masks_dir"] = str(out_dir)

        # 프레임별 시간 로그 (팀 문서의 raw_frame_log 중 AI 추론 부분). 역할 4 가 전체 루프 로그와 합칠 수 있음
        log_path = args.out / f"raw_frame_log_{tag}.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)
        result["frame_log"] = str(log_path)

        seg.close()
        del seg
        gc.collect()
        return result, masks


# ---------- 표 ----------

def make_table(results: list[dict], class_names: list[str], ref_tag: str, budget_ms: float) -> str:
    ko = [CLASS_KO.get(name, name) for name in class_names]
    header = (["모델", "mIoU"] + [f"IoU {c}" for c in ko] + ["장애물→바닥%"]
              + ["p50 ms(전체)", "p95 ms(전체)", "p50 ms(추론만)", "FPS", f"{budget_ms:g}ms 초과%", "반복 변동%",
                 "파일 MB", "활성화 MB", "RSS 증가 MB", f"{ref_tag} 대비 일치%"]
              + [f"일치% {c}" for c in ko])

    def cell(value, digits=1):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        acc = r.get("accuracy", {})
        agree = r.get("agreement", {})
        row = ([r["tag"], cell(acc.get("miou"), 3)]
               + [cell(v, 3) for v in acc.get("iou", [None] * len(ko))]
               + [cell(acc.get("obstacle_to_floor_pct"), 2)]
               + [cell(r["total_ms"]["p50"], 2), cell(r["total_ms"]["p95"], 2), cell(r["infer_ms"]["p50"], 2),
                  cell(r["fps_from_p50"]), cell(r["over_budget_pct"]), cell(r.get("run_variation_pct")),
                  cell(r["file_mb"]),
                  cell(r.get("device_memory_mb")), cell(r["rss_delta_mb"]), cell(agree.get("pixel_agree_pct"), 2)]
               + [cell(v, 2) for v in agree.get("class_agree_pct", [None] * len(ko))])
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------- main ----------

def guess_tag(path: Path) -> str:
    """파일명으로 행 이름 추정"""
    stem = path.stem.lower()
    if path.suffix.lower() == ".onnx":
        return "cpu_fp32"
    if "qat" in stem:
        return "qat_int8"
    if "int8" in stem:
        return "ptq_int8"
    if "fp32" in stem:
        return "gpu_fp32"
    if "fp16" in stem:
        return "fp16"
    return path.stem


def load_classes(classes_path: Path | None, models: list) -> tuple[list, list]:
    """classes.json → (클래스 이름, 팔레트). 안 주면 첫 모델 폴더에서 찾고, 없으면 기본값"""
    class_names, palette = list(DEFAULT_CLASS_NAMES), list(DEFAULT_PALETTE_RGB)
    if classes_path is None:
        classes_path = next((m.parent / "classes.json" for _, m in models if (m.parent / "classes.json").exists()), None)
    if classes_path and classes_path.exists():
        raw = json.loads(classes_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "classes" in raw:          # export_onnx 가 만든 형식
            class_names = list(raw["classes"])
            palette = [list(color) for color in raw.get("palette_rgb", palette)]
        elif isinstance(raw, dict):                             # {"0": "floor", ...}
            class_names = [raw[key] for key in sorted(raw, key=int)]
        else:                                                   # ["floor", ...]
            class_names = list(raw)
        print(f"[classes] {classes_path}: {class_names}, palette(RGB) {palette}")
    return class_names, palette


def parse_args():
    parser = argparse.ArgumentParser(description="모델별 속도·정확도·메모리 비교")
    parser.add_argument("--models", nargs="+", required=True, help="경로 또는 이름=경로")
    parser.add_argument("--images", required=True, type=Path, help="사진 폴더 (<이름>_color.png)")
    parser.add_argument("--gt", type=Path, help="정답 폴더 (<이름>_label_RGB.png 색 라벨, 또는 0/1/2 값 png)")
    parser.add_argument("--list", type=Path, help="사진 이름 목록 (splits/val.txt). 없으면 폴더 전체")
    parser.add_argument("--img-suffix", default="_color.png")
    parser.add_argument("--gt-suffix", default="_label_RGB.png")
    parser.add_argument("--out", type=Path, default=Path("results/bench"))
    parser.add_argument("--classes", type=Path, help="classes.json (기본: 첫 모델 폴더에서 찾음)")
    parser.add_argument("--ref", default=None, help="일치율 기준 모델 (기본: cpu_fp32 → 없으면 fp32 든 것)")
    parser.add_argument("--onnx-cuda", action="store_true", help=".onnx 를 onnxruntime CUDA 로 (기본 CPU)")
    parser.add_argument("--onnx-threads", type=int, default=0, help="onnxruntime CPU 스레드 수 (0 = 코어 수)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1, help="측정 반복 횟수 (팀 규칙 5)")
    parser.add_argument("--cooldown", type=float, default=0.0, help="반복 사이 쉬는 초")
    parser.add_argument("--budget-ms", type=float, default=33.3)
    parser.add_argument("--target-fps", type=float, default=0.0,
                        help="카메라 속도 흉내 (예: 30). 전력을 '같은 처리 속도' 에서 비교할 때. 0 = 최대 속도")
    parser.add_argument("--limit", type=int, default=0, help="사진 수 제한 (0 = 전부)")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--tag", default="", help="결과 json 에 남길 메모 (예: MAXN, clocks on)")
    return parser.parse_args()


def main():
    args = parse_args()

    models = []   # (행 이름, 경로)
    for item in args.models:
        tag, _, path_str = item.partition("=") if "=" in item else ("", "", item)
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"모델이 없음: {path}")
        models.append((tag or guess_tag(path), path))

    class_names, palette = load_classes(args.classes, models)
    metrics = SegMetrics(len(class_names))

    names = None
    if args.list:
        lines = args.list.read_text(encoding="utf-8").splitlines()
        names = [line.split()[0] for line in lines if line.strip() and not line.startswith("#")]
    data = GroundTruth(args.images, args.gt, args.img_suffix, args.gt_suffix, len(class_names), palette)
    images = data.list_images(names)
    if args.limit > 0:
        images = images[:args.limit]
    gts = [data.load(path) for path in images]
    n_gt = sum(gt is not None for gt in gts)
    print(f"[data] 사진 {len(images)}장, 정답 마스크 {n_gt}장")
    if args.gt and n_gt == 0:
        raise SystemExit(f"{args.gt} 에서 정답을 하나도 못 찾음 (--gt-suffix 확인: 지금 '{args.gt_suffix}')")

    onnx_kwargs = {"providers": None if args.onnx_cuda else ["CPUExecutionProvider"]}
    if args.onnx_threads > 0:
        onnx_kwargs["threads"] = args.onnx_threads

    args.out.mkdir(parents=True, exist_ok=True)
    env = collect_env()
    env["note"] = args.tag
    print("[env]", json.dumps(env, ensure_ascii=False))
    prime_runtimes(models)

    bench = ModelBenchmark(images, gts, metrics, args, onnx_kwargs)
    results, mask_sets = [], {}
    for tag, path in models:
        result, masks = bench.run(tag, path, args.out / "masks" if args.save_masks else None)
        results.append(result)
        mask_sets[tag] = masks

    # 기준 모델: --ref → 이름에 cpu 든 것 → fp32 든 것 → 첫 번째
    tags = [tag for tag, _ in models]
    ref = args.ref or next((t for t in tags if "cpu" in t.lower()), next((t for t in tags if "fp32" in t.lower()), tags[0]))
    for result in results:
        if result["tag"] != ref:
            result["agreement"] = metrics.agreement(mask_sets[ref], mask_sets[result["tag"]])
            result["agreement"]["ref"] = ref

    table = make_table(results, class_names, ref, args.budget_ms)
    print("\n" + table)
    (args.out / "bench_table.md").write_text(f"# 성능표 ({env['datetime']}, {env.get('nvpmodel', '')})\n\n{table}\n", encoding="utf-8")
    (args.out / "bench_results.json").write_text(
        json.dumps({"env": env, "args": {k: str(v) for k, v in vars(args).items()},
                    "class_names": class_names, "ref": ref, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[done] {args.out}/bench_results.json, bench_table.md")


if __name__ == "__main__":
    main()
