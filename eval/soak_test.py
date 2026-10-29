"""장시간 실행 안정성 테스트. 모델 하나를 카메라 속도로 N 분 계속 돌리며 속도가 떨어지는지·프레임을 놓치는지 봄

eval/tegrastats_log.sh start results/soak_fp16.log
python -m eval.soak_test --model models/fast_scnn_fp16.engine --images data/images --list data/splits/val.txt \\
    --minutes 10 --target-fps 30 --out results/soak_fp16
eval/tegrastats_log.sh stop
python -m eval.parse_tegrastats results/soak_fp16.log

- 사진을 순서대로 반복하며 30 fps 간격으로 넣음 (실제 카메라처럼)
- 10초마다 p50/p95/최대 지연과 "제때 못 끝낸 프레임" 수를 기록
- 끝나면 처음 1분 vs 마지막 1분을 비교 → 지연이 늘었으면 발열로 클럭이 내려간 것(스로틀링)
- 출력: <out>/soak.csv (10초 단위), <out>/soak_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.segmenter import create_segmenter  # noqa: E402

REPORT_EVERY_S = 10   # 몇 초마다 한 줄 기록할지


def load_frames(img_dir: Path, list_path: Path | None, img_suffix: str) -> list:
    if list_path:
        names = [ln.split()[0] for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        paths = [img_dir / f"{n}{img_suffix}" for n in names]
    else:
        paths = sorted(p for p in img_dir.iterdir() if p.name.endswith(img_suffix))
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    if not frames or any(f is None for f in frames):
        raise SystemExit(f"{img_dir} 에서 사진을 못 읽음")
    return frames


def summarize(window: list[float], budget_ms: float) -> dict:
    arr = np.array(window)
    return {"frames": len(arr), "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2), "max_ms": round(float(arr.max()), 2),
            "over_budget": int((arr > budget_ms).sum())}   # 한 프레임 간격 안에 못 끝낸 수


def main():
    parser = argparse.ArgumentParser(description="장시간 실행 안정성 테스트")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--list", type=Path, default=None)
    parser.add_argument("--img-suffix", default="_color.png")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--target-fps", type=float, default=30.0, help="카메라 속도. 0 이면 최대 속도")
    parser.add_argument("--out", type=Path, default=Path("results/soak"))
    args = parser.parse_args()

    frames = load_frames(args.images, args.list, args.img_suffix)
    budget_ms = 1000.0 / args.target_fps if args.target_fps > 0 else 33.3
    frame_period = 1.0 / args.target_fps if args.target_fps > 0 else 0.0
    args.out.mkdir(parents=True, exist_ok=True)

    kwargs = {"providers": ["CPUExecutionProvider"]} if args.model.suffix == ".onnx" else {}
    seg = create_segmenter(args.model, **kwargs)
    for i in range(20):   # 워밍업
        seg.segment(frames[0], i)

    print(f"[soak] {args.model.name}, {args.minutes} 분, 목표 {args.target_fps} fps (프레임 간격 {budget_ms:.1f} ms)")
    rows = []            # 10초 단위 요약
    window = []          # 현재 10초 동안의 지연
    all_latency = []
    started = time.perf_counter()
    next_report = started + REPORT_EVERY_S
    next_due = started
    frame_id = 0
    while time.perf_counter() - started < args.minutes * 60:
        if frame_period:
            now = time.perf_counter()
            if now < next_due:
                time.sleep(next_due - now)
            next_due = max(next_due, now) + frame_period
        t0 = time.perf_counter()
        seg.segment(frames[frame_id % len(frames)], frame_id)
        latency = (time.perf_counter() - t0) * 1e3
        window.append(latency)
        all_latency.append(latency)
        frame_id += 1

        if time.perf_counter() >= next_report:
            row = {"t_s": round(time.perf_counter() - started), **summarize(window, budget_ms)}
            rows.append(row)
            print(f"  {row['t_s']:>5}s  p50 {row['p50_ms']:.2f}  p95 {row['p95_ms']:.2f}  max {row['max_ms']:.2f} ms  "
                  f"놓친 프레임 {row['over_budget']}/{row['frames']}")
            window = []
            next_report += REPORT_EVERY_S
    seg.close()

    # 처음 1분 vs 마지막 1분
    per_min = max(1, 60 // REPORT_EVERY_S)
    first = [r["p50_ms"] for r in rows[:per_min]]
    last = [r["p50_ms"] for r in rows[-per_min:]]
    summary = {
        "model": str(args.model), "minutes": args.minutes, "target_fps": args.target_fps,
        "total_frames": frame_id,
        "p50_ms_all": round(float(np.percentile(all_latency, 50)), 2),
        "p95_ms_all": round(float(np.percentile(all_latency, 95)), 2),
        "max_ms_all": round(float(np.max(all_latency)), 2),
        "over_budget_total": int((np.array(all_latency) > budget_ms).sum()),
        "p50_first_minute": round(float(np.mean(first)), 2) if first else None,
        "p50_last_minute": round(float(np.mean(last)), 2) if last else None,
    }
    if first and last:
        drift = (summary["p50_last_minute"] - summary["p50_first_minute"]) / summary["p50_first_minute"] * 100
        summary["drift_pct"] = round(drift, 1)   # +면 느려진 것. 10% 넘으면 스로틀링 의심 → tegrastats 의 GPU 클럭 확인

    with open(args.out / "soak.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["t_s"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "soak_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[soak] 총 {frame_id} 프레임, p50 {summary['p50_ms_all']} ms, 놓친 프레임 {summary['over_budget_total']}, "
          f"처음 1분 {summary['p50_first_minute']} → 마지막 1분 {summary['p50_last_minute']} ms"
          + (f" (변화 {summary['drift_pct']:+.1f}%)" if "drift_pct" in summary else ""))
    print(f"[done] {args.out}/soak.csv, soak_summary.json")


if __name__ == "__main__":
    main()
