"""카메라 앱 처리 파이프라인 프로파일: 어느 함수가 프레임 시간을 쓰는가. Jetson 에서 실행

python -m eval.app_profile --source camera --frames 20 --every 3 --save results/app_profile_1101/captures \\
    --runs 3 --dump results/app_profile_1101/before --out results/app_profile_1101/before
python -m eval.app_profile --source results/app_profile_1101/captures --runs 5 \\
    --dump results/app_profile_1101/after --compare results/app_profile_1101/before --out results/app_profile_1101/after

- --source camera : RealSense 에서 프레임을 받아 메모리에 두고 처리. --save 를 주면 앱의 캡처 형식(…_metadata.json 등)으로 저장해
                    `python v3.5.py --replay DIR --headless --sequence` 와 이 도구의 --source DIR 로 다시 쓸 수 있다
- --source DIR    : 저장된 캡처 폴더들을 d435map.io.load_capture 로 읽어 처리 (재현, 최적화 전후 비교, 반복 변동)
- 처리는 앱의 Processor.process 그대로 (SCNN 추론 포함, cv2 스레드 2 = 앱과 같음). 프레임을 순서대로 처리하고
  회차 시작마다 지도·후보 기록·중복 프레임 검사만 초기화한다 (앱 재현 모드와 같은 규칙)
- 출력 (<out>/): frames.csv (프레임별 단계 ms, eval.app_timing 로 요약 가능), summary.json (회차별 p50 과 반복 변동),
  profile.txt / profile.prof (마지막 회차 cProfile: tottime 상위, d435map 함수 cumtime 상위)
- --dump DIR 은 프레임별 결과(점유 격자 physical·비용 cost·자유 근거 free_evidence·점 분류·바닥 평면)를 npz 로 저장, --compare DIR 은 그 저장분과 셀 단위로 비교해
  최적화가 출력을 바꾸지 않았는지 보인다
- 시간 측정 규칙: jetson_clocks 고정, 다른 프로세스 없이. 첫 회차 전에 2 프레임 워밍업
"""
from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import pstats
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "camera" / "D435_Fusion_v3.5"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(APP))
from eval.bench_engines import collect_env  # noqa: E402

FRAME_COLUMNS = ["run", "index", "source_id", "elapsed_s", "processing_ms", "semantics_ms", "scnn_prepare_ms", "scnn_infer_ms", "scnn_decode_ms",
                 "geometry_floor_ms", "costmap_ms", "candidate_review_ms", "finalize_ms", "scnn_status", "floor_ok", "map_updated",
                 "point_count", "valid_depth_fraction", "free_cells", "obstacle_cells", "unknown_cells"]


def grab_from_camera(cfg, frames: int, every: int, skip: int):
    from d435map import sensor
    args = SimpleNamespace(fps=None, serial="")
    pipeline, profile, fps, failures = sensor.start_camera(args, cfg["camera"]["target_fps"])
    try:
        device = profile.get_device()
        depth = profile.get_stream(sensor.rs.stream.depth)
        color = profile.get_stream(sensor.rs.stream.color)
        calibration = sensor.calibration_info(device, depth, color, device.first_depth_sensor().get_depth_scale())
        aligner = sensor.rs.align(sensor.rs.stream.color)
        skew = 750 / fps
        captures, seen = [], 0
        while len(captures) < frames:
            ok, frameset = pipeline.try_wait_for_frames(2000)
            if not ok:
                raise RuntimeError("카메라 프레임을 2초 동안 받지 못함")
            cap = sensor.capture_from_frames(frameset, aligner, calibration, skew)
            del frameset
            if cap is None or not cap.pairing["within_threshold"]:
                continue
            seen += 1
            if seen <= skip or (seen - skip - 1) % every:
                continue
            captures.append(cap)
    finally:
        pipeline.stop()
    session = {"schema": "realsense-capture-v2.2-session", "producer": "eval.app_profile", "session_id": "profile_" + time.strftime("%Y%m%d_%H%M%S"),
               "started_at": sensor.now_iso(), "calibration": calibration, "selected_fps": fps, "max_skew_ms": skew,
               "startup_failures": failures, "python_version": sys.version, "note": "frames grabbed for pipeline profiling"}
    return captures, calibration, session, fps


def save_captures(root: Path, captures, calibration, session) -> list[Path]:
    from d435map.capture_storage import save_capture
    root.mkdir(parents=True, exist_ok=True)
    return [save_capture(root, cap, calibration, session) for cap in captures]


def load_packets(root: Path):
    from d435map.io import load_capture
    folders = [root] if list(root.glob("*_metadata.json")) else sorted({p.parent for p in root.rglob("*_metadata.json")})
    if not folders:
        raise SystemExit(f"캡처 폴더가 없음: {root}")
    packets = [load_capture(folder) for folder in folders]
    packets.sort(key=lambda p: p.capture.received_unix_ns)
    return packets


def frame_row(run: int, index: int, snapshot, elapsed: float) -> dict:
    s = snapshot.stats
    stage = s.get("stage_ms", {})
    scnn = s.get("scnn") or {}
    timing = scnn.get("timing") or {}
    review = s.get("map_diagnostics", {}).get("candidate_review", {})
    r = lambda v: None if v is None else round(float(v), 2)
    return {"run": run, "index": index, "source_id": s.get("source_id"), "elapsed_s": round(elapsed, 3),
            "processing_ms": r(s["processing_ms"]), "semantics_ms": r(stage.get("semantics")),
            "scnn_prepare_ms": r(timing.get("prepare_ms")), "scnn_infer_ms": r(timing.get("infer_ms")), "scnn_decode_ms": r(timing.get("decode_ms")),
            "geometry_floor_ms": r(stage.get("geometry_floor")), "costmap_ms": r(stage.get("costmap")),
            "candidate_review_ms": r(review.get("elapsed_ms")), "finalize_ms": r(stage.get("finalize")),
            "scnn_status": scnn.get("status"), "floor_ok": snapshot.floor.ok, "map_updated": s.get("map_updated"),
            "point_count": s.get("point_count"), "valid_depth_fraction": r(s.get("valid_depth_fraction")),
            "free_cells": int((snapshot.physical == 0).sum()), "obstacle_cells": int((snapshot.physical == 100).sum()),
            "unknown_cells": int((snapshot.physical == -1).sum())}


def reset(processor):
    processor.grid.reset()
    processor.last_signature = None
    processor.reviewer.reset()


def run_once(processor, packets, run: int, started: float, profiler=None, dump: Path | None = None) -> list[dict]:
    reset(processor)
    rows = []
    for index, packet in enumerate(packets):
        packet.capture.received_monotonic = time.monotonic()   # 재현 패킷(replay=True)이라 나이 검사는 건너뛴다. 후보 기록 시각용
        if profiler:
            profiler.enable()
        snapshot = processor.process(packet)
        if profiler:
            profiler.disable()
        rows.append(frame_row(run, index, snapshot, time.monotonic() - started))
        if dump is not None:
            np.savez_compressed(dump / f"frame_{index:03d}.npz", physical=snapshot.physical, cost=snapshot.cost,
                                evidence=snapshot.free_evidence if snapshot.free_evidence is not None else np.zeros(0, np.uint8),
                                classes=snapshot.point_classes, floor_normal=np.asarray(snapshot.floor.normal), floor_offset=np.float64(snapshot.floor.offset_m),
                                floor_ok=np.bool_(snapshot.floor.ok))
    return rows


def compare_dumps(after: Path, before: Path) -> dict:
    report = {"frames": 0, "identical_frames": 0, "diff": []}
    for path in sorted(after.glob("frame_*.npz")):
        other = before / path.name
        if not other.exists():
            continue
        a, b = np.load(path), np.load(other)
        diff = {"frame": path.stem}
        same = True
        for key in ("physical", "cost", "evidence", "classes"):
            if a[key].shape != b[key].shape:
                diff[key] = f"shape {a[key].shape} vs {b[key].shape}"; same = False
            else:
                n = int((a[key] != b[key]).sum())
                diff[key] = n
                same &= n == 0
        diff["floor_ok"] = bool(a["floor_ok"]) == bool(b["floor_ok"])
        diff["floor_normal_max_abs"] = float(np.abs(a["floor_normal"] - b["floor_normal"]).max())
        diff["floor_offset_abs"] = float(abs(a["floor_offset"] - b["floor_offset"]))
        same &= diff["floor_ok"] and diff["floor_normal_max_abs"] < 1e-9 and diff["floor_offset_abs"] < 1e-9
        report["frames"] += 1
        report["identical_frames"] += int(same)
        report["diff"].append(diff)
    return report


def percentile(rows, key, q):
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(float(np.percentile(values, q)), 2) if values else None


def profile_text(profiler: cProfile.Profile, top: int) -> str:
    out = io.StringIO()
    stats = pstats.Stats(profiler, stream=out)
    stats.sort_stats("tottime").print_stats(top)
    out.write("\n=== d435map 함수만, cumtime 순 ===\n")
    stats.sort_stats("cumtime").print_stats("d435map", top)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description="카메라 앱 처리 파이프라인 프로파일")
    parser.add_argument("--source", required=True, help="camera 또는 저장 캡처 폴더")
    parser.add_argument("--frames", type=int, default=20, help="camera: 받을 프레임 수")
    parser.add_argument("--every", type=int, default=3, help="camera: N 프레임마다 하나 사용")
    parser.add_argument("--skip", type=int, default=15, help="camera: 앞 프레임 버림 (노출 안정)")
    parser.add_argument("--save", type=Path, default=None, help="camera: 캡처를 앱 형식으로 저장할 폴더")
    parser.add_argument("--models-dir", type=Path, default=None, help="앱 config 대신 쓸 모델 폴더")
    parser.add_argument("--config", type=Path, default=None, help="앱 config.json 덮어쓰기 파일")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--dump", type=Path, default=None, help="프레임별 결과 npz 저장 폴더")
    parser.add_argument("--compare", type=Path, default=None, help="이전 --dump 폴더와 비교")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from d435map.configuration import load_config, validate
    from d435map.engine import Packet, Processor
    cfg = load_config(args.config)
    if args.models_dir:
        cfg["semantics"]["model_dir"] = str(args.models_dir.resolve())
        cfg["semantics"]["enabled"] = True
    validate(cfg)
    cv2.setNumThreads(2)   # 앱(application.main)과 같은 조건

    args.out.mkdir(parents=True, exist_ok=True)
    if args.source == "camera":
        captures, calibration, session, fps = grab_from_camera(cfg, args.frames, args.every, args.skip)
        print(f"[camera] {len(captures)} frames at {fps} fps (every {args.every}, skipped {args.skip})")
        if args.save:
            folders = save_captures(args.save, captures, calibration, session)
            print(f"[save] {len(folders)} capture folders → {args.save}")
        packets = [Packet(cap, calibration, f"cam{i:03d}", True) for i, cap in enumerate(captures)]
    else:
        packets = load_packets(Path(args.source))
        print(f"[load] {len(packets)} captures from {args.source}")

    processor = Processor(cfg)
    print("SCNN:", processor.semantic.status, "|", processor.semantic.message)
    started = time.monotonic()
    run_once(processor, packets[:2], -1, started)   # 워밍업 (lru 캐시, 첫 커널)

    rows, per_run = [], []
    profiler = None
    for run in range(1, args.runs + 1):
        last = run == args.runs
        profiler = cProfile.Profile() if last else None
        dump = None
        if last and args.dump:
            args.dump.mkdir(parents=True, exist_ok=True)
            dump = args.dump
        run_rows = run_once(processor, packets, run, started, profiler, dump)
        rows += run_rows
        summary = {"run": run, "frames": len(run_rows),
                   "p50": {k: percentile(run_rows, k, 50) for k in ("processing_ms", "semantics_ms", "scnn_prepare_ms", "scnn_infer_ms", "scnn_decode_ms",
                                                                    "geometry_floor_ms", "costmap_ms", "candidate_review_ms", "finalize_ms")},
                   "p95_processing_ms": percentile(run_rows, "processing_ms", 95), "profiled": last}
        per_run.append(summary)
        print(f"[run {run}] processing p50 {summary['p50']['processing_ms']} ms (SCNN {summary['p50']['semantics_ms']}, "
              f"geometry {summary['p50']['geometry_floor_ms']}, costmap {summary['p50']['costmap_ms']}, review {summary['p50']['candidate_review_ms']})"
              + (" [cProfile on: 시간이 조금 늘어남]" if last else ""))

    with open(args.out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_COLUMNS)
        writer.writeheader(); writer.writerows(rows)
    clean = [r for r in per_run if not r["profiled"]] or per_run
    p50s = [r["p50"]["processing_ms"] for r in clean if r["p50"]["processing_ms"] is not None]
    result = {"env": collect_env(), "args": {k: str(v) for k, v in vars(args).items()}, "frames": len(packets),
              "model": {"file": processor.semantic.model_filename, "sha256": processor.semantic.model_sha256, "message": processor.semantic.message},
              "runs": per_run,
              "processing_p50_runs_without_profiler": p50s,
              "run_variation_pct": round((max(p50s) - min(p50s)) / (sum(p50s) / len(p50s)) * 100, 2) if len(p50s) > 1 else None}
    if profiler:
        text = profile_text(profiler, args.top)
        (args.out / "profile.txt").write_text(text, encoding="utf-8")
        profiler.dump_stats(str(args.out / "profile.prof"))
        print("\n" + "\n".join(text.splitlines()[:args.top + 8]))
    if args.compare and args.dump:
        result["compare"] = compare_dumps(args.dump, args.compare)
        c = result["compare"]
        print(f"\n[compare] {c['identical_frames']}/{c['frames']} frames identical to {args.compare}")
        for d in c["diff"]:
            if not all(d.get(k) == 0 for k in ("physical", "cost", "evidence", "classes")):
                print("  differs:", d)
    (args.out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
