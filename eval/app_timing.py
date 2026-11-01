"""카메라 앱 프레임별 로그 요약 (측정 스펙 2·5번: 단계별 시간, p50/p95, fps, 마감 초과, 드롭, 입력 나이, 지연)

python -m eval.app_timing int8=results/app_live_1101/int8/frame_log.csv fp32=results/app_live_1101/fp32/frame_log.csv \\
    --budget-ms 66.7 --skip 10 --out results/app_live_1101

- 입력: 앱(camera/D435_Fusion_v3.5)이 라이브 실행마다 쓰는 frame_log CSV (+ 옆의 .json 조건 블록),
  또는 `v3.5.py --replay ... --report x.json` 의 재현 보고서 (오프라인 계산 시간만; 수신·지연 열은 비어 있음)
- --skip N: 앞 N 프레임(워밍업)은 버린다. --budget-ms: 마감 (기본 1000/15 = 66.7 ms, 카메라 15 fps 간격)
- 열: 처리 fps, 수신 fps, 드롭%, 처리 ms p50/p95/max, 마감 초과%, 단계별 p50 (SCNN 준비/추론/디코드, 바닥 기하, 코스트맵, 후보 검토, 마무리, 화면),
  입력 나이·끝까지 지연 p50/p95, 바닥 검출 성공%, 점 개수 p50
- 출력: 표를 화면에 찍고 --out 에 app_timing.md / app_timing.json 저장
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STAGE_COLUMNS = [("processing_ms", "처리 전체"), ("semantics_ms", "SCNN 단계"), ("scnn_prepare_ms", "  입력 준비"),
                 ("scnn_infer_ms", "  모델 실행"), ("scnn_decode_ms", "  softmax 디코드"), ("geometry_floor_ms", "깊이→3D·바닥"),
                 ("costmap_ms", "코스트맵"), ("candidate_review_ms", "  후보 검토"), ("finalize_ms", "마무리"), ("render_ms", "화면 그리기(UI)")]


def _f(value):
    try:
        return float(value) if value not in (None, "", "None") else None
    except ValueError:
        return None


def load_csv(path: Path) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    side = path.with_suffix(".json")
    conditions = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    return rows, conditions


def load_report(path: Path) -> tuple[list[dict], dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for cap in report.get("captures", []):
        if "error" in cap:
            continue
        stage = cap.get("stage_ms", {}); scnn = cap.get("scnn") or {}; timing = scnn.get("timing") or {}
        review = cap.get("map_diagnostics", {}).get("candidate_review", {})
        rows.append({"processing_ms": cap.get("processing_ms"), "semantics_ms": stage.get("semantics"),
                     "scnn_prepare_ms": timing.get("prepare_ms"), "scnn_infer_ms": timing.get("infer_ms"), "scnn_decode_ms": timing.get("decode_ms"),
                     "geometry_floor_ms": stage.get("geometry_floor"), "costmap_ms": stage.get("costmap"),
                     "candidate_review_ms": review.get("elapsed_ms"), "finalize_ms": stage.get("finalize"),
                     "floor_ok": cap.get("map_updated"), "scnn_status": scnn.get("status"), "point_count": cap.get("point_count")})
    conditions = {"mode": report.get("mode"), "replay": True}
    return rows, conditions


def pct(values, q):
    return round(float(np.percentile(values, q)), 2) if len(values) else None


def summarize(rows: list[dict], skip: int, budget_ms: float) -> dict:
    rows = rows[skip:]
    out = {"frames": len(rows), "skipped": skip}
    if not rows:
        return out
    col = lambda name: np.array([v for v in (_f(r.get(name)) for r in rows) if v is not None])
    elapsed = col("elapsed_s")
    if len(elapsed) > 1:
        out["span_s"] = round(float(elapsed[-1] - elapsed[0]), 1)
        out["processed_fps"] = round((len(elapsed) - 1) / max(elapsed[-1] - elapsed[0], 1e-6), 2)
    acquire = col("acquire_fps")
    if len(acquire):
        out["acquire_fps_mean"] = round(float(acquire.mean()), 2)
    received, dropped = col("received_frames"), col("dropped_pending")
    if len(received) > 1 and len(dropped) > 1:
        got = int(received[-1] - received[0]); lost = int(dropped[-1] - dropped[0])
        out["received_frames"] = got; out["dropped_frames"] = lost
        out["dropped_pct"] = round(lost / got * 100, 1) if got else None
    processing = col("processing_ms")
    out["processing"] = {"p50": pct(processing, 50), "p95": pct(processing, 95), "max": pct(processing, 100)}
    out["deadline_miss_pct"] = round(float((processing > budget_ms).mean()) * 100, 1) if len(processing) else None
    out["budget_ms"] = budget_ms
    out["stages_p50"] = {name: pct(col(name), 50) for name, _ in STAGE_COLUMNS}
    out["stages_p95"] = {name: pct(col(name), 95) for name, _ in STAGE_COLUMNS}
    for name in ("input_age_ms", "latency_ms", "capture_age_ms"):
        values = col(name)
        if len(values):
            out[name] = {"p50": pct(values, 50), "p95": pct(values, 95)}
    floor = [str(r.get("floor_ok")) for r in rows if r.get("floor_ok") not in (None, "")]
    if floor:
        out["floor_ok_pct"] = round(sum(v == "True" for v in floor) / len(floor) * 100, 1)
    status = [r.get("scnn_status") for r in rows if r.get("scnn_status")]
    if status:
        out["scnn_ok_pct"] = round(sum(v == "OK" for v in status) / len(status) * 100, 1)
    points = col("point_count")
    if len(points):
        out["point_count_p50"] = int(np.percentile(points, 50))
    return out


def cell(value, nd=1):
    if value is None:
        return "-"
    return f"{value:.{nd}f}" if isinstance(value, float) else str(value)


def table(runs: list[tuple[str, dict]]) -> str:
    names = [name for name, _ in runs]
    lines = ["| 항목 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def row(label, getter, nd=1):
        lines.append(f"| {label} | " + " | ".join(cell(getter(s), nd) for _, s in runs) + " |")

    row("처리 프레임 수 (워밍업 제외)", lambda s: s.get("frames"))
    row("구간 길이 s", lambda s: s.get("span_s"))
    row("처리 fps", lambda s: s.get("processed_fps"), 2)
    row("수신 fps", lambda s: s.get("acquire_fps_mean"), 2)
    row("수신 프레임 / 드롭 / 드롭%", lambda s: None if s.get("received_frames") is None else f"{s['received_frames']} / {s['dropped_frames']} / {cell(s.get('dropped_pct'))}")
    row("처리 ms p50 / p95 / max", lambda s: f"{cell(s['processing']['p50'])} / {cell(s['processing']['p95'])} / {cell(s['processing']['max'])}" if s.get("processing") else None)
    row("마감 초과% (처리 > budget)", lambda s: s.get("deadline_miss_pct"))
    for name, label in STAGE_COLUMNS:
        row(f"{label} p50 ms", lambda s, n=name: (s.get("stages_p50") or {}).get(n))
    row("입력 나이 ms p50 / p95", lambda s: f"{cell(s['input_age_ms']['p50'])} / {cell(s['input_age_ms']['p95'])}" if s.get("input_age_ms") else None)
    row("끝까지 지연 ms p50 / p95", lambda s: f"{cell(s['latency_ms']['p50'])} / {cell(s['latency_ms']['p95'])}" if s.get("latency_ms") else None)
    row("바닥 검출 성공%", lambda s: s.get("floor_ok_pct"))
    row("SCNN OK%", lambda s: s.get("scnn_ok_pct"))
    row("점 개수 p50", lambda s: s.get("point_count_p50"))
    return "\n".join(lines)


def conditions_table(runs: list[tuple[str, dict]]) -> str:
    lines = ["| 실행 | 모델 | sha256 앞 16 | 백엔드 | git | 전력 모드 | GPU 클럭 고정 | 시작 |", "|---|---|---|---|---|---|---|---|"]
    for name, c in runs:
        model = c.get("model") or {}
        lines.append(f"| {name} | {model.get('file', '-')} | {str(model.get('sha256', '-'))[:16]} | {str(model.get('message', '-')).split('|')[-1].strip()} | "
                     f"{c.get('git_commit', '-')} | {str(c.get('nvpmodel', '-')).replace('NV Power Mode: ', '')} | {c.get('gpu_clock_locked', '-')} | {str(c.get('started_at', '-'))[:19]} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="카메라 앱 프레임별 로그 요약")
    parser.add_argument("logs", nargs="+", help="이름=frame_log.csv 또는 이름=replay_report.json")
    parser.add_argument("--skip", type=int, default=10, help="앞 N 프레임(워밍업) 제외")
    parser.add_argument("--budget-ms", type=float, default=1000 / 15, help="마감 ms (기본 66.7 = 15 fps)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs, conditions = [], []
    for item in args.logs:
        name, _, path = item.partition("=") if "=" in item else (Path(item).parent.name, "", item)
        path = Path(path)
        rows, cond = load_report(path) if path.suffix == ".json" else load_csv(path)
        runs.append((name, summarize(rows, args.skip, args.budget_ms)))
        conditions.append((name, cond))

    text = ["## 조건", "", conditions_table(conditions), "", f"## 프레임별 요약 (앞 {args.skip} 프레임 제외, 마감 {args.budget_ms:.1f} ms)", "", table(runs)]
    print("\n".join(text))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "app_timing.md").write_text("\n".join(text) + "\n", encoding="utf-8")
        (args.out / "app_timing.json").write_text(json.dumps({"args": vars(args) | {"out": str(args.out)}, "runs": dict(runs), "conditions": dict(conditions)},
                                                             ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
