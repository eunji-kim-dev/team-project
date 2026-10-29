"""모델별 전력·온도 측정. 모델 하나씩 1분 돌리면서 tegrastats 를 따로 기록 → 전력표

python -m eval.power_sweep \\
    --models cpu_fp32=models/fast_scnn.onnx gpu_fp32=models/fast_scnn_fp32.engine fp16=models/fast_scnn_fp16.engine \\
             ptq_int8=models/fast_scnn_int8.engine qat_int8=models/fast_scnn_qat_int8.engine \\
    --images data/images --list data/splits/val.txt --target-fps 30 --seconds 60 --out results/power_30fps

- 왜 bench 와 따로 재나: bench 는 모델 하나가 26장을 0.2초에 끝내는데 tegrastats 는 1초에 한 번 찍음 → 모델별로 안 나뉨
- 순서: 대기 30초(기준선) → 모델마다 [tegrastats 켬 → soak 1분 → 끔] → 로그 파싱 → 표
- --target-fps 30: 모든 모델을 같은 속도(카메라 속도)로 돌림 → "같은 처리 속도에서 전력 비교" (선배 조언)
  --target-fps 0 이면 최대 속도. 이땐 빠른 모델이 더 많이 처리하니 W 가 아니라 mJ/프레임으로 비교할 것
- 출력: <out>/power_table.md, power_results.json, 모델별 tegrastats 로그·soak 결과
- sudo 비밀번호를 처음에 한 번 물어봄 (tegrastats 가 root 필요)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.soak_test import load_frames, run_soak  # noqa: E402
from eval.parse_tegrastats import TegrastatsLog  # noqa: E402

TEGRASTATS_SH = Path(__file__).with_name("tegrastats_log.sh")
IDLE_SECONDS = 30   # 기준선(아무것도 안 돌릴 때) 기록 시간


def tegrastats(cmd: str, log: Path | None = None) -> None:
    args = ["bash", str(TEGRASTATS_SH), cmd] + ([str(log)] if log else [])
    subprocess.run(args, check=True)


def power_summary(log: Path) -> dict:
    """tegrastats 로그 → 이 표에 쓸 값만"""
    s = TegrastatsLog(log).summary()
    rail = s["main_rail"]
    acc_rail = next((r for r in ("VDD_CPU_GPU_CV", "VDD_GPU_SOC") if r in s["power_mw"]), None)
    return {
        "samples": s["samples"],
        "board_w": round(s["power_mw"][rail]["mean"] / 1000, 2) if rail else None,          # 보드 전체 평균 W
        "board_w_max": round(s["power_mw"][rail]["max"] / 1000, 2) if rail else None,
        "cpu_gpu_w": round(s["power_mw"][acc_rail]["mean"] / 1000, 2) if acc_rail else None,   # CPU+GPU 쪽만
        "gpu_util_pct": s.get("gpu_util_pct", {}).get("mean"),
        "gpu_freq_min_mhz": s.get("gpu_freq_mhz", {}).get("min"),
        "temp_max_c": s["temps_max_c"].get(s["main_temp"]) if s["main_temp"] else None,
        "ram_max_mb": s["ram_used_mb"]["max"],
    }


def make_table(rows: list[dict], idle: dict | None) -> str:
    header = ["모델", "실제 fps", "보드 전력 W", "CPU+GPU W", "대기 대비 +W", "프레임당 mJ", "GPU 사용률 %", "최고 온도 °C", "p50 ms", "놓친 프레임"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def cell(v, d=2):
        return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))

    if idle:
        lines.append(f"| (대기) | - | {cell(idle['board_w'])} | {cell(idle['cpu_gpu_w'])} | - | - | {cell(idle['gpu_util_pct'], 1)} | {cell(idle['temp_max_c'], 1)} | - | - |")
    for r in rows:
        p, s = r["power"], r["soak"]
        lines.append("| " + " | ".join([
            r["tag"], cell(s["achieved_fps"], 1), cell(p["board_w"]), cell(p["cpu_gpu_w"]),
            cell(r.get("delta_w")), cell(r.get("mj_per_frame"), 1), cell(p["gpu_util_pct"], 1),
            cell(p["temp_max_c"], 1), cell(s["p50_ms_all"]), str(s["over_budget_total"]),
        ]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="모델별 전력·온도 측정")
    parser.add_argument("--models", nargs="+", required=True, help="이름=경로")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--list", type=Path, default=None)
    parser.add_argument("--img-suffix", default="_color.png")
    parser.add_argument("--target-fps", type=float, default=30.0, help="0 = 최대 속도")
    parser.add_argument("--seconds", type=float, default=60.0, help="모델당 측정 시간")
    parser.add_argument("--cooldown", type=float, default=20.0, help="모델 사이 쉬는 초 (앞 모델의 열이 넘어오지 않게)")
    parser.add_argument("--out", type=Path, default=Path("results/power"))
    parser.add_argument("--skip-idle", action="store_true")
    args = parser.parse_args()

    models = []
    for item in args.models:
        tag, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"이름=경로 형식으로 줄 것: {item}")
        models.append((tag, Path(path)))
    frames = load_frames(args.images, args.list, args.img_suffix)
    args.out.mkdir(parents=True, exist_ok=True)

    idle = None
    if not args.skip_idle:
        log = args.out / "idle.log"
        print(f"[idle] {IDLE_SECONDS}초 대기 기록 (아무것도 안 돌림)")
        tegrastats("start", log)
        time.sleep(IDLE_SECONDS)
        tegrastats("stop")
        idle = power_summary(log)
        print(f"[idle] 보드 {idle['board_w']} W, CPU+GPU {idle['cpu_gpu_w']} W")

    rows = []
    for i, (tag, path) in enumerate(models):
        if i > 0 and args.cooldown > 0:
            print(f"  {args.cooldown:.0f}s 식히는 중...")
            time.sleep(args.cooldown)
        print(f"\n=== {tag}: {path} ({args.seconds:.0f}초, 목표 {args.target_fps} fps) ===")
        log = args.out / f"{tag}.log"
        tegrastats("start", log)
        try:
            soak = run_soak(path, frames, args.seconds / 60.0, args.target_fps, args.out / tag, quiet=True)
        finally:
            tegrastats("stop")
        power = power_summary(log)
        row = {"tag": tag, "path": str(path), "soak": soak, "power": power}
        if power["board_w"] is not None:
            row["mj_per_frame"] = round(power["board_w"] * 1000 / soak["achieved_fps"], 1)   # W → mW, ÷ fps
            if idle and idle["board_w"] is not None:
                row["delta_w"] = round(power["board_w"] - idle["board_w"], 2)
        rows.append(row)
        print(f"  실제 {soak['achieved_fps']} fps, 보드 {power['board_w']} W, CPU+GPU {power['cpu_gpu_w']} W, "
              f"프레임당 {row.get('mj_per_frame')} mJ, 최고 {power['temp_max_c']} °C")

    table = make_table(rows, idle)
    print("\n" + table)
    (args.out / "power_table.md").write_text(
        f"# 전력표 (목표 {args.target_fps} fps, 모델당 {args.seconds:.0f}초, {time.strftime('%Y-%m-%d %H:%M')})\n\n{table}\n", encoding="utf-8")
    (args.out / "power_results.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "idle": idle, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[done] {args.out}/power_table.md, power_results.json")


if __name__ == "__main__":
    main()
