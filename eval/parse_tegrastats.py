"""tegrastats 로그 → 전력·RAM·온도 요약 (성능표의 "보드를 얼마나 쓰나" 항목)

python -m eval.parse_tegrastats results/bench.log --baseline results/idle.log --fps 48.3 --json results/power.json

  --baseline  아무것도 안 돌릴 때 기록한 로그. 있으면 "우리 프로그램 때문에 늘어난" RAM·전력도 냄
  --fps       bench_engines 가 낸 FPS. 있으면 프레임당 에너지(mJ) = 평균 전력(mW) ÷ FPS

tegrastats 한 줄 예 (Orin, JetPack 6)
  RAM 2345/7620MB ... GR3D_FREQ 45%@[612] ... gpu@44C tj@45.5C VDD_IN 4567mW/4512mW VDD_CPU_GPU_CV 1234mW/1200mW ...
  VDD_IN = 보드 전체 전력 (순간/평균). 발표엔 이걸 씀 / tj = 가장 뜨거운 곳 온도. 오르면 클럭 내려감(스로틀링)
  GR3D_FREQ = GPU 사용률%@클럭MHz. 전력 레일(VDD_*)이 없으면 sudo 없이 돌렸거나 센서 없는 보드
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
RE_GPU = re.compile(r"GR3D_FREQ (\d+)%(?:@\[?(\d+)\]?)?")
RE_TEMP = re.compile(r"(\w+)@([\d.]+)C")
RE_POWER = re.compile(r"(VDD_\w+|POM_\w+) (\d+)mW/(\d+)mW")
MAIN_RAILS = ("VDD_IN", "POM_5V_IN")        # 보드 전체 전력로 쓰는 레일 (앞에 있는 것 우선)
MAIN_TEMPS = ("tj", "gpu", "cpu")           # 대표 온도로 쓰는 센서 (앞에 있는 것 우선)


class TegrastatsLog:
    """로그 한 파일을 읽어서 샘플 리스트로 들고 있다가 summary() 로 요약"""

    def __init__(self, path: Path):
        self.path = path
        self.ram_used: list[int] = []
        self.ram_total: int | None = None
        self.gpu_util: list[int] = []
        self.gpu_freq: list[int] = []
        self.temps: dict[str, list[float]] = {}
        self.power: dict[str, list[int]] = {}
        self._parse()

    def _parse(self):
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                ram = RE_RAM.search(line)
                if not ram:
                    continue   # tegrastats 줄이 아님
                self.ram_used.append(int(ram.group(1)))
                self.ram_total = int(ram.group(2))
                gpu = RE_GPU.search(line)
                if gpu:
                    self.gpu_util.append(int(gpu.group(1)))
                    if gpu.group(2):
                        self.gpu_freq.append(int(gpu.group(2)))
                for name, value in RE_TEMP.findall(line):
                    self.temps.setdefault(name.lower(), []).append(float(value))
                for rail, instant, _average in RE_POWER.findall(line):
                    self.power.setdefault(rail, []).append(int(instant))
        if not self.ram_used:
            raise SystemExit(f"{self.path}: tegrastats 줄을 하나도 못 읽음")

    def summary(self) -> dict:
        s = {
            "samples": len(self.ram_used),
            "ram_total_mb": self.ram_total,
            "ram_used_mb": {"max": int(np.max(self.ram_used)), "mean": round(float(np.mean(self.ram_used)), 1),
                            "min": int(np.min(self.ram_used))},
        }
        if self.gpu_util:
            s["gpu_util_pct"] = {"mean": round(float(np.mean(self.gpu_util)), 1), "max": int(np.max(self.gpu_util))}
        if self.gpu_freq:
            freq = np.array(self.gpu_freq)
            s["gpu_freq_mhz"] = {"max": int(freq.max()), "min": int(freq.min()),
                                 # 최고 클럭의 90% 아래 샘플 비율. 바쁜데 이게 크면 스로틀링 의심
                                 "below_90pct_of_max_pct": round(float((freq < 0.9 * freq.max()).mean()) * 100, 1)}
        s["temps_max_c"] = {name: max(values) for name, values in self.temps.items()}
        s["main_temp"] = next((name for name in MAIN_TEMPS if name in self.temps), None)
        s["power_mw"] = {rail: {"mean": round(float(np.mean(v)), 0), "max": int(np.max(v))} for rail, v in self.power.items()}
        s["main_rail"] = next((rail for rail in MAIN_RAILS if rail in self.power), None)
        return s


def print_summary(s: dict, fps: float | None) -> dict:
    """요약을 사람이 읽게 출력. 프레임당 에너지 등 계산값은 dict 로 돌려줌"""
    extra = {}
    print(f"샘플 {s['samples']}개 (1초 간격이면 약 {s['samples']}초)")
    print(f"RAM 사용 최대 {s['ram_used_mb']['max']} MB / 평균 {s['ram_used_mb']['mean']} MB (전체 {s['ram_total_mb']} MB)")
    if "gpu_util_pct" in s:
        print(f"GPU 사용률 평균 {s['gpu_util_pct']['mean']}% 최대 {s['gpu_util_pct']['max']}%")
    if "gpu_freq_mhz" in s:
        g = s["gpu_freq_mhz"]
        print(f"GPU 클럭 최대 {g['max']} MHz / 최소 {g['min']} MHz, 최고의 90% 미만 샘플 {g['below_90pct_of_max_pct']}%")
    if s["main_temp"]:
        print(f"온도 최고 {s['main_temp']} {s['temps_max_c'][s['main_temp']]} °C")

    rail = s["main_rail"]
    if rail:
        p = s["power_mw"][rail]
        print(f"전력 (보드 전체, {rail}) 평균 {p['mean'] / 1000:.2f} W, 최대 {p['max'] / 1000:.2f} W")
        if fps:
            energy_mj = p["mean"] / fps   # mW ÷ (frame/s) = mJ/frame
            extra["energy_mj_per_frame"] = round(energy_mj, 1)
            print(f"프레임당 에너지 {energy_mj:.1f} mJ  (= {p['mean'] / 1000:.2f} W ÷ {fps} fps)")
        # 보드 전체 말고 CPU+GPU+가속기만 따로 (전력을 어디까지 잰 건지 구분해 적으라는 조언 반영)
        for name in ("VDD_CPU_GPU_CV", "VDD_GPU_SOC"):
            if name in s["power_mw"]:
                q = s["power_mw"][name]
                print(f"전력 (CPU+GPU 쪽만, {name}) 평균 {q['mean'] / 1000:.2f} W, 최대 {q['max'] / 1000:.2f} W")
                extra["accelerator_rail"] = name
    else:
        print("전력 레일(VDD_*)이 로그에 없음 → sudo 로 기록했는지, 보드에 전력 센서가 있는지 확인")
    return extra


def main():
    parser = argparse.ArgumentParser(description="tegrastats 로그 요약")
    parser.add_argument("log", type=Path)
    parser.add_argument("--baseline", type=Path, help="대기 상태 로그")
    parser.add_argument("--fps", type=float, help="프레임당 에너지 계산용 FPS")
    parser.add_argument("--json", type=Path, help="요약 저장 경로")
    args = parser.parse_args()

    summary = TegrastatsLog(args.log).summary()
    out = {"log": str(args.log), "summary": summary}
    out.update(print_summary(summary, args.fps))

    if args.baseline:
        base = TegrastatsLog(args.baseline).summary()
        delta = {"ram_mb": round(summary["ram_used_mb"]["max"] - base["ram_used_mb"]["mean"], 1)}
        print(f"\n기준선 대비: RAM 증가 {delta['ram_mb']} MB (측정 최대 − 대기 평균)")
        rail = summary["main_rail"]
        if rail and rail in base["power_mw"]:
            delta["power_mw"] = round(summary["power_mw"][rail]["mean"] - base["power_mw"][rail]["mean"], 0)
            print(f"            전력 증가 {delta['power_mw'] / 1000:.2f} W (평균 − 대기 평균)")
        out["baseline"] = {"log": str(args.baseline), "summary": base}
        out["delta_vs_baseline"] = delta

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장: {args.json}")


if __name__ == "__main__":
    main()
