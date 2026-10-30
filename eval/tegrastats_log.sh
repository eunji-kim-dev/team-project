#!/usr/bin/env bash
# Jetson 전력·온도·RAM 을 1초마다 기록 (tegrastats 를 백그라운드로 켜고 끔)
#
#   eval/tegrastats_log.sh start results/idle.log    # 아무것도 안 돌리고 30초 → stop (기준선)
#   eval/tegrastats_log.sh stop
#   eval/tegrastats_log.sh start results/bench.log   # 켜 두고 bench 돌림
#   eval/tegrastats_log.sh stop
#   python -m eval.parse_tegrastats results/bench.log --baseline results/idle.log --fps 48.3
#
# 로그 옆에 <로그>.meta 로 그때의 전력 모드(nvpmodel)·시각을 남김
set -euo pipefail

CMD="${1:-}"
LOG="${2:-results/tegrastats_$(date +%m%d_%H%M%S).log}"
LOG="$(realpath -m "$LOG")"           # 절대경로로. tegrastats 는 백그라운드 데몬이라 상대경로가 엉뚱한 데 써질 수 있음
INTERVAL_MS="${INTERVAL_MS:-1000}"   # 기록 간격

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"   # 전력 레일(VDD_*)은 root 여야 보이는 보드가 있음
fi

case "$CMD" in
  start)
    mkdir -p "$(dirname "$LOG")"
    {
      echo "started_at: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "interval_ms: $INTERVAL_MS"
      echo "nvpmodel: $($SUDO nvpmodel -q 2>/dev/null | tr '\n' ' ' || true)"
      echo "jetson_clocks_maxfreq_lines: $($SUDO jetson_clocks --show 2>/dev/null | grep -c MaxFreq || true)"
      echo "l4t: $(head -n1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
    } > "$LOG.meta"
    $SUDO tegrastats --stop >/dev/null 2>&1 || true   # 이미 돌고 있으면 끄고 시작
    $SUDO tegrastats --start --interval "$INTERVAL_MS" --logfile "$LOG"
    echo "기록 시작: $LOG (${INTERVAL_MS} ms 간격). 끝나면: $0 stop"
    ;;
  stop)
    $SUDO tegrastats --stop
    echo "기록 종료"
    ;;
  *)
    echo "사용법: $0 start [로그파일] | stop"
    exit 1
    ;;
esac
