#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NCU_BIN="${NCU_BIN:-/media/drink_ice/Data/cuda-tools/nsight-compute-2026.2.1/ncu}"
REPORT_DIR="${NCU_REPORT_DIR:-/media/drink_ice/Data/gemm-opt-profiles/ncu}"
CONFIG="${1:-bm128_bn128_bk16_tm8_tn8}"
SIZE="${2:-2048}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_BASE="${REPORT_DIR}/autotune_${CONFIG}_${SIZE}_${STAMP}"
EXTRA_METRICS="${EXTRA_METRICS:-dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,gpu__time_duration.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct}"

if [[ ! -x "${NCU_BIN}" ]]; then
    echo "ncu not found or not executable: ${NCU_BIN}" >&2
    exit 1
fi

mkdir -p "${REPORT_DIR}"

"${NCU_BIN}" \
    --kernel-name-base function \
    --kernel-name regex:gemmTunable \
    --launch-skip 5 \
    --launch-count 1 \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section LaunchStats \
    --section Occupancy \
    --section MemoryWorkloadAnalysis \
    --section SchedulerStats \
    --section WarpStateStats \
    --metrics "${EXTRA_METRICS}" \
    --export "${REPORT_BASE}" \
    --force-overwrite \
    "${PROJECT_DIR}/bin/autotune_gemm" \
    --profile "${CONFIG}" \
    --size "${SIZE}"

echo "NCU report: ${REPORT_BASE}.ncu-rep"
