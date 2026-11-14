#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
version="${1:-v6_warp_tiling}"
executable="${project_dir}/bin/${version}"
report_dir="${project_dir}/profiles/nsys"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_base="${report_dir}/${version}_${timestamp}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "Error: nsys is not available in PATH." >&2
    exit 1
fi

if [[ ! -x "${executable}" ]]; then
    echo "Error: executable not found: ${executable}" >&2
    echo "Build it first with: make bin/${version}" >&2
    exit 1
fi

mkdir -p "${report_dir}"

echo "Profiling: ${executable}"
echo "Report:    ${report_base}.nsys-rep"

nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --backtrace=none \
    --stats=true \
    --force-overwrite=true \
    --output="${report_base}" \
    "${executable}"
