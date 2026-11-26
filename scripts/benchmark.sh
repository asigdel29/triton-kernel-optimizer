#!/usr/bin/env bash
# Record reproducible CPU baselines or paired CUDA backend measurements.
set -euo pipefail
device=${1:-cpu}
output=${2:-benchmark-results}
python=${PYTHON:-python3}
case "$device" in
  cpu) backends=(torch); dtype=float32 ;;
  cuda) backends=(torch triton); dtype=float16 ;;
  *) echo 'Use cpu or cuda.' >&2; exit 2 ;;
esac
mkdir "$output"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$python" -VV > "$output/python.txt"
"$python" -m pip freeze > "$output/packages.txt"
uname -sm > "$output/platform.txt"
git rev-parse HEAD > "$output/revision.txt" 2>/dev/null || echo uncommitted > "$output/revision.txt"
if [[ "$device" == cuda ]]; then
  nvidia-smi -q > "$output/nvidia-smi.txt"
fi
for tokens in 128 513; do
  for backend in "${backends[@]}"; do
    "$python" -m kernel_optimizer.bench --device "$device" --backend "$backend" \
      --dtype "$dtype" --tokens "$tokens" --depth 64 --repeats 20 >> "$output/trials.jsonl"
  done
done
if [[ "$device" == cuda ]]; then
  MPLBACKEND=Agg "$python" -m kernel_optimizer.roofline "$output/trials.jsonl" \
    "$output/roofline.svg" > "$output/ceilings.json"
fi
