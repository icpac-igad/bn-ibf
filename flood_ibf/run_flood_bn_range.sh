#!/usr/bin/env bash
# Run flood BN IBF v1 for a date range.
#   ./run_flood_bn_range.sh 2026-03-01 2026-03-10
set -euo pipefail

START_DATE="${1:-2026-03-01}"
END_DATE="${2:-2026-03-10}"
RP_YEARS="${RP_YEARS:-2}"
COST_LOSS_RATIO="${COST_LOSS_RATIO:-0.2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.juliaup/bin:$PATH"

INPUT_DIR="bn_inputs"
OUTPUT_DIR="output"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

UV_PKGS=(
  --with icechunk --with xarray --with "zarr>=3"
  --with numpy --with pandas --with geopandas --with regionmask
  --with netcdf4 --with pyarrow --with scipy
)

D="$START_DATE"
while [[ "$D" < "$(date -I -d "$END_DATE + 1 day")" ]]; do
  echo "================================================================"
  echo "  Flood BN IBF v1 — date: $D  (RP=${RP_YEARS}yr)"
  echo "================================================================"

  IN_CSV="${INPUT_DIR}/flood_inputs_${D}.csv"
  OUT_CSV="${OUTPUT_DIR}/flood_bn_v1_${D}.csv"
  MEMBER_CSV="${INPUT_DIR}/member_risk_${D}.csv"

  echo "[step 1/2] data prep..."
  uv run "${UV_PKGS[@]}" python flood_data_prep.py \
      --date "$D" \
      --rp-years "$RP_YEARS" \
      --out "$IN_CSV" \
      --member-sidecar "$MEMBER_CSV"

  echo "[step 2/2] Julia BN inference..."
  julia --project=. flood_bn_ibf_v1.jl \
      --input-csv "$IN_CSV" \
      --output-csv "$OUT_CSV" \
      --no-agreement \
      --tail-risk \
      --cost-loss-ratio "$COST_LOSS_RATIO"

  D="$(date -I -d "$D + 1 day")"
done

echo "================================================================"
echo "All dates processed. Outputs in $OUTPUT_DIR/"
echo "================================================================"
