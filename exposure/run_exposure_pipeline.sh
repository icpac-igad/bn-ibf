#!/usr/bin/env bash
# run_exposure_pipeline.sh
# ========================
# End-to-end exposure/vulnerability collection and risk integration
# for the ICPAC East-Africa Flood IBF system.
#
# Steps
# -----
#   1. Download WorldPop 2020 population GeoTIFFs → mosaic → icechunk ARCO store
#   2. Download INFORM Risk Index 2024 (admin-1) + query OSM critical infra
#   3. Integrate exposure × vulnerability with BN flood-risk outputs
#
# Prerequisites
# -------------
#   uv, icechunk, rasterio, overpy (all installed via inline uv run headers)
#
# Adjust paths below before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOOD_DIR="${SCRIPT_DIR}/../flood_ibf"
ADM1="${FLOOD_DIR}/icpac_adm1v3.geojson"
RISK_CSV_GLOB="${FLOOD_DIR}/output/flood_bn_v1_2026-03-*.csv"

WP_STORE="${SCRIPT_DIR}/worldpop_ea_icechunk"
WP_CACHE="${SCRIPT_DIR}/worldpop_cache"
INFORM_PARQUET="${SCRIPT_DIR}/inform_ea_adm1.parquet"
OSM_PARQUET="${SCRIPT_DIR}/osm_infra_ea_adm1.parquet"
OUT_DIR="${SCRIPT_DIR}/output"

echo "======================================================="
echo " Step 1: WorldPop 2020 → ARCO icechunk store"
echo "======================================================="
uv run "${SCRIPT_DIR}/collect_worldpop_arco.py" \
    --years 2020 \
    --out   "${WP_STORE}" \
    --cache-dir "${WP_CACHE}"

echo ""
echo "======================================================="
echo " Step 2a: INFORM Risk Index → Parquet"
echo "======================================================="
uv run "${SCRIPT_DIR}/collect_inform_osm.py" \
    --adm1        "${ADM1}" \
    --out         "${SCRIPT_DIR}" \
    --cache-dir   "${SCRIPT_DIR}/inform_cache" \
    --inform-only

echo ""
echo "======================================================="
echo " Step 2b: OSM Critical Infrastructure → Parquet"
echo " (skip with Ctrl-C if offline; OSM is optional)"
echo "======================================================="
uv run "${SCRIPT_DIR}/collect_inform_osm.py" \
    --adm1    "${ADM1}" \
    --out     "${SCRIPT_DIR}" \
    --osm-only || echo "[warn] OSM step failed or skipped"

echo ""
echo "======================================================="
echo " Step 3: Integrate exposure × vulnerability"
echo "======================================================="
OSM_ARG=""
if [[ -f "${OSM_PARQUET}" ]]; then
    OSM_ARG="--osm ${OSM_PARQUET}"
fi

uv run "${SCRIPT_DIR}/integrate_exposure_risk.py" \
    --risk-csv    "${RISK_CSV_GLOB}" \
    --worldpop    "${WP_STORE}" \
    --inform      "${INFORM_PARQUET}" \
    ${OSM_ARG} \
    --adm1        "${ADM1}" \
    --out         "${OUT_DIR}"

echo ""
echo "======================================================="
echo " Done. Impact CSVs written to: ${OUT_DIR}"
echo "======================================================="
