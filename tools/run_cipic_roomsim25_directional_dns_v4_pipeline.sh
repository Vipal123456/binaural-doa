#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk2/bywang/DOA-net"
PYTHON="/home/bywang/miniconda3/envs/doa/bin/python"
INVENTORY_ROOT="$ROOT/data/dns3_directional_v4_inventory"
INVENTORY="$INVENTORY_ROOT/dns3_noise_inventory.csv"
INVENTORY_SUMMARY="$INVENTORY_ROOT/dns3_noise_inventory_summary.json"
SMOKE_ROOT="$ROOT/data/librispeech_cipic_roomsim25_directional_dns_v4_smoke"
FULL_ROOT="$ROOT/data/librispeech_cipic_roomsim25_directional_dns_v4"

cd "$ROOT"
echo "[$(date --iso-8601=seconds)] Waiting for audited DNS3 inventory"
while [[ ! -s "$INVENTORY" || ! -s "$INVENTORY_SUMMARY" ]]; do
  if ! pgrep -f "build_dns3_noise_inventory.py.*dns3_directional_v4_inventory" >/dev/null; then
    echo "[$(date --iso-8601=seconds)] DNS3 inventory process ended without a complete inventory" >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date --iso-8601=seconds)] Starting three-split smoke generation"
"$PYTHON" -u tools/generate_cipic_roomsim25_directional_dns_v4.py \
  --noise_inventory "$INVENTORY" \
  --output_root "$SMOKE_ROOT" \
  --mode smoke \
  --workers 3 \
  --parallel_splits

echo "[$(date --iso-8601=seconds)] Smoke passed; starting parallel full generation"
"$PYTHON" -u tools/generate_cipic_roomsim25_directional_dns_v4.py \
  --noise_inventory "$INVENTORY" \
  --output_root "$FULL_ROOT" \
  --mode full \
  --workers 6 \
  --parallel_splits

echo "[$(date --iso-8601=seconds)] Directional DNS v4 generation completed"
