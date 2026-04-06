#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

VIVADO_SETTINGS="/mnt/extra/local/AMD/2025.2/Vivado/.settings64-Vivado.sh"
if [[ ! -f "$VIVADO_SETTINGS" ]]; then
  echo "[ERROR] Vivado environment not found: $VIVADO_SETTINGS"
  echo "[HINT] 请确认 /mnt/extra 已挂载。"
  exit 1
fi

source "$VIVADO_SETTINGS"

OUT_ROOT="vivado_projects/matmul_top_eval"
mkdir -p "$OUT_ROOT"
rm -rf "$OUT_ROOT"/*

USE_GATE_LEVEL_SAIF=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate-level-saif)
      USE_GATE_LEVEL_SAIF=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--gate-level-saif]"
      echo "  --gate-level-saif   Run end-to-end gate-level SAIF flow"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: $0 [--gate-level-saif]"
      exit 1
      ;;
  esac
done

echo "=== [1/3] Build 4 matmul implementations ==="
./scripts/run_matmul_timings.sh

echo "=== [2/3] Run xsim SAIF + power for 16 use cases ==="
if [[ "$USE_GATE_LEVEL_SAIF" == "1" ]]; then
  ./scripts/run_matmul_top_xsim_saif_power.sh --gate-level-saif
else
  ./scripts/run_matmul_top_xsim_saif_power.sh
fi

echo "=== [3/3] Done ==="
