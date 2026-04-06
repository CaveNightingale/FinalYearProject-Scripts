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

USE_GATE_LEVEL_SAIF=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate-level-saif)
      USE_GATE_LEVEL_SAIF=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--gate-level-saif]"
      echo "  --gate-level-saif   End-to-end gate-level SAIF (simulate gate netlist from impl DCP)"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: $0 [--gate-level-saif]"
      exit 1
      ;;
  esac
done

if [[ "$USE_GATE_LEVEL_SAIF" == "1" ]]; then
  echo "[INFO] SAIF mode: gate-level (e2e netlist simulation)"
else
  echo "[INFO] SAIF mode: rtl(strip_path=tb_matmul_top_wnam/dut)"
fi

XSIM_JOBS="${XSIM_JOBS:-$(nproc)}"
if ! [[ "$XSIM_JOBS" =~ ^[0-9]+$ ]] || [[ "$XSIM_JOBS" -lt 1 ]]; then
  echo "[ERROR] XSIM_JOBS must be a positive integer (got: $XSIM_JOBS)"
  exit 1
fi
echo "[INFO] XSIM parallel jobs: $XSIM_JOBS"

ensure_gate_netlist() {
  local cfg="$1"
  local dcp_path="$OUT_ROOT/$cfg/impl_${cfg}.dcp"
  local netlist_path="$OUT_ROOT/$cfg/matmul_top_${cfg}_gate.v"

  if [[ -f "$netlist_path" ]]; then
    return 0
  fi
  if [[ ! -f "$dcp_path" ]]; then
    echo "[ERROR] DCP not found for gate-level netlist export: $dcp_path"
    echo "[HINT] 先运行 ./scripts/run_matmul_timings.sh"
    exit 1
  fi

  local tcl_file="/tmp/export_gate_netlist_${cfg}.tcl"
  cat > "$tcl_file" <<EOF
if {![file exists "${dcp_path}"]} {
  puts "ERROR: DCP not found: ${dcp_path}"
  exit 1
}
open_checkpoint "${dcp_path}"
write_verilog -force -mode funcsim "${netlist_path}"
close_design
exit 0
EOF

  echo "[INFO] Export gate netlist: $netlist_path"
  vivado -mode batch \
    -source "$tcl_file" \
    -journal "$OUT_ROOT/$cfg/vivado_export_netlist.jou" \
    -log "$OUT_ROOT/$cfg/vivado_export_netlist.log"
}

declare -A CASE_MACRO
declare -A CASE_BATCH
declare -A CASE_CFG

register_case() {
  local name="$1"
  local macro="$2"
  local batch="$3"
  local cfg="$4"
  CASE_MACRO["$name"]="$macro"
  CASE_BATCH["$name"]="$batch"
  CASE_CFG["$name"]="$cfg"
}

register_case "tb_matmul_top_we5m10ae5m10_batched"   "CASE_WE5M10_AE5M10" "batched"   "aw16_q0"
register_case "tb_matmul_top_we5m10ae5m10_unbatched" "CASE_WE5M10_AE5M10" "unbatched" "aw16_q0"
register_case "tb_matmul_top_wint8ae5m10_batched"    "CASE_WINT8_AE5M10"  "batched"   "aw16_q1"
register_case "tb_matmul_top_wint8ae5m10_unbatched"  "CASE_WINT8_AE5M10"  "unbatched" "aw16_q1"
register_case "tb_matmul_top_we4m3ae5m10_batched"    "CASE_WE4M3_AE5M10"  "batched"   "aw16_q1"
register_case "tb_matmul_top_we4m3ae5m10_unbatched"  "CASE_WE4M3_AE5M10"  "unbatched" "aw16_q1"
register_case "tb_matmul_top_wint4ae5m10_batched"    "CASE_WINT4_AE5M10"  "batched"   "aw16_q1"
register_case "tb_matmul_top_wint4ae5m10_unbatched"  "CASE_WINT4_AE5M10"  "unbatched" "aw16_q1"
register_case "tb_matmul_top_we2m1ae5m10_batched"    "CASE_WE2M1_AE5M10"  "batched"   "aw16_q1"
register_case "tb_matmul_top_we2m1ae5m10_unbatched"  "CASE_WE2M1_AE5M10"  "unbatched" "aw16_q1"
register_case "tb_matmul_top_we4m3ae4m3_batched"     "CASE_WE4M3_AE4M3"   "batched"   "aw8_q0"
register_case "tb_matmul_top_we4m3ae4m3_unbatched"   "CASE_WE4M3_AE4M3"   "unbatched" "aw8_q0"
register_case "tb_matmul_top_wint4ae4m3_batched"     "CASE_WINT4_AE4M3"   "batched"   "aw8_q1"
register_case "tb_matmul_top_wint4ae4m3_unbatched"   "CASE_WINT4_AE4M3"   "unbatched" "aw8_q1"
register_case "tb_matmul_top_we2m1ae4m3_batched"     "CASE_WE2M1_AE4M3"   "batched"   "aw8_q1"
register_case "tb_matmul_top_we2m1ae4m3_unbatched"   "CASE_WE2M1_AE4M3"   "unbatched" "aw8_q1"

USE_CASES=(
  tb_matmul_top_we5m10ae5m10_batched
  tb_matmul_top_we5m10ae5m10_unbatched
  tb_matmul_top_wint8ae5m10_batched
  tb_matmul_top_wint8ae5m10_unbatched
  tb_matmul_top_we4m3ae5m10_batched
  tb_matmul_top_we4m3ae5m10_unbatched
  tb_matmul_top_wint4ae5m10_batched
  tb_matmul_top_wint4ae5m10_unbatched
  tb_matmul_top_we2m1ae5m10_batched
  tb_matmul_top_we2m1ae5m10_unbatched
  tb_matmul_top_we4m3ae4m3_batched
  tb_matmul_top_we4m3ae4m3_unbatched
  tb_matmul_top_wint4ae4m3_batched
  tb_matmul_top_wint4ae4m3_unbatched
  tb_matmul_top_we2m1ae4m3_batched
  tb_matmul_top_we2m1ae4m3_unbatched
)

run_case_xsim() {
  local case_name="$1"
  local macro="${CASE_MACRO[$case_name]}"
  local batch="${CASE_BATCH[$case_name]}"
  local cfg="${CASE_CFG[$case_name]}"

  local snapshot="tb_matmul_top_wnam_${case_name}"
  local saif_file="/tmp/${case_name}.saif"
  local case_dir="$ROOT_DIR/$OUT_ROOT/xsim_runs/${case_name}"
  local tcl_file="$case_dir/xsim_saif_${case_name}.tcl"

  mkdir -p "$case_dir"
  echo "[XSIM][START] $case_name (macro=$macro, batch=$batch, cfg=$cfg)"

  (
    cd "$case_dir"
    ln -sfn "$ROOT_DIR/c" "$case_dir/c"

    if [[ "$USE_GATE_LEVEL_SAIF" == "1" ]]; then
      local gate_netlist="$ROOT_DIR/$OUT_ROOT/$cfg/matmul_top_${cfg}_gate.v"
      if [[ "$batch" == "unbatched" ]]; then
        xvlog --relax -sv -d "$macro" -d BATCH1 "$ROOT_DIR/tb/tb_matmul_top_wnam.v" "$gate_netlist"
      else
        xvlog --relax -sv -d "$macro" "$ROOT_DIR/tb/tb_matmul_top_wnam.v" "$gate_netlist"
      fi
      xelab --relax --debug typical -L unisims_ver -L unimacro_ver -L secureip tb_matmul_top_wnam glbl -s "$snapshot"
    else
      if [[ "$batch" == "unbatched" ]]; then
        xvlog --relax -sv -d "$macro" -d BATCH1 "$ROOT_DIR/tb/tb_matmul_top_wnam.v" "$ROOT_DIR"/rtl/*.v
      else
        xvlog --relax -sv -d "$macro" "$ROOT_DIR/tb/tb_matmul_top_wnam.v" "$ROOT_DIR"/rtl/*.v
      fi
      xelab --relax --debug typical tb_matmul_top_wnam -s "$snapshot"
    fi

    cat > "$tcl_file" <<EOF
set cyc_ns 10
set waited 0
set max_wait 200000000
while {1} {
  set en_v ""
  if {![catch {set en_v [string trim [get_value /tb_matmul_top_wnam/en]]}]} {
  } elseif {![catch {set en_v [string trim [examine /tb_matmul_top_wnam/en]]}]} {
  }
  if {[string first "1" \$en_v] >= 0} { break }
  run \${cyc_ns}ns
  incr waited
  if {\$waited > \$max_wait} {
    puts "ERROR: timeout waiting /tb_matmul_top_wnam/en == 1"
    quit
  }
}
run [expr 48000 * \${cyc_ns}]ns
open_saif ${saif_file}
log_saif [get_objects -r /tb_matmul_top_wnam/dut/*]
run [expr 48000 * \${cyc_ns}]ns
close_saif
quit
EOF

    xsim "$snapshot" --tclbatch "$tcl_file" --onfinish quit 2>&1 | tee "/tmp/${case_name}.xsim.log"
  )

  echo "[XSIM][DONE] $case_name"
}

if [[ "$USE_GATE_LEVEL_SAIF" == "1" ]]; then
  for cfg in aw16_q0 aw16_q1 aw8_q0 aw8_q1; do
    ensure_gate_netlist "$cfg"
  done
fi

echo "=== [1/2] Run xsim SAIF for 16 use cases in parallel (en后 48000~96000 cycles) ==="
declare -a xsim_pids=()
for case_name in "${USE_CASES[@]}"; do
  while [[ "$(jobs -pr | wc -l)" -ge "$XSIM_JOBS" ]]; do
    sleep 1
  done
  run_case_xsim "$case_name" &
  xsim_pids+=("$!")
done

xsim_failed=0
for pid in "${xsim_pids[@]}"; do
  if ! wait "$pid"; then
    xsim_failed=1
  fi
done
if [[ "$xsim_failed" -ne 0 ]]; then
  echo "[ERROR] One or more xsim jobs failed."
  exit 1
fi

echo "=== [2/2] Run Vivado power report from SAIF (sequential) ==="
for case_name in "${USE_CASES[@]}"; do
  cfg="${CASE_CFG[$case_name]}"
  saif_file="/tmp/${case_name}.saif"

  dcp_path="$OUT_ROOT/$cfg/impl_${cfg}.dcp"
  rpt_path="$OUT_ROOT/$cfg/report_power_saif_${case_name}.txt"
  rpt_xml="$OUT_ROOT/$cfg/report_power_saif_${case_name}.xml"
  pw_tcl="/tmp/power_from_dcp_${case_name}.tcl"

  cat > "$pw_tcl" <<EOF
if {![file exists "${dcp_path}"]} {
  puts "ERROR: DCP not found: ${dcp_path}"
  exit 1
}
if {![file exists "${saif_file}"]} {
  puts "ERROR: SAIF not found: ${saif_file}"
  exit 1
}
open_checkpoint "${dcp_path}"
if {![catch {read_saif -strip_path tb_matmul_top_wnam/dut "${saif_file}"}]} {
  puts "INFO: SAIF loaded with strip_path=tb_matmul_top_wnam/dut"
} else {
  puts "ERROR: read_saif failed for ${saif_file}; abort to avoid slow vector-less power run"
  close_design
  exit 2
}
report_power -file "${rpt_path}"
report_power -format xml -file "${rpt_xml}"
close_design
exit 0
EOF

  vivado -mode batch \
    -source "$pw_tcl" \
    -journal "$OUT_ROOT/$cfg/vivado_power_${case_name}.jou" \
    -log "$OUT_ROOT/$cfg/vivado_power_${case_name}.log"
done

echo ""
echo "=== Power Summary (16 use cases) ==="
for case_name in "${USE_CASES[@]}"; do
  cfg="${CASE_CFG[$case_name]}"
  rpt="$OUT_ROOT/$cfg/report_power_saif_${case_name}.txt"
  echo "--- $case_name ($cfg) ---"
  grep -E "Total On-Chip Power|Dynamic \(W\)|Device Static \(W\)" "$rpt" | head -n 10 || true
done
