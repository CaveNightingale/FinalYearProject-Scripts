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

CONFIGS=(
  "aw16_q0 16 0"
  "aw16_q1 16 1"
  "aw8_q0 8 0"
  "aw8_q1 8 1"
)

echo "=== Building 4 matmul configs (synth+impl+timing+dcp) ==="

for cfg in "${CONFIGS[@]}"; do
  read -r CFG_NAME ACT_W Q_EN <<<"$cfg"
  CFG_DIR="$OUT_ROOT/$CFG_NAME"
  mkdir -p "$CFG_DIR"

  echo "[RUN] $CFG_NAME (ActivationWidth=$ACT_W, WeightQuantized=$Q_EN)"

  TCL_FILE="/tmp/run_matmul_timing_${CFG_NAME}.tcl"
  cat > "$TCL_FILE" <<EOF
set root_dir [pwd]
set part "xcku3p-ffva676-2-e"
set top  "matmul_top"
set out_dir "\${root_dir}/${CFG_DIR}"
file mkdir \$out_dir

create_project -in_memory -part \$part
read_verilog [glob \${root_dir}/rtl/*.v]
set_property top \$top [current_fileset]

synth_design -top \$top -part \$part -generic ActivationWidth=${ACT_W} -generic WeightQuantized=${Q_EN}

if {[llength [get_ports clk]] > 0} {
    create_clock -period 5.000 -name sys_clk [get_ports clk]
}

report_utilization  -file \${out_dir}/report_utilization_synth.txt
report_timing_summary -file \${out_dir}/report_timing_synth.txt
report_power -file \${out_dir}/report_power_synth.txt
report_timing -max_paths 50 -delay_type max -sort_by group -file \${out_dir}/report_timing_synth_paths.txt

opt_design
place_design -directive Explore
phys_opt_design
route_design -directive Explore
phys_opt_design

write_checkpoint -force \${out_dir}/impl_${CFG_NAME}.dcp

report_utilization  -file \${out_dir}/report_utilization_impl.txt
report_timing_summary -file \${out_dir}/report_timing_impl.txt
report_timing -max_paths 100 -delay_type max -sort_by group -file \${out_dir}/report_timing_impl_paths.txt
report_power -file \${out_dir}/report_power_impl.txt
report_clock_utilization -file \${out_dir}/report_clock_utilization.txt

close_project
exit 0
EOF

  vivado -mode batch \
    -source "$TCL_FILE" \
    -journal "$CFG_DIR/vivado_timing.jou" \
    -log "$CFG_DIR/vivado_timing.log"
done
