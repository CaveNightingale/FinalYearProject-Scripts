#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/scripts/test_logs"
BUILD_ROOT="$ROOT_DIR/obj_dir/tests"
VERILATOR_BIN="/usr/bin/verilator"
VIVADO_SETTINGS="/mnt/extra/local/AMD/2025.2/Vivado/.settings64-Vivado.sh"

mkdir -p "$LOG_DIR" "$BUILD_ROOT"

[[ -f "$VIVADO_SETTINGS" ]] && source "$VIVADO_SETTINGS"

PASS_LIST=()
FAIL_LIST=()

run_matmul_e2e_case() {
  local case_name="$1"
  local case_macro="$2"
  local batch_mode="$3"
  local top_name="tb_matmul_top_wnam"
  local proj_root="$ROOT_DIR"
  local tb_file="$proj_root/tb/${top_name}.v"
  local rtl_dir="$proj_root/rtl"
  local test_tag="${case_name}"
  local build_dir="$BUILD_ROOT/$test_tag"
  local compile_log="$LOG_DIR/${test_tag}.compile.log"
  local run_log="$LOG_DIR/${test_tag}.run.log"

  local batch_desc="BATCH24"
  if [[ "$batch_mode" == "unbatched" ]]; then
    batch_desc="BATCH1"
  fi

  echo "[RUN] tb/${case_name}.v (-D${case_macro} ${batch_desc}, src=tb/${top_name}.v)"

  rm -rf "$build_dir"
  mkdir -p "$build_dir"

  if [[ "$batch_mode" == "unbatched" ]]; then
    if ! "$VERILATOR_BIN" -Wall -Wno-fatal --binary --timing \
      -D"$case_macro" -DBATCH1 \
      --top-module "$top_name" \
      "$tb_file" "$rtl_dir"/*.v \
      --build-jobs 16 \
      --Mdir "$build_dir" >"$compile_log" 2>&1; then
      echo "[FAIL] tb/${case_name}.v"
      FAIL_LIST+=("tb/${case_name}.v (compile, ${case_macro}, ${batch_desc})")
      return
    fi
  else
    if ! "$VERILATOR_BIN" -Wall -Wno-fatal --binary --timing \
      -D"$case_macro" \
      --top-module "$top_name" \
      "$tb_file" "$rtl_dir"/*.v \
      --build-jobs 16 \
      --Mdir "$build_dir" >"$compile_log" 2>&1; then
      echo "[FAIL] tb/${case_name}.v"
      FAIL_LIST+=("tb/${case_name}.v (compile, ${case_macro}, ${batch_desc})")
      return
    fi
  fi

  if ! (cd "$proj_root" && stdbuf -oL "$build_dir/V${top_name}") >"$run_log" 2>&1; then
    echo "[FAIL] tb/${case_name}.v"
    FAIL_LIST+=("tb/${case_name}.v (run, ${case_macro}, ${batch_desc})")
    return
  fi

  if grep -Eq "\[PASS\]" "$run_log"; then
    echo "[PASS] tb/${case_name}.v"
    PASS_LIST+=("tb/${case_name}.v (${case_macro}, ${batch_desc})")
  else
    echo "[FAIL] tb/${case_name}.v"
    FAIL_LIST+=("tb/${case_name}.v (no-pass-line, ${case_macro}, ${batch_desc})")
  fi
}

run_matmul_e2e_all() {
  run_matmul_e2e_case "tb_matmul_top_we5m10ae5m10_batched" "CASE_WE5M10_AE5M10" "batched"
  run_matmul_e2e_case "tb_matmul_top_we5m10ae5m10_unbatched" "CASE_WE5M10_AE5M10" "unbatched"
  run_matmul_e2e_case "tb_matmul_top_wint8ae5m10_batched" "CASE_WINT8_AE5M10" "batched"
  run_matmul_e2e_case "tb_matmul_top_wint8ae5m10_unbatched" "CASE_WINT8_AE5M10" "unbatched"
  run_matmul_e2e_case "tb_matmul_top_we4m3ae5m10_batched" "CASE_WE4M3_AE5M10" "batched"
  run_matmul_e2e_case "tb_matmul_top_we4m3ae5m10_unbatched" "CASE_WE4M3_AE5M10" "unbatched"
  run_matmul_e2e_case "tb_matmul_top_wint4ae5m10_batched" "CASE_WINT4_AE5M10" "batched"
  run_matmul_e2e_case "tb_matmul_top_wint4ae5m10_unbatched" "CASE_WINT4_AE5M10" "unbatched"
  run_matmul_e2e_case "tb_matmul_top_we2m1ae5m10_batched" "CASE_WE2M1_AE5M10" "batched"
  run_matmul_e2e_case "tb_matmul_top_we2m1ae5m10_unbatched" "CASE_WE2M1_AE5M10" "unbatched"
  run_matmul_e2e_case "tb_matmul_top_we4m3ae4m3_batched" "CASE_WE4M3_AE4M3" "batched"
  run_matmul_e2e_case "tb_matmul_top_we4m3ae4m3_unbatched" "CASE_WE4M3_AE4M3" "unbatched"
  run_matmul_e2e_case "tb_matmul_top_wint4ae4m3_batched" "CASE_WINT4_AE4M3" "batched"
  run_matmul_e2e_case "tb_matmul_top_wint4ae4m3_unbatched" "CASE_WINT4_AE4M3" "unbatched"
  run_matmul_e2e_case "tb_matmul_top_we2m1ae4m3_batched" "CASE_WE2M1_AE4M3" "batched"
  run_matmul_e2e_case "tb_matmul_top_we2m1ae4m3_unbatched" "CASE_WE2M1_AE4M3" "unbatched"
}

run_matmul_e2e_named() {
  local tb="$1"
  case "$tb" in
    tb_matmul_top_we5m10ae5m10_batched)
      run_matmul_e2e_case "$tb" "CASE_WE5M10_AE5M10" "batched"
      ;;
    tb_matmul_top_we5m10ae5m10_unbatched)
      run_matmul_e2e_case "$tb" "CASE_WE5M10_AE5M10" "unbatched"
      ;;
    tb_matmul_top_wint8ae5m10_batched)
      run_matmul_e2e_case "$tb" "CASE_WINT8_AE5M10" "batched"
      ;;
    tb_matmul_top_wint8ae5m10_unbatched)
      run_matmul_e2e_case "$tb" "CASE_WINT8_AE5M10" "unbatched"
      ;;
    tb_matmul_top_we4m3ae5m10_batched)
      run_matmul_e2e_case "$tb" "CASE_WE4M3_AE5M10" "batched"
      ;;
    tb_matmul_top_we4m3ae5m10_unbatched)
      run_matmul_e2e_case "$tb" "CASE_WE4M3_AE5M10" "unbatched"
      ;;
    tb_matmul_top_wint4ae5m10_batched)
      run_matmul_e2e_case "$tb" "CASE_WINT4_AE5M10" "batched"
      ;;
    tb_matmul_top_wint4ae5m10_unbatched)
      run_matmul_e2e_case "$tb" "CASE_WINT4_AE5M10" "unbatched"
      ;;
    tb_matmul_top_we2m1ae5m10_batched)
      run_matmul_e2e_case "$tb" "CASE_WE2M1_AE5M10" "batched"
      ;;
    tb_matmul_top_we2m1ae5m10_unbatched)
      run_matmul_e2e_case "$tb" "CASE_WE2M1_AE5M10" "unbatched"
      ;;
    tb_matmul_top_we4m3ae4m3_batched)
      run_matmul_e2e_case "$tb" "CASE_WE4M3_AE4M3" "batched"
      ;;
    tb_matmul_top_we4m3ae4m3_unbatched)
      run_matmul_e2e_case "$tb" "CASE_WE4M3_AE4M3" "unbatched"
      ;;
    tb_matmul_top_wint4ae4m3_batched)
      run_matmul_e2e_case "$tb" "CASE_WINT4_AE4M3" "batched"
      ;;
    tb_matmul_top_wint4ae4m3_unbatched)
      run_matmul_e2e_case "$tb" "CASE_WINT4_AE4M3" "unbatched"
      ;;
    tb_matmul_top_we2m1ae4m3_batched)
      run_matmul_e2e_case "$tb" "CASE_WE2M1_AE4M3" "batched"
      ;;
    tb_matmul_top_we2m1ae4m3_unbatched)
      run_matmul_e2e_case "$tb" "CASE_WE2M1_AE4M3" "unbatched"
      ;;
    *)
      return 1
      ;;
  esac
  return 0
}

run_one() {
  local tb_name="$1"
  local proj_root="$ROOT_DIR"
  local tb_file="$proj_root/tb/${tb_name}.v"
  local rtl_dir="$proj_root/rtl"
  local test_tag="$tb_name"
  local build_dir="$BUILD_ROOT/$test_tag"
  local compile_log="$LOG_DIR/${test_tag}.compile.log"
  local run_log="$LOG_DIR/${test_tag}.run.log"

  echo "[RUN] tb/${tb_name}.v"

  rm -rf "$build_dir"
  mkdir -p "$build_dir"

  if ! "$VERILATOR_BIN" -Wall -Wno-fatal --binary --timing \
    --top-module "$tb_name" \
    "$tb_file" "$rtl_dir"/*.v \
    --build-jobs 16 \
    --Mdir "$build_dir" >"$compile_log" 2>&1; then
    echo "[FAIL] tb/${tb_name}.v"
    FAIL_LIST+=("tb/${tb_name}.v (compile)")
    return
  fi

  if ! (cd "$proj_root" && stdbuf -oL "$build_dir/V${tb_name}") >"$run_log" 2>&1; then
    echo "[FAIL] tb/${tb_name}.v"
    FAIL_LIST+=("tb/${tb_name}.v (run)")
    return
  fi

  if grep -Eq "\\[PASS\\]" "$run_log"; then
    echo "[PASS] tb/${tb_name}.v"
    PASS_LIST+=("tb/${tb_name}.v")
  else
    echo "[FAIL] tb/${tb_name}.v"
    FAIL_LIST+=("tb/${tb_name}.v (no-pass-line)")
  fi
}

if [[ $# -gt 0 ]]; then
  for tb in "$@"; do
    case "$tb" in
      tb_matmul_top_wnam)
        run_matmul_e2e_all
        ;;
      *)
        if ! run_matmul_e2e_named "$tb"; then
          run_one "$tb"
        fi
        ;;
    esac
  done
else
  run_one "tb_add"
  run_one "tb_blink"
  run_one "tb_dequant"
  run_one "tb_fp16_cvt"
  run_one "tb_gemm"
  run_one "tb_gemm_wnam"
  run_one "tb_mul"
  run_one "tb_sram"
  run_one "tb_pe"
  run_one "tb_systolic_array"
  run_matmul_e2e_all
fi

echo "PASS: ${#PASS_LIST[@]}"
echo "FAIL: ${#FAIL_LIST[@]}"

if [[ ${#FAIL_LIST[@]} -gt 0 ]]; then
  for item in "${FAIL_LIST[@]}"; do
    echo "$item"
  done
  exit 1
fi
