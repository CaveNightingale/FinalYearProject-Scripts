#!/bin/bash
kv_fp8=0

rm -rf ../LLama-2-7B-Quantized
python quantize.py $@

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kv-fp8) # The only argument that need to pass to vLLM
      kv_fp8=1
      shift
      ;;
    *) # Skip other arguments
      shift
  esac
done

if [[ $kv_fp8 -eq 1 ]]; then
  bash start-vllm.sh --kv-cache-dtype fp8 &
else
  bash start-vllm.sh &
fi

bash start-lm-eval.sh

kill $(jobs -p)
