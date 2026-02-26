#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve ../LLaMA-2-7B-Quantized \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4096 \
  --max-model-len 4096 \
  --port 9093