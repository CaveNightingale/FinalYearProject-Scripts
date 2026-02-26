#!/bin/bash
lm-eval \
  --model local-completions \
  --model_args base_url=http://localhost:9093/v1/completions,model=../LLaMA-2-7B-Quantized,api_key=EMPTY,tokenizer=../LLaMA-2-7B-Quantized,max_length=2048,sliding_window=256 \
  --tasks c4 \
  --include_path ./tasks
