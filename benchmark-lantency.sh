#!/bin/bash

rm -rf ../LLama-2-7B-Quantized
python quantize.py $@
python benchmark_latency.py