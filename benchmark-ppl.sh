#!/bin/bash
python quantize.py $@
bash start-vllm.sh &
pid=$!
sleep 5
bash start-lm-eval.sh
kill -term $pid