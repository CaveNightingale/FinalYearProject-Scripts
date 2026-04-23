# Final Year Project Scripts

This repository contains some scripts related to the LLaMA-2-7B quantization case study.

These scripts are tested to be runnable on 4090. 

All scripts, assuming that there is one `../LLaMA-2-7B` directory that contains the original LLaMA-2-7B-hf model.

Apart from scripts, `slides` folder contains some slides about this project topic.

## Environment Setup

As described in the final report, create the environment by installing the Python dependencies in this repository first:

```bash
pip install -r requirements.txt
```

For the W4A8 schemes on SM89 GPUs, the report also requires using a patched vLLM source tree. A typical setup flow is:

```bash
cd /path/to/vllm
git apply /path/to/scripts/vllm-4090-w4a8.patch
pip install .
```

To create the calibration subset used by the quantization workflow:

```bash
python create_c4_calib_4k.py
```

All commands below assume you are running inside this `scripts` directory and that the original model is available at `../LLaMA-2-7B`.

## Main Entry Points

The final report describes two main user-facing entry points in this repository: one for benchmarking perplexity and one for prediction.

### `benchmark-ppl.sh`

This is the shell entry point used to benchmark perplexity.

```bash
bash benchmark-ppl.sh \
	[--algo <RTN|AWQ|GPTQ>] \
	[--type <e2m1|int4|e4m3|int8|e5m10>] \
	[--group-size <int>] \
	[--symmetric] \
	[--kv-fp8] \
	[--act-fp8]
```

The meanings follow the report:

- `--algo` defaults to `RTN`.
- `--type` defaults to `e5m10`.
- `--group-size` defaults to `64`.
- without `--symmetric`, integer quantization uses asymmetric quantization.
- without `--kv-fp8`, KV cache is not quantized.
- without `--act-fp8`, activations are not quantized.

### `benchmark-lantency.sh`

This script is the tracked shell entry point for latency benchmarking.

```bash
bash benchmark-lantency.sh \
	[--algo <RTN|AWQ|GPTQ>] \
	[--type <e2m1|int4|e4m3|int8|e5m10>] \
	[--group-size <int>] \
	[--symmetric] \
	[--kv-fp8] \
	[--act-fp8]
```

The filename is spelled `benchmark-lantency.sh` in this repository.

### `fpga_power/scripts/run_matmul_top_eval.sh`

This script runs the FPGA matmul-top evaluation flow.

```bash
bash fpga_power/scripts/run_matmul_top_eval.sh
```

To run the end-to-end gate-level SAIF flow:

```bash
bash fpga_power/scripts/run_matmul_top_eval.sh --gate-level-saif
```

### `predict.sh`

This is the shell entry point used in the report for the fitted predictor.

```bash
bash predict.sh \
	--input-length <int> \
	--output-length <int> \
	[--model-length <int>] \
	[--algo <RTN|AWQ|GPTQ>] \
	[--type <e2m1|int4|e4m3|int8|e5m10>] \
	[--group-size <int>] \
	[--symmetric] \
	[--kv-fp8] \
	[--act-fp8]
```

As described in the report, this script prints the predicted memory footprint, latency, power consumption, and estimated perplexity for the requested quantization scheme and sequence lengths.

