# Benchmark the lantency of a single request
import asyncio

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer
import random
import time
import uuid
import numpy as np
from datasets import load_from_disk

MODEL = "../LLaMA-2-7B-Quantized"

calib = ''.join(load_from_disk("./data/c4_calib_4k")["text"])
llama_tokenizer = LlamaTokenizer.from_pretrained(MODEL)
calib_token_ids = llama_tokenizer(calib)["input_ids"]
vocab = llama_tokenizer.get_vocab()  # token -> id

def random_id() -> str:
    return f"id-{uuid.uuid4()}"

def make_random_sentence(length: int) -> str:
    start = random.randint(0, len(calib_token_ids) - length - 1)
    token_ids = calib_token_ids[start : start + length]
    text = llama_tokenizer.decode(token_ids)
    return text

async def benchmark_lenngth(engine: AsyncLLM, input_length: int = 2048, output_length: int = 2048, num_trials: int = 10):
    print(f"Running latency benchmark with input length {input_length}, output length {output_length}, over {num_trials} trials...")
    e2e = []
    prefill = []
    decode = []
    for _ in range(num_trials):
        prompt = make_random_sentence(input_length)
        sampling_params = SamplingParams(
            max_tokens=output_length,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        )
        await asyncio.sleep(0.5)  # Short pause between trials
        async for output in engine.generate("1 + 1 =", sampling_params=SamplingParams(max_tokens=2), request_id=random_id()):
            pass
        async for output in engine.generate(prompt, sampling_params=sampling_params, request_id=random_id()):
            pass
        latency = output.metrics.last_token_ts - output.metrics.scheduled_ts
        prefill_time = output.metrics.first_token_ts - output.metrics.scheduled_ts
        decode_time = output.metrics.last_token_ts - output.metrics.first_token_ts
        e2e.append(latency)
        prefill.append(prefill_time)
        decode.append(decode_time / (output_length - 1))  # ITL
        sampling_params = SamplingParams(
            max_tokens=output_length,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        )
        await asyncio.sleep(0.5)  # Short pause between trials
    print(f"Average latency over {num_trials} trials: {np.mean(e2e):.6f} seconds, std {np.std(e2e):.6f} seconds")
    print(f"Average prefill time: {np.mean(prefill):.6f} seconds, std {np.std(prefill):.6f} seconds")
    print(f"Average decoding time per token: {np.mean(decode):.6f} seconds, std {np.std(decode):.6f} seconds")
    e2e_median = np.median(e2e)
    prefill_median = np.median(prefill)
    decode_median = np.median(decode)
    print(f"Median latency: {e2e_median:.6f} seconds")
    print(f"Median prefill time: {prefill_median:.6f} seconds")
    print(f"Median decoding time per token: {decode_median:.6f} seconds")
    return e2e_median.item(), prefill_median.item(), decode_median.item()

async def benchmark(engine: AsyncLLM, kv_fp8: bool):
    kv_fp8 = "yes" if kv_fp8 else "no"
    with open("latency_results.txt", "a") as f:
        for lb_input_length in range(8, 12):
            for lb_output_length in range(8, 12):
                input_length = 2 ** lb_input_length
                output_length = 2 ** lb_output_length
                e2e, prefill, decode = await benchmark_lenngth(engine, input_length=input_length, output_length=output_length, num_trials=16)
                f.write(f"{MODEL},{kv_fp8},{input_length},{output_length},{e2e:.9f},{prefill:.9f},{decode:.9f}\n")
                f.flush()  # Ensure results are written to file immediately

async def run_benchmark_with_kv(kv_fp8=False):
    engine_args = AsyncEngineArgs(
        model=MODEL,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=4096,
        max_model_len=4096,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        kv_cache_dtype="fp8" if kv_fp8 else "auto",
    )
    print(f"Initializing engine with args: {engine_args}")
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        warm_up_prompts = [
            "What is the capital of France?",
            "What is the largest mammal?",
            "What is the square root of 16?",
        ]
        print("Warming up the engine with prompts:")
        for prompt in warm_up_prompts:
            sampling_params = SamplingParams(
                max_tokens=16,
                output_kind=RequestOutputKind.DELTA,
            )
            async for output in engine.generate(prompt, sampling_params=sampling_params, request_id=random_id()):
                for completion in output.outputs:
                    new_text = completion.text
                    if new_text:
                        print(f"{new_text}", end="", flush=True)
            print("\n")
        print("Warm-up completed. Starting latency benchmark...")
        await benchmark(engine, kv_fp8)
    finally:
        print("Shutting down the engine...")
        engine.shutdown()

async def main():
    print("Running benchmark with auto KV cache dtype...")
    await run_benchmark_with_kv(kv_fp8=False)
    print("Running benchmark with fp8 KV cache...")
    await run_benchmark_with_kv(kv_fp8=True)

if __name__ == "__main__":
    asyncio.run(main())
    # print(make_random_sentence(512))