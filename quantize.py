#!/usr/bin/env python3
"""
Python-only quantization script using llmcompressor Python API.
"""

import argparse
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.observers import MinMaxTuple, Observer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
    QuantizationScheme,
)


SRC = "../LLaMA-2-7B"
DST = "../LLaMA-2-7B-Quantized"
DATASET = "./data/c4_calib_4k"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512


def ensure_dst_parent():
    parent = os.path.dirname(DST)
    if not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def prepare_calibration_dataset(tokenizer):
    return load_from_disk(DATASET)

# compressed_tensors 的量化类型/策略/scheme


def main(type=QuantizationType.INT, bits=4, group_size=64, symmetric=False, algo="AWQ"):
    print(f"Source: {SRC}")
    print(f"Destination: {DST}")

    if not os.path.exists(SRC):
        print(f"Source path does not exist: {SRC}")
        sys.exit(2)

    ensure_dst_parent()

    model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(SRC)

    ds = prepare_calibration_dataset(tokenizer)

    if type == QuantizationType.INT:
        weights_args = QuantizationArgs(
            num_bits=bits,
            type=type,
            strategy=QuantizationStrategy.GROUP,
            group_size=group_size,
            symmetric=symmetric,
            dynamic=False,
        )
    elif type == QuantizationType.FLOAT:
        scale_dtype = torch.float8_e4m3fn if bits == 4 else torch.float16
        weights_args = QuantizationArgs(
            num_bits=bits,
            type=type,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            scale_dtype=scale_dtype,
            group_size=group_size,
            symmetric=symmetric,
            dynamic=False,
        )

    quant_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=weights_args
    )

    config_groups = {"group_0": quant_scheme}

    if algo == "AWQ":
        mod = AWQModifier(
            targets="Linear",
            ignore=["lm_head"],
            config_groups=config_groups,
        )
    elif algo == "GPTQ":
        mod = GPTQModifier(
            targets="Linear",
            ignore=["lm_head"],
            config_groups=config_groups,
        )
    elif algo == "RTN":
        mod = QuantizationModifier(
            targets="Linear",
            ignore=["lm_head"],
            config_groups=config_groups,
        )

    recipe = [mod]
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    model.save_pretrained(DST, save_compressed=True,
                          safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(DST)

    print("Quantization finished. Compressed model saved to:", DST)


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Quantize a model using llmcompressor.")
        parser.add_argument("--bits", type=int, default=4,
                            help="Number of bits for quantization (default: 4)")
        parser.add_argument("--group-size", type=int, default=64,
                            help="Group size for quantization (default: 64)")
        parser.add_argument("--symmetric", action="store_true",
                            help="Use symmetric quantization")
        parser.add_argument("--algo", type=str, choices=[
                            "AWQ", "GPTQ", "RTN"], default="RTN", help="Quantization algorithm (default: AWQ)")
        parser.add_argument("--type", type=str, choices=[QuantizationType.INT, QuantizationType.FLOAT, "none"],
                            default="none", help="Quantization type (default: none)")
        # Calibrating KV cache to fp8 will fail for LLaMA-2-7B for unknown reasons, so we keep it as a no-op for now.
        # Just use default scale as a fallback to enable fp8 quantization of KV cache in vLLM, which is pure inference-time operation and does not require calibration.
        parser.add_argument("--kv-fp8", action="store_true",
                            help="Whether to quantize KV cache to fp8 (no-op)")
        return parser.parse_args()

    if __name__ == "__main__":
        args = parse_args()
        main(
            type=args.type,
            bits=args.bits,
            group_size=args.group_size,
            symmetric=args.symmetric,
            algo=args.algo,
        )
