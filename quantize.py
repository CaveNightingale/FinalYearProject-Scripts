#!/usr/bin/env python3
"""
Python-only quantization script using llmcompressor Python API.
"""

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


import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
import argparse

# compressed_tensors 的量化类型/策略/scheme
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
    QuantizationScheme,
)

def main(type=QuantizationType.INT, bits=4, group_size=64, symmetric=False, algo="AWQ"):
    print(f"Source: {SRC}")
    print(f"Destination: {DST}")

    if not os.path.exists(SRC):
        print(f"Source path does not exist: {SRC}")
        sys.exit(2)

    ensure_dst_parent()

    model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(SRC)

    ds = prepare_calibration_dataset(tokenizer)

    # 只配置权重：W4A16 = 权重4bit + activation fp16
    strategy = QuantizationStrategy.GROUP if type == "int" else QuantizationStrategy.TENSOR_GROUP

    weights_args = QuantizationArgs(
        num_bits=bits,
        type=type,
        strategy=strategy,
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

    model.save_pretrained(DST, save_compressed=True)
    tokenizer.save_pretrained(DST)

    print("Quantization finished. Compressed model saved to:", DST)


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Quantize a model using llmcompressor.")
        parser.add_argument("--bits", type=int, default=4, help="Number of bits for quantization (default: 4)")
        parser.add_argument("--group-size", type=int, default=64, help="Group size for quantization (default: 64)")
        parser.add_argument("--symmetric", action="store_true", help="Use symmetric quantization")
        parser.add_argument("--algo", type=str, choices=["AWQ", "GPTQ", "RTN"], default="AWQ", help="Quantization algorithm (default: AWQ)")
        parser.add_argument("--type", type=str, choices=[QuantizationType.INT, QuantizationType.FLOAT], default=QuantizationType.INT, help="Quantization type (default: int)")
        return parser.parse_args()

    if __name__ == "__main__":
        args = parse_args()
        main(
            type=args.type,
            bits=args.bits,
            group_size=args.group_size,
            symmetric=args.symmetric,
            algo=args.algo
        )
