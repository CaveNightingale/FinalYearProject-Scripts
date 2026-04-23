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
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from datasets import load_dataset, load_from_disk
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
    QuantizationScheme,
    fake_quantize,
)

def _get_no_split_modules(self, auto):
    return self._no_split_modules

# patch llm-compressor for transformers 5.3 compatiblity
PreTrainedModel._get_no_split_modules = _get_no_split_modules


SRC = "../LLaMA-2-7B"
DST = "../LLaMA-2-7B-Quantized"
DATASET = "./data/c4_calib_4k"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

TYPE_CONFIG = {
    "e2m1": {"quant_type": QuantizationType.FLOAT, "bits": 4},
    "int4": {"quant_type": QuantizationType.INT, "bits": 4},
    "e4m3": {"quant_type": QuantizationType.FLOAT, "bits": 8},
    "int8": {"quant_type": QuantizationType.INT, "bits": 8},
    "e5m10": {"quant_type": None, "bits": 16},
}


def ensure_dst_parent():
    parent = os.path.dirname(DST)
    if not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def prepare_calibration_dataset(tokenizer):
    return load_from_disk(DATASET)

def main(type_name="int4", group_size=64, symmetric=False, algo="AWQ", act_fp8=False):
    print(f"Source: {SRC}")
    print(f"Destination: {DST}")

    config = TYPE_CONFIG.get(type_name)
    if config is None:
        raise ValueError(f"Unsupported --type: {type_name}")

    quant_type = config["quant_type"]
    bits = config["bits"]

    if not os.path.exists(SRC):
        print(f"Source path does not exist: {SRC}")
        sys.exit(2)

    ensure_dst_parent()

    model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(SRC)

    ds = prepare_calibration_dataset(tokenizer)

    weights_args = None
    if quant_type is not None:
        if group_size <= 0:
            weights_args = QuantizationArgs(
                num_bits=bits,
                type=quant_type,
                strategy=QuantizationStrategy.CHANNEL,
                symmetric=symmetric,
                dynamic=False,
            )
        elif quant_type == QuantizationType.INT or bits != 4:
            weights_args = QuantizationArgs(
                num_bits=bits,
                type=quant_type,
                strategy=QuantizationStrategy.GROUP,
                group_size=group_size,
                symmetric=symmetric,
                dynamic=False,
            )
        else:
            scale_dtype = torch.float8_e4m3fn
            weights_args = QuantizationArgs(
                num_bits=bits,
                type=quant_type,
                strategy=QuantizationStrategy.TENSOR_GROUP,
                scale_dtype=scale_dtype,
                group_size=group_size,
                symmetric=symmetric,
                dynamic=False,
            )

    activations_args = None
    if act_fp8:
        activations_args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
            observer=None,
        )

    if weights_args is not None or activations_args is not None:
        quant_scheme = QuantizationScheme(
            targets=["Linear"],
            weights=weights_args,
            input_activations=activations_args,
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

    model = model.half()
    extra_args = {}
    if bits == 4 and quant_type == QuantizationType.INT:
        extra_args["quantization_format"] = "pack-quantized" # Allow loaded by marlin kernels
    model.save_pretrained(DST, save_compressed=True,
                          safe_serialization=True, max_shard_size="2GB", **extra_args)
    tokenizer.save_pretrained(DST)
            
    print("Quantization finished. Compressed model saved to:", DST)


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Quantize a model using llmcompressor.")
        parser.add_argument("--group-size", type=int, default=64,
                            help="Group size for quantization (default: 64)")
        parser.add_argument("--symmetric", action="store_true",
                            help="Use symmetric quantization")
        parser.add_argument("--algo", type=str, choices=[
                            "AWQ", "GPTQ", "RTN"], default="RTN", help="Quantization algorithm (default: AWQ)")
        parser.add_argument("--type", type=str, choices=["e2m1", "int4", "e4m3", "int8", "e5m10"],
                            default="e5m10", help="Quantization type")
        # Calibrating KV cache to fp8 will fail for LLaMA-2-7B for unknown reasons, so we keep it as a no-op for now.
        # Just use default scale as a fallback to enable fp8 quantization of KV cache in vLLM, which is pure inference-time operation and does not require calibration.
        parser.add_argument("--kv-fp8", action="store_true",
                            help="Whether to quantize KV cache to fp8 (no-op)")
        parser.add_argument("--act-fp8", action="store_true",
                    help="Whether to quantize input activations to fp8")
        return parser.parse_args()

    if __name__ == "__main__":
        args = parse_args()
        main(
            type_name=args.type,
            group_size=args.group_size,
            symmetric=args.symmetric,
            algo=args.algo,
            act_fp8=args.act_fp8,
        )
