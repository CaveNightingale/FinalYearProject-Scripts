import torch
import numpy as np
import torch.nn as nn
import argparse
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
from transformers import PreTrainedModel

class WrapperModel(PreTrainedModel):
    def __init__(self, model):
        nn.Module.__init__(self)
        self._no_split_modules = []
        self.model = model

def int4_quant_args(group_size, zeropoint=False):
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        symmetric=not zeropoint,
        dynamic=False
    )

def e2m1_quant_args(group_size, zeropoint=False):
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        scale_dtype=torch.float8_e4m3fn,
        group_size=group_size,
        symmetric=not zeropoint,
        dynamic=False
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARGS = {
    "int4g128": int4_quant_args(128),
    "int4g64": int4_quant_args(64),
    "int4g32": int4_quant_args(32),
    "int4g128zp": int4_quant_args(128, zeropoint=True),
    "int4g64zp": int4_quant_args(64, zeropoint=True),
    "int4g32zp": int4_quant_args(32, zeropoint=True),
    "nvfp4": e2m1_quant_args(16),
    "int4g32mock": "__int4g32mock",
    "e2m1g16mock": "__e2m1g16mock",
}

e2m1_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def mock_quantize(tensor, quant_args):
    if quant_args == "__int4g32mock":
        grouped = tensor.reshape(-1, 32)
        scale = grouped.abs().amax(dim=1, keepdim=True) / 7.5
        print(scale.shape, grouped.shape)
        quantized = torch.clamp(torch.round(grouped / scale), -7, 7)
        dequantized = quantized * scale
        return dequantized.reshape(tensor.shape)
    elif quant_args == "__e2m1g16mock":
        OFFSET = 2.0 ** 119
        grouped = tensor.reshape(-1, 16)
        scale = grouped.abs().amax(dim=1, keepdim=True) / 6.0
        global_scale = scale.float().abs().amax() / 448.0
        scale = (scale / global_scale).to(torch.float8_e4m3fn).float()
        scaled = (grouped / scale / global_scale).abs()
        quantized = torch.zeros_like(scaled)
        assigned = torch.zeros_like(scaled, dtype=torch.bool)
        for i in range(len(e2m1_values) - 1):
            mid = (e2m1_values[i] + e2m1_values[i + 1]) / 2
            mask = (scaled <= mid if i & 1 == 0 else scaled < mid) & ~assigned
            quantized[mask] = e2m1_values[i]
            assigned |= mask
        quantized[~assigned] = e2m1_values[-1]
        quantized = quantized * torch.sign(grouped)
        quantized /= OFFSET
        global_scale = (global_scale * OFFSET).to(torch.bfloat16).float()
        dequantized = quantized * scale * global_scale
        return dequantized.reshape(tensor.shape)
    else:
        raise ValueError(f"Unsupported quant_args: {quant_args}")

def quantize(tensor, quant_args):
    if isinstance(quant_args, str):
        return mock_quantize(tensor, quant_args)
    in_channel = tensor.shape[0]
    out_channel = tensor.shape[1]
    linear = nn.Linear(out_channel, in_channel, bias=False)
    linear.weight.data.copy_(tensor)
    quant_scheme = QuantizationScheme(
        weights=quant_args,
        targets=["Linear"],
    )
    config_group = {"group_0": quant_scheme}
    mod = QuantizationModifier(
        targets=["Linear"],
        config_groups=config_group,
    )
    recipe = [mod]
    oneshot(
        model=WrapperModel(linear),
        recipe=recipe,
    )
    return fake_quantize(
        linear.weight.data,
        linear.weight_scale.data,
        linear.weight_zero_point.data,
        args=quant_args,
        global_scale=linear.weight_global_scale.data if hasattr(linear, "weight_global_scale") else None
    )

def main(in_channel, out_channel):
    print(f"Quantization error for random tensor of shape ({out_channel}, {in_channel}):")
    tensor = torch.randn(out_channel, in_channel, dtype=torch.float32).to(device)
    weight_mses = {}
    weight_error_means = {}
    for name, quant_args in ARGS.items():
        quantized_tensor = quantize(tensor, quant_args)
        diff = (quantized_tensor.float() - tensor.float()).reshape(-1)
        mse = (diff ** 2).mean().item()
        mean_error = diff.mean().item()
        weight_mses[name] = mse
        weight_error_means[name] = mean_error
    print("| Type       | Squared Error | Mean Error |")
    print("|------------|---------------|------------|")
    for name in ARGS.keys():
        print(f"| {name:<10} | {weight_mses[name]:.9f}  | {weight_error_means[name]:.9f} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization error analysis")
    parser.add_argument("--in-channel", type=int, default=4096, help="Input channel size")
    parser.add_argument("--out-channel", type=int, default=4096, help="Output channel size")
    args = parser.parse_args()
    main(args.in_channel, args.out_channel)