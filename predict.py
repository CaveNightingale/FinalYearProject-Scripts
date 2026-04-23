import argparse
import math
import re

import numpy as np


BASELINE_WORD_PPL = 18.0785
BASELINE_LOG_PPL = math.log(BASELINE_WORD_PPL)
LINEAR_WEIGHT_ELEMENTS = 6476005376
NON_QUANTIZED_ELEMENTS = 262412288
NUM_HIDDEN_LAYERS = 32
HIDDEN_SIZE = 4096
KV_ELEMENTS_PER_TOKEN = 2 * HIDDEN_SIZE * NUM_HIDDEN_LAYERS
MONTE_CARLO_SAMPLES = 1 << 18

TYPE_BITS = {
    "e2m1": 4,
    "int4": 4,
    "e4m3": 8,
    "int8": 8,
    "e5m10": 16,
}

WEIGHT_K = {
    ("GPTQ", "integer", False): 3.50776,
    ("GPTQ", "integer", True): 3.15066,
    ("GPTQ", "float", False): 5.42808,
    ("AWQ", "integer", False): 6.03438,
    ("AWQ", "integer", True): 4.05217,
    ("AWQ", "float", False): 4.20382,
    ("RTN", "integer", False): 7.56832,
    ("RTN", "integer", True): 10.07440,
    ("RTN", "float", False): 8.08094,
}

WEIGHT_FIT_STATS = {
    ("GPTQ", "integer", False): (0.00000274377, 0.00000735011),
    ("GPTQ", "integer", True): (0.00000031579, 0.00000072579),
    ("AWQ", "integer", False): (0.00002155663, 0.00004604933),
    ("AWQ", "integer", True): (0.00000294156, 0.00000785641),
    ("RTN", "integer", False): (0.00000101000, 0.00000200112),
    ("RTN", "integer", True): (0.00003366037, 0.00008645267),
}

KV_ACT_LOG_DELTAS = {
    "AWQ": {
        "KV8": 0.00925147,
        "A8": 0.01401755,
        "KV8A8": 0.01630965,
    },
    "Other": {
        "KV8": 0.01733844,
        "A8": 0.01688828,
        "KV8A8": 0.01949285,
    },
}

KV_ACT_FIT_STATS = {
    "AWQ": {
        "KV8": (0.000012774605, 0.000018395431),
        "A8": (0.000074364961, 0.000107085545),
        "KV8A8": (0.000086173312, 0.000124089569),
    },
    "Other": {
        "KV8": (0.000029947673, 0.000035640205),
        "A8": (0.000010545093, 0.000012549533),
        "KV8A8": (0.000011380906, 0.000013544218),
    },
}

LATENCY_COEFFS = {
    False: {
        "a": 5.70723e-4,
        "b": 1.03891e-12,
        "c": 1.76975e-12,
        "train_mse": 2.91676e-8,
        "loocv_mse": 2.957625e-8,
    },
    True: {
        "a": -12.24177e-4,
        "b": 1.53963e-12,
        "c": 1.77136e-12,
        "train_mse": 1.62342e-8,
        "loocv_mse": 1.65660e-8,
    },
}

TTFT_COEFFS = {
    False: {
        "a": -0.003923471416673796,
        "b": 1.916365738078171e-12,
        "c": 8.217825604779592e-05,
        "d": 3.725772500892736e-09,
        "train_mse": 3.430715997623731e-05,
        "loocv_mse": 3.5160984081429594e-05,
    },
    True: {
        "a": -0.004698044669026239,
        "b": 2.6343875894362176e-12,
        "c": 5.063860139531439e-05,
        "d": 3.0858586702479528e-09,
        "train_mse": 1.1839620471071669e-05,
        "loocv_mse": 1.2287995166125678e-05,
    },
}

FPGA_GEMM_POWER = {
    ("e2m1", False): 0.472,
    ("e2m1", True): 0.348,
    ("int4", False): 0.506,
    ("int4", True): 0.368,
    ("e4m3", False): 0.495,
    ("e4m3", True): 0.353,
    ("int8", False): 0.522,
    ("e5m10", False): 0.485,
}

FPGA_ON_CHIP_POWER = {
    ("e2m1", False): 0.722,
    ("e2m1", True): 0.534,
    ("int4", False): 0.756,
    ("int4", True): 0.555,
    ("e4m3", False): 0.782,
    ("e4m3", True): 0.539,
    ("int8", False): 0.823,
    ("e5m10", False): 0.784,
}

FPGA_DRAM_POWER = {
    ("e2m1", False): 0.141,
    ("e2m1", True): 0.137,
    ("int4", False): 0.144,
    ("int4", True): 0.139,
    ("e4m3", False): 0.275,
    ("e4m3", True): 0.266,
    ("int8", False): 0.279,
    ("e5m10", False): 0.532,
}

MSE_CACHE = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict perplexity, memory footprint, 4090 first-token/inter-token latency, and FPGA power for LLaMA-2-7B quantization schemes."
    )
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--model-length", type=int, default=4096)
    parser.add_argument("--algo", choices=["RTN", "AWQ", "GPTQ"], default="RTN")
    parser.add_argument("--type", choices=["e2m1", "int4", "e4m3", "int8", "e5m10"], default="e5m10")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--kv-fp8", action="store_true")
    parser.add_argument("--act-fp8", action="store_true")
    return parser


def type_bitwidth(type_name: str) -> int:
    return TYPE_BITS[type_name]


def effective_group_size(type_name: str, group_size: int) -> int:
    if type_name in {"e4m3", "e5m10"}:
        return -1
    return group_size


def uses_zero_point(type_name: str, symmetric: bool) -> bool:
    return type_name.startswith("int") and not symmetric


def weight_family(type_name: str) -> str:
    return "float" if type_name in {"e2m1", "e4m3"} else "integer"


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.input_length <= 0:
        parser.error("--input-length must be positive")
    if args.output_length <= 0:
        parser.error("--output-length must be positive")
    if args.model_length <= 0:
        parser.error("--model-length must be positive")
    if args.group_size <= 0:
        parser.error("--group-size must be positive")

    if args.type == "int4" and args.group_size not in {32, 64, 128}:
        parser.error("int4 is only supported with group sizes 32, 64, or 128")
    if args.type == "int8" and args.group_size not in {32, 64, 128}:
        parser.error("int8 is only supported with group sizes 32, 64, or 128")
    if args.type == "e2m1" and args.group_size != 16:
        parser.error("e2m1 is only supported with group size 16")

    if args.type in {"e5m10", "e4m3", "e2m1"} and (args.kv_fp8 or args.act_fp8):
        parser.error("KV-cache and activation FP8 perplexity fitting is only supported for integer weight quantization")

    if args.act_fp8 and args.type != "int4":
        parser.error("FPGA power lookup with activation FP8 is only supported for int4 weights in this predictor")


def quantize(
    data: np.ndarray,
    type_name: str,
    symmetric: bool,
    group_size: int,
    scale_type: str | None,
    global_scale_type: str | None,
) -> np.ndarray:
    def quantize_scale(
        scales: np.ndarray,
        local_scale_type: str | None,
        local_global_scale_type: str | None,
    ) -> np.ndarray:
        if local_scale_type is None:
            return np.ones_like(scales, dtype=np.float32)
        if local_scale_type == "e8m23":
            return scales
        return quantize(
            scales,
            local_scale_type,
            symmetric=True,
            group_size=-1,
            scale_type=local_global_scale_type,
            global_scale_type=None,
        )

    original_shape = data.shape
    group_size = group_size if group_size > 0 else data.size
    data = data.astype(np.float32).reshape(-1, group_size)

    if type_name.startswith("int"):
        bits = int(type_name[3:])
        qmin = -2 ** (bits - 1)
        qmax = 2 ** (bits - 1) - 1
        if symmetric:
            scale = np.abs(data).max(axis=1) / ((qmax - qmin) / 2)
            scale = quantize_scale(scale, scale_type, global_scale_type)
            quants = np.round(data / scale[:, None])
            quants = np.clip(quants, qmin, qmax)
            dequants = quants * scale[:, None]
        else:
            scale = (data.max(axis=1) - data.min(axis=1)) / (qmax - qmin)
            scale = quantize_scale(scale, scale_type, global_scale_type)
            zero_point = np.round(-data.min(axis=1) / scale).astype(np.int32)
            quants = np.round(data / scale[:, None] + zero_point[:, None])
            quants = np.clip(quants, 0, qmax - qmin)
            dequants = (quants - zero_point[:, None]) * scale[:, None]
    elif re.match(r"e\d+m\d+", type_name):
        exponent_bits = int(type_name[1:type_name.find("m")])
        mantissa_bits = int(type_name[type_name.find("m") + 1:])
        exponent_bias = (1 << (exponent_bits - 1)) - 1
        max_exponent = (1 << exponent_bits) - 1 - exponent_bias
        qmax = (1 + (1 - 2 ** (-mantissa_bits))) * (2 ** max_exponent)
        scale = np.abs(data).max(axis=1) / qmax
        scale = quantize_scale(scale, scale_type, global_scale_type)
        sample_scaled = data / scale[:, None]
        sample_bits = sample_scaled.view(np.int32)

        sign = (sample_bits >> 31) & 0x1
        exponent = ((sample_bits >> 23) & 0xFF) - 127
        mantissa = sample_bits & 0x7FFFFF

        normal_mantissa = (mantissa >> (23 - mantissa_bits)) + (
            (mantissa >> (23 - mantissa_bits - 1)) & 0x1
        )
        normal_exponent = exponent + exponent_bias + (normal_mantissa >> mantissa_bits)
        normal_mantissa = normal_mantissa & ((1 << mantissa_bits) - 1)
        normal_mantissa = np.where(
            normal_exponent < (1 << exponent_bits),
            normal_mantissa,
            (1 << mantissa_bits) - 1,
        )
        normal_exponent = np.where(
            normal_exponent < (1 << exponent_bits),
            normal_exponent,
            (1 << exponent_bits) - 1,
        )

        subnormal_mantissa = mantissa | (1 << 23)
        subnormal_mantissa = (
            subnormal_mantissa >> (23 - mantissa_bits + 1 - exponent_bias - exponent)
        ) + (
            (subnormal_mantissa >> (23 - mantissa_bits + 1 - exponent_bias - exponent - 1)) & 0x1
        )
        subnormal_exponent = np.where(subnormal_mantissa == 1 << mantissa_bits, 1, 0)
        subnormal_mantissa = np.where(subnormal_mantissa == 1 << mantissa_bits, 0, subnormal_mantissa)

        final_mantissa = np.where(exponent > -exponent_bias, normal_mantissa, subnormal_mantissa)
        final_exponent = np.where(exponent > -exponent_bias, normal_exponent, subnormal_exponent)

        finite_quants = (
            (sign << (exponent_bits + mantissa_bits))
            | (final_exponent << mantissa_bits)
            | final_mantissa
        )
        quants = np.where(
            np.abs(sample_scaled) <= qmax,
            finite_quants,
            (sign << (exponent_bits + mantissa_bits))
            | (((1 << exponent_bits) - 1) << mantissa_bits)
            | ((1 << mantissa_bits) - 1),
        )

        quants_sign = (quants >> (exponent_bits + mantissa_bits)) & 0x1
        quants_exponent = (quants >> mantissa_bits) & ((1 << exponent_bits) - 1)
        quants_mantissa = quants & ((1 << mantissa_bits) - 1)
        dequants_normal = (
            (1 + quants_mantissa / (1 << mantissa_bits))
            * (2.0 ** (quants_exponent - exponent_bias))
        ) * scale[:, None]
        dequants_subnormal = (quants_mantissa / (1 << mantissa_bits)) * scale[:, None]
        dequants = np.where(quants_exponent > 0, dequants_normal, dequants_subnormal)
        dequants = np.where(quants_sign == 1, -dequants, dequants)
    else:
        raise ValueError(f"Unsupported quantization type: {type_name}")

    return dequants.reshape(original_shape).astype(np.float32)


def quant_mse_key(type_name: str, symmetric: bool, group_size: int) -> tuple[str, bool, int]:
    return (type_name, symmetric, group_size)


def scale_spec(type_name: str) -> tuple[str | None, str | None]:
    if type_name == "e2m1":
        return ("e4m3", "e4m3")
    if type_name != "e5m10":
        return ("e5m10", None)
    return (None, None)


def estimate_quant_mse(type_name: str, symmetric: bool, group_size: int) -> float:
    key = quant_mse_key(type_name, symmetric, group_size)
    if key in MSE_CACHE:
        return MSE_CACHE[key]

    sample_group_size = group_size if group_size > 0 else 4096
    scale_type, global_scale_type = scale_spec(type_name)
    sample_data = np.random.standard_normal(MONTE_CARLO_SAMPLES).astype(np.float32)
    dequantized = quantize(
        sample_data,
        type_name,
        symmetric,
        sample_group_size,
        scale_type,
        global_scale_type,
    )
    mse = float(np.mean((sample_data - dequantized) ** 2))
    MSE_CACHE[key] = mse
    return mse


def weight_bits_per_parameter(type_name: str, group_size: int, zero_point: bool) -> float:
    bits = float(type_bitwidth(type_name))
    if group_size > 0:
        bits += 16.0 / group_size
        if zero_point:
            bits += type_bitwidth(type_name) / group_size
    return bits


def parameter_memory_bytes(type_name: str, group_size: int, zero_point: bool) -> float:
    bits = weight_bits_per_parameter(type_name, group_size, zero_point)
    return LINEAR_WEIGHT_ELEMENTS * bits / 8.0 + NON_QUANTIZED_ELEMENTS * 2.0


def kv_cache_bytes(model_length: int, kv_fp8: bool) -> float:
    bytes_per_element = 1.0 if kv_fp8 else 2.0
    return model_length * KV_ELEMENTS_PER_TOKEN * bytes_per_element


def kv_cache_access_bytes(input_length: int, output_length: int, kv_fp8: bool) -> float:
    bytes_per_element = 1.0 if kv_fp8 else 2.0
    return (input_length + output_length / 2.0) * KV_ELEMENTS_PER_TOKEN * bytes_per_element


def fit_variance(train_mse: float, loocv_mse: float) -> float:
    return math.sqrt(train_mse * loocv_mse)


def variance_to_rmse(variance: float) -> float:
    return math.sqrt(variance)


def predict_perplexity(
    algo: str,
    type_name: str,
    group_size: int,
    symmetric: bool,
    kv_fp8: bool,
    act_fp8: bool,
) -> tuple[float, float | None]:
    log_ppl = BASELINE_LOG_PPL
    log_variance_terms = []

    if type_name != "e5m10":
        family = weight_family(type_name)
        zero_point = uses_zero_point(type_name, symmetric)
        mse = estimate_quant_mse(type_name, symmetric, group_size)
        log_ppl += WEIGHT_K[(algo, family, zero_point)] * mse
        stats = WEIGHT_FIT_STATS.get((algo, family, zero_point))
        if stats is None:
            return math.exp(log_ppl), None
        log_variance_terms.append(fit_variance(*stats))

    if kv_fp8 or act_fp8:
        delta_key = "KV8A8" if kv_fp8 and act_fp8 else "KV8" if kv_fp8 else "A8"
        family_key = algo if algo == "AWQ" else "Other"
        log_ppl += KV_ACT_LOG_DELTAS[family_key][delta_key]
        log_variance_terms.append(fit_variance(*KV_ACT_FIT_STATS[family_key][delta_key]))

    ppl = math.exp(log_ppl)
    if not log_variance_terms:
        return ppl, None

    total_log_variance = sum(log_variance_terms)
    total_log_rmse = variance_to_rmse(total_log_variance)
    ppl_rmse = ppl * (math.exp(total_log_rmse) - 1.0)
    return ppl, ppl_rmse


def predict_memory_footprint_gib(
    type_name: str,
    group_size: int,
    symmetric: bool,
    kv_fp8: bool,
    model_length: int,
) -> float:
    zero_point = uses_zero_point(type_name, symmetric)
    total_bytes = parameter_memory_bytes(type_name, group_size, zero_point) + kv_cache_bytes(model_length, kv_fp8)
    return total_bytes / (2 ** 30)


def predict_latency(
    type_name: str,
    group_size: int,
    symmetric: bool,
    kv_fp8: bool,
    act_fp8: bool,
    input_length: int,
    output_length: int,
) -> tuple[float, float]:
    zero_point = uses_zero_point(type_name, symmetric)
    coeffs = LATENCY_COEFFS[act_fp8]
    weight_bytes = parameter_memory_bytes(type_name, group_size, zero_point)
    kv_access_bytes_value = kv_cache_access_bytes(input_length, output_length, kv_fp8)
    latency = coeffs["a"] + coeffs["b"] * weight_bytes + coeffs["c"] * kv_access_bytes_value
    variance = fit_variance(coeffs["train_mse"], coeffs["loocv_mse"])
    return latency, variance_to_rmse(variance)


def predict_ttft(
    type_name: str,
    group_size: int,
    symmetric: bool,
    act_fp8: bool,
    input_length: int,
) -> tuple[float, float]:
    zero_point = uses_zero_point(type_name, symmetric)
    coeffs = TTFT_COEFFS[act_fp8]
    weight_bytes = parameter_memory_bytes(type_name, group_size, zero_point)
    ttft = (
        coeffs["a"]
        + coeffs["b"] * weight_bytes
        + coeffs["c"] * input_length
        + coeffs["d"] * (input_length ** 2)
    )
    variance = fit_variance(coeffs["train_mse"], coeffs["loocv_mse"])
    return ttft, variance_to_rmse(variance)


def lookup_fpga_powers(type_name: str, act_fp8: bool) -> tuple[float, float, float, float]:
    key = (type_name, act_fp8)
    if key not in FPGA_GEMM_POWER or key not in FPGA_ON_CHIP_POWER or key not in FPGA_DRAM_POWER:
        raise ValueError("Unsupported FPGA power lookup for this weight and activation combination")
    gemm_power = FPGA_GEMM_POWER[key]
    on_chip_power = FPGA_ON_CHIP_POWER[key]
    dram_power = FPGA_DRAM_POWER[key]
    total_power = on_chip_power + dram_power
    return gemm_power, on_chip_power, dram_power, total_power


def format_metric(value: float, rmse: float | None, decimals: int) -> str:
    if rmse is None:
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}f} +/- {rmse:.{decimals}f}"


def propagate_e2e_latency(ttft: float, ttft_rmse: float, itl: float, itl_rmse: float, output_length: int) -> tuple[float, float]:
    e2e_latency = ttft + itl * output_length
    e2e_rmse = math.sqrt(ttft_rmse ** 2 + (output_length * itl_rmse) ** 2)
    return e2e_latency, e2e_rmse


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    group_size = effective_group_size(args.type, args.group_size)
    perplexity, perplexity_rmse = predict_perplexity(
        args.algo,
        args.type,
        group_size,
        args.symmetric,
        args.kv_fp8,
        args.act_fp8,
    )
    memory_gib = predict_memory_footprint_gib(
        args.type,
        group_size,
        args.symmetric,
        args.kv_fp8,
        args.model_length,
    )
    ttft, ttft_rmse = predict_ttft(
        args.type,
        group_size,
        args.symmetric,
        args.act_fp8,
        args.input_length,
    )
    latency, latency_rmse = predict_latency(
        args.type,
        group_size,
        args.symmetric,
        args.kv_fp8,
        args.act_fp8,
        args.input_length,
        args.output_length,
    )
    e2e_latency, e2e_latency_rmse = propagate_e2e_latency(
        ttft,
        ttft_rmse,
        latency,
        latency_rmse,
        args.output_length,
    )
    gemm_power, on_chip_power, dram_power, total_power = lookup_fpga_powers(args.type, args.act_fp8)

    print(f"Perplexity: {format_metric(perplexity, perplexity_rmse, 4)}")
    print(f"Memory Footprint (GiB): {memory_gib:.6f}")
    print(f"4090 First-token Latency (s): {format_metric(ttft, ttft_rmse, 6)}")
    print(f"4090 Inter-token Latency (s): {format_metric(latency, latency_rmse, 6)}")
    print(f"4090 End-to-end Latency (s): {format_metric(e2e_latency, e2e_latency_rmse, 6)}")
    print(f"FPGA GEMM Power (W): {gemm_power:.3f}")
    print(f"FPGA Total On-chip Power (W): {on_chip_power:.3f}")
    print(f"FPGA DRAM Power (W): {dram_power:.3f}")
    print(f"FPGA Total Power (W): {total_power:.3f}")


if __name__ == "__main__":
    main()
