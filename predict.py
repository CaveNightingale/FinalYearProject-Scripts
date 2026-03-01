# This is a placeholder prediction script without fitted coefficients.

import argparse
import math
import random


WEIGHT_BITS = {
    "int4": 4,
    "e2m1": 4,
    "int8": 8,
    "e4m3": 8,
}

KV_BITS = {
    "e4m3": 8,
}


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError(
        "weight-symmetric only accepts literal lowercase 'true' or 'false'"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict perplexity, energy, TTFT, and ITL from quantization settings."
    )
    parser.add_argument(
        "--weight-type",
        default=None,
        choices=["int4", "e2m1", "int8", "e4m3"],
        help="Weight quantization type",
    )
    parser.add_argument(
        "--weight-symmetric",
        type=parse_bool,
        default=None,
        help="Whether weight quantization is symmetric (only accepts true/false)",
    )
    parser.add_argument(
        "--weight-algorithm",
        default=None,
        choices=["rtn", "awq", "gptq"],
        help="Weight quantization algorithm",
    )
    parser.add_argument(
        "--weight-group-size",
        type=int,
        default=None,
        help="Weight quantization group size",
    )
    parser.add_argument(
        "--kv-type",
        default=None,
        choices=["e4m3"],
        help="KV cache quantization type",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        required=True,
        help="Input sequence length used in latency/energy predictors",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        required=True,
        help="Output sequence length used in energy predictor",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=200000,
        help="Number of Gaussian samples for Monte Carlo GaussianMSE estimation",
    )
    return parser


def estimate_gaussian_mse_monte_carlo(
    quant_type: str,
    bit_width: int,
    group_size: int,
    num_samples: int,
) -> float:
    def build_float_codebook(exp_bits: int, mantissa_bits: int) -> list[float]:
        bias = (2 ** (exp_bits - 1)) - 1
        exp_max = (2**exp_bits) - 1
        mantissa_max = (2**mantissa_bits) - 1
        values = []

        for sign in (0, 1):
            sign_factor = -1.0 if sign == 1 else 1.0
            for exp in range(exp_max + 1):
                for mantissa in range(mantissa_max + 1):
                    # No negative zero. Reuse original -0 code as NaN and exclude it.
                    if sign == 1 and exp == 0 and mantissa == 0:
                        continue

                    if exp == 0:
                        if mantissa == 0:
                            values.append(0.0)
                        else:
                            frac = mantissa / (2**mantissa_bits)
                            val = sign_factor * frac * (2 ** (1 - bias))
                            values.append(val)
                    else:
                        frac = 1.0 + mantissa / (2**mantissa_bits)
                        val = sign_factor * frac * (2 ** (exp - bias))
                        values.append(val)

        # Unique and sorted representable finite values.
        return sorted(set(values))

    def nearest_value(target: float, codebook: list[float]) -> float:
        return min(codebook, key=lambda x: abs(x - target))

    if quant_type == "e2m1":
        float_codebook = build_float_codebook(exp_bits=2, mantissa_bits=1)
    elif quant_type == "e4m3":
        float_codebook = build_float_codebook(exp_bits=4, mantissa_bits=3)
    else:
        float_codebook = []

    levels = 2**bit_width
    mid = (levels - 1) / 2.0
    total_error = 0.0
    total_count = 0

    remaining = num_samples
    while remaining > 0:
        current_group_size = min(group_size, remaining)
        group = [random.gauss(0.0, 1.0) for _ in range(current_group_size)]

        max_abs = max(abs(v) for v in group)
        if quant_type in ("int4", "int8"):
            scale = max_abs / mid if max_abs > 0 else 1.0

            for value in group:
                q = round(value / scale + mid)
                q = min(max(q, 0), levels - 1)
                dequant = (q - mid) * scale
                diff = value - dequant
                total_error += diff * diff
        else:
            max_float_abs = max(abs(v) for v in float_codebook)
            scale = max_abs / max_float_abs if max_abs > 0 else 1.0

            for value in group:
                normalized = value / scale
                quantized_normalized = nearest_value(normalized, float_codebook)
                dequant = quantized_normalized * scale
                diff = value - dequant
                total_error += diff * diff

        total_count += current_group_size
        remaining -= current_group_size

    return total_error / total_count if total_count > 0 else 0.0


def predict_metrics(
    weight_type,
    weight_symmetric,
    weight_algorithm,
    weight_group_size,
    kv_type,
    input_length: int,
    output_length: int,
    mc_samples: int,
) -> dict:
    weight_quantized = (
        weight_type is not None
        and weight_symmetric is not None
        and weight_algorithm is not None
        and weight_group_size is not None
    )
    kv_quantized = kv_type is not None

    weight_bits = WEIGHT_BITS[weight_type] if weight_quantized else 16
    kv_bits = KV_BITS[kv_type] if kv_quantized else 16

    if weight_quantized:
        asym = not weight_symmetric
        effective_weight_algorithm = weight_algorithm
        effective_weight_symmetric = weight_symmetric
    else:
        # Required fallback when weight is not quantized.
        effective_weight_algorithm = "rtn"
        asym = False
        effective_weight_symmetric = True

    # NOTE: These coefficients are placeholder random values.
    # TODO: Fit these coefficients with your benchmark data.
    ppl_non_quantized = 2.42
    kv_penalty_const = 0.028

    # C1 is constant for each (algorithm, asym) pair.
    # NOTE: Placeholder random constants.
    c1_by_algo_asym = {
        ("rtn", False): 138.0,
        ("rtn", True): 149.0,
        ("awq", False): 121.0,
        ("awq", True): 133.0,
        ("gptq", False): 106.0,
        ("gptq", True): 114.0,
    }

    if weight_quantized:
        # GaussianMSE depends on bit-width and group-size.
        # NOTE: Estimated by Monte Carlo with clipped N(0,1) samples in [-4, 4].
        gaussian_mse = estimate_gaussian_mse_monte_carlo(
            quant_type=weight_type,
            bit_width=weight_bits,
            group_size=weight_group_size,
            num_samples=mc_samples,
        )
    else:
        gaussian_mse = 0.0

    c1 = c1_by_algo_asym[(effective_weight_algorithm, asym)]
    kv_penalty = kv_penalty_const if kv_quantized else 0.0
    perplexity = math.exp(ppl_non_quantized + c1 * gaussian_mse + kv_penalty)

    # NOTE: Placeholder random coefficients.
    # TODO: Fit with measured TTFT/ITL data.
    a_ttft_w = 1.25e-3
    a_ttft_kv_l = 8.4e-6
    a_ttft_kv_l2 = 2.6e-9
    a_ttft_bias = 0.017

    b_itl_w = 4.8e-4
    b_itl_kv_l = 3.1e-6
    b_itl_bias = 0.0016

    ttft_sec = (
        a_ttft_w * weight_bits
        + a_ttft_kv_l * kv_bits * input_length
        + a_ttft_kv_l2 * kv_bits * (input_length**2)
        + a_ttft_bias
    )
    itl_sec = b_itl_w * weight_bits + b_itl_kv_l * \
        kv_bits * input_length + b_itl_bias

    # NOTE: Placeholder random coefficients.
    # TODO: Fit with measured energy data.
    e_w = 0.42
    e_kv_in_l = 2.2e-3
    e_kv_out_l = 2.0e-3
    e_kv_in_l2 = 6.0e-7
    e_kv_out_l2 = 5.5e-7
    e_kv_cross = 5.8e-7
    e_bias = 38.0
    e_arith_kv = 0.9

    # KV e4m3 (fp8) arithmetic-unit operation count scales with (input+output)^2.
    kv_compute_count = (input_length + output_length) ** 2

    energy_joule = (
        e_w * weight_bits
        + e_kv_in_l * kv_bits * input_length
        + e_kv_out_l * kv_bits * output_length
        + e_kv_in_l2 * kv_bits * (input_length**2)
        + e_kv_out_l2 * kv_bits * (output_length**2)
        + e_kv_cross * kv_bits * input_length * output_length
        + e_arith_kv * kv_compute_count * (1.0 if kv_quantized else 0.0)
        + e_bias
    )

    return {
        "inputs": {
            "weight_type": weight_type,
            "weight_symmetric": weight_symmetric,
            "weight_algorithm": weight_algorithm,
            "weight_group_size": weight_group_size,
            "kv_type": kv_type,
            "weight_quantized": weight_quantized,
            "kv_quantized": kv_quantized,
            "effective_weight_algorithm": effective_weight_algorithm,
            "effective_asym": asym,
            "effective_weight_symmetric": effective_weight_symmetric,
            "input_length": input_length,
            "output_length": output_length,
            "mc_samples": mc_samples,
            "weight_bits": weight_bits,
            "kv_bits": kv_bits,
            "gaussian_mse": gaussian_mse,
            "ppl_non_quantized": ppl_non_quantized,
            "c1": c1,
            "kv_penalty": kv_penalty,
        },
        "predictions": {
            "perplexity": perplexity,
            "energy_joule": energy_joule,
            "time_to_first_token_sec": ttft_sec,
            "inter_token_latency_sec": itl_sec,
        },
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.input_length <= 0:
        parser.error("input-length must be a positive integer")
    if args.output_length <= 0:
        parser.error("output-length must be a positive integer")
    if args.mc_samples <= 0:
        parser.error("mc-samples must be a positive integer")

    weight_fields = [
        args.weight_type,
        args.weight_symmetric,
        args.weight_algorithm,
        args.weight_group_size,
    ]
    weight_any_present = any(field is not None for field in weight_fields)
    weight_all_present = all(field is not None for field in weight_fields)

    if weight_any_present and not weight_all_present:
        parser.error(
            "if one --weight-* argument appears, all --weight-* arguments "
            "(--weight-type, --weight-symmetric, --weight-algorithm, --weight-group-size) must be present"
        )

    if weight_all_present and args.weight_group_size <= 0:
        parser.error("weight-group-size must be a positive integer")

    result = predict_metrics(
        weight_type=args.weight_type,
        weight_symmetric=args.weight_symmetric,
        weight_algorithm=args.weight_algorithm,
        weight_group_size=args.weight_group_size,
        kv_type=args.kv_type,
        input_length=args.input_length,
        output_length=args.output_length,
        mc_samples=args.mc_samples,
    )

    print("=== Prediction Inputs ===")
    for key, value in result["inputs"].items():
        print(f"{key}: {value}")

    print("\n=== Predicted Metrics ===")
    print(f"perplexity: {result['predictions']['perplexity']:.6f}")
    print(f"energy_joule: {result['predictions']['energy_joule']:.6f}")
    print(
        "time_to_first_token_sec: "
        f"{result['predictions']['time_to_first_token_sec']:.6f}"
    )
    print(
        "inter_token_latency_sec: "
        f"{result['predictions']['inter_token_latency_sec']:.6f}"
    )


if __name__ == "__main__":
    main()
