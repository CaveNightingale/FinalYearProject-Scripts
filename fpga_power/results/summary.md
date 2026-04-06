# Summary of FPGA Power Results
## Design
We use a a systolic array matrix multiplication unit as our design under test.
### Dequantization
The design supports the following data types:
- `e5m10`: floating point with 5 exponent bits and 10 mantissa bits, symmetric
- `int8`: 8-bit integer, asymmetric
- `e4m3`: floating point with 4 exponent bits and 3 mantissa bits, symmetric
- `int4`: 4-bit integer, asymmetric
- `e2m1`: floating point with 2 exponent bits and 1 mantissa bit, symmetric

where asymmetric means that the quantization zero point is not necessarily 0, and symmetric means that the quantization zero point is 0. For weight quantization, group size is set to 64, and for activation quantization, group size is the entire tensor.

All the float types does not support `+inf`, `-inf`, `NaN`, `-0`, and only normalized, subnormal numbers and `0` are supported. Bit-patterns for `+inf`, `-inf`, `NaN` are used to represent normalized numbers (i.e. 0x7f in `e5m10` means `-1 * 0b11111111111 * 2 ** (0b11111 - 15)`), and the intepretation of the bit-patterns for `-0` (e.g. `0x80` in `e5m10`) is undefined and may cause chaotic results.

The design dequantizes the input data in the following way:
- If the activation bitwidth is larger than the weight bitwidth, then the weight is dequantized to the activation bitwidth, with a scale same bitwidth as the activation bitwidth.
- If the weight bitwidth is equal to the activation bitwidth, then no dequantization is performed.
- If the weight bitwidth is smaller than the activation bitwidth, then we don't support this configuration and output constant 0.
### Matrix Multiplication
A 4 x 16 systolic array is used to perform the matrix multiplication Y = XW. The data type of X is the activation data type, and the data type of W is the weight data type, but the output data type is always `e5m10` regardless of the input data types.
- Matrix tiles are of 4 x 16 regions in the output matrix Y.
- The input activation goes into the 4 rows of the systolic array, and moves one step to the right every cycle. The input weight goes into the 16 columns of the systolic array, and moves one step down every cycle. Inputs are delayed $i$ cycles for the $i$-th row and $j$ cycles for the $j$-th column to create the wavefront pattern.
- The clear signal goes with the activation input, when processing elements (PEs) receive input activation with clear signal, they replace the current sum with the multiplication result instead of adding the multiplication result to the current sum. When a new tile begins, the first input activations of the tile carry the clear signal not to accumulate with the previous tile's result.
- In each cycle, each PE multiplies the input activation and weight, keeping them in exact precision (i.e. no rounding, truncation, or overflow). Then one accumulator sum up all products in odd cycles, and another accumulator sum up all products in even cycles. These sums are stored in `e8m23` format, but without subnormal numbers (i.e. flush to zero when the exponent is all 0s). Odd-cycle sums and even-cycle sums are added into `e5m10` format at the output step of each tile.

### Testset
We generate 8 random test cases, one for each combination of weight and activation data types. 
- The input activation is a B x 4096 tensor, where each element is randomly generated from a standard laplace distribution. It is quantized to the activation data type, rounded to nearest but with ties rounded away from zero, saturated to maximum representable value if overflow.
- The input weight is a 4096 x 22016 tensor, where each element is randomly generated from a standard normal distribution. It is quantized with min/max calibration, and group size of 64. The scale is first determined and quantized to the required bitwidth, then the quantized weight is obtained by rounding the original weight divided by the quantized scale to nearest, but with ties rounded away from zero.
- The output is a B x 22016 tensor, where each element is computed by dequantizing the input activation and weight, performing the matrix multiplication in IEEE `e11m52` format, and quantizing the result to `e5m10` format by rounding to nearest but with ties rounded away from zero.
- There are two modes for running the test cases: batched or unbatched. In the batched mode, the batch size B is 24 (i.e. using entire input file), and in the unbatched mode, the batch size B is 1 (i.e. using the first column of the input activation and the full input weight).

## Power Results
We run the design on Vivado's simulator with the generated test cases, and collect the power reports. Here we present the power results. The staic power is 0.456 W.

### Compute Unit Total Dynamic Power
Power estimated by Vivado 2025.2 on xcku3p-ffva676-2-e, with a clock frequency of 200 MHz (default setting in Vivado).

|  | e5m10 (batched) | e4m3 (batched) | e5m10 (unbatched) | e4m3 (unbatched) |
| --- | --- | --- | --- | --- |
| e2m1 | 0.996 | 0.723 | 0.558 | 0.400 |
| int4 | 1.013 | 0.738 | 0.581 | 0.413 |
| e4m3 | 1.021 | 0.726 | 0.601 | 0.408 |
| int8 | 1.044 | N/A | 0.624 | N/A |
| e5m10 | 0.995 | N/A | 0.594 | N/A |

Summary: The total dynamic power is mostly affected by the activation data type. And the 16-bit activation (e5m10) consumes around 1.4x more dynamic power than the 8-bit activation (e4m3).

### Gemm Unit Dynamic Power
|  | e5m10 (batched) | e4m3 (batched) | e5m10 (unbatched) | e4m3 (unbatched) |
| --- | --- | --- | --- | --- |
| e2m1 | 0.790 | 0.572 | 0.334 | 0.232 |
| int4 | 0.806 | 0.588 | 0.356 | 0.244 |
| e4m3 | 0.809 | 0.586 | 0.354 | 0.240 |
| int8 | 0.830 | N/A | 0.371 | N/A |
| e5m10 | 0.812 | N/A | 0.351 | N/A |

Summary: Same pattern as the total dynamic power.

### DRAM Dynamic Power (Estimated through manufacturer's datasheet)
Using DDR4-2400, the DRAM dynamic power is estimated to be 0.083 W/GB/s. Static power is 0.336W.

|  | e5m10 (batched) | e4m3 (batched) | e5m10 (unbatched) | e4m3 (unbatched) |
| --- | --- | --- | --- | --- |
| e2m1 | 0.024 | 0.023 | 0.141 | 0.137 |
| int4 | 0.024 | 0.023 | 0.144 | 0.139 |
| e4m3 | 0.046 | 0.044 | 0.275 | 0.266 |
| int8 | 0.046 | N/A | 0.279 | N/A |
| e5m10 | 0.089 | N/A | 0.532 | N/A |

Summary: The DRAM dynamic power is mostly affected by the weight data type, which determines the DRAM bandwidth. The 16-bit weight (e5m10) consumes around 2x more DRAM dynamic power than the 8-bit weight (int8 or e4m3), and around 3.7x more DRAM dynamic power than the 4-bit weight (e2m1 or int4).

## Limitations
- This design is not fully pipelined and optimized for higher frequency, which may significantly reduce the time required for the computation.
- The power results are estimated by Vivado's simulator, which may not be accurate. The actual power consumption may be different when the design is implemented on the FPGA.
- The design does not have any gating or clock enable signal, so even the idle PEs consume dynamic power, which may overestimate the dynamic power in unbatched mode.
