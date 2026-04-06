`timescale 1ns / 1ps
// Convert fp32 to fp16 with error no more than 1 ulp (and ties are unspecified)
// Note that input does not have subnormals, but output can be subnormal (when input is very small)
// Both input and output do not have inf or NaN, and output subnormal is kept as subnormal
module fp16_cvt (
    input  wire [31:0] in_fp32_ftz,
    output wire [15:0] out_fp16
);
  wire sign = in_fp32_ftz[31];
  wire [7:0] exp_fp32 = in_fp32_ftz[30:23];
  wire [22:0] mant_fp32 = in_fp32_ftz[22:0];
  wire [10:0] mant_fp16 = mant_fp32[22:13] + mant_fp32[12];  // round to nearest
  wire [8:0] exp_fp16 = exp_fp32 + 256 - 127 + 15 + mant_fp16[10];  // adjust exponent and add carry
  wire [7:0] exp_subnormal_shift = (127 - 15) + (23 - 10) - exp_fp32;
  wire [9:0] mant_fp16_subnormal = (({1'b1, mant_fp32[22:0]} >> exp_subnormal_shift) + 1) >> 1;
  assign out_fp16 = exp_fp32 == 0 ? 0 :  // zero
      exp_fp16 < 257 ? {sign, 5'b0, mant_fp16_subnormal[9:0]} :  // subnormal
      exp_fp16 >= 256 + 32 ? {sign, 5'b11111, 10'b1111111111} :  // clamp to max normal
      {sign, exp_fp16[4:0], mant_fp16[9:0]};  // normal
endmodule
