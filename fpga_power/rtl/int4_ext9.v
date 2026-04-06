`timescale 1ns / 1ps

// Dequantize an int4 value to e5m3 format with FTZ
// Rounding: round to nearest, ties to up
// Overflow: undefined
module int4_ext9 (
    input  wire [3:0] in_int4,
    input  wire [3:0] zero_int4,
    input  wire [7:0] scale_ue5m3_ftz,
    output wire [8:0] out_e5m3_ftz
);
  wire sign = in_int4 < zero_int4;
  wire [3:0] abs_int4 = sign ? zero_int4 - in_int4 : in_int4 - zero_int4;
  wire [7:0] mant_prod = abs_int4 * {1'b1, scale_ue5m3_ftz[2:0]};
  wire leading_zero_1 = mant_prod[7:4] == 0;
  wire [7:0] shifted_mant_1 = leading_zero_1 ? mant_prod << 4 : mant_prod;
  wire leading_zero_2 = shifted_mant_1[7:6] == 0;
  wire [7:0] shifted_mant_2 = leading_zero_2 ? shifted_mant_1 << 2 : shifted_mant_1;
  wire leading_zero_3 = shifted_mant_2[7] == 0;
  wire [7:0] shifted_mant_3 = leading_zero_3 ? shifted_mant_2 << 1 : shifted_mant_2;
  wire [2:0] leading_zeros = {leading_zero_1, leading_zero_2, leading_zero_3};
  wire [4:0] exp_prod = scale_ue5m3_ftz[7:3] + 4 - leading_zeros;
  wire [7:0] abs_out = {exp_prod, shifted_mant_3[6:4]} + shifted_mant_3[3];
  assign out_e5m3_ftz = abs_int4 == 0 || scale_ue5m3_ftz == 0 ? 0 : {sign, abs_out};
endmodule
