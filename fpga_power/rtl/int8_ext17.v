`timescale 1ns / 1ps

// Dequantize an int8 value to e6m10 format with FTZ
// Rounding: round to nearest, ties to up
// Overflow: undefined
module int8_ext17 (
    input  wire [ 7:0] in_int8,
    input  wire [ 7:0] zero_int8,
    input  wire [15:0] scale_ue6m10_ftz,
    output wire [16:0] out_e6m10_ftz
);
  wire sign = in_int8 < zero_int8;
  wire [7:0] abs_int8 = sign ? zero_int8 - in_int8 : in_int8 - zero_int8;
  wire [18:0] mant_prod = abs_int8 * {1'b1, scale_ue6m10_ftz[9:0]};
  wire leading_zero_1 = mant_prod[18:15] == 0;
  wire [18:0] shifted_mant_1 = leading_zero_1 ? mant_prod << 4 : mant_prod;
  wire leading_zero_2 = shifted_mant_1[18:17] == 0;
  wire [18:0] shifted_mant_2 = leading_zero_2 ? shifted_mant_1 << 2 : shifted_mant_1;
  wire leading_zero_3 = shifted_mant_2[18] == 0;
  wire [18:0] shifted_mant_3 = leading_zero_3 ? shifted_mant_2 << 1 : shifted_mant_2;
  wire [2:0] leading_zeros = {leading_zero_1, leading_zero_2, leading_zero_3};
  wire [5:0] exp_prod = scale_ue6m10_ftz[15:10] + 8 - leading_zeros;
  wire [15:0] abs_out = {exp_prod, shifted_mant_3[17:8]} + shifted_mant_3[7];
  assign out_e6m10_ftz = abs_int8 == 0 || scale_ue6m10_ftz == 0 ? 0 :
        abs_int8 == 1 ? {sign, scale_ue6m10_ftz} :
        {sign, abs_out};
endmodule
