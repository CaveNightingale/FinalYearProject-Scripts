`timescale 1ns / 1ps

// Extend the 16-bit floating point number to 17 bits by adding 1 more exponent bit
// and convert subnormal numbers to normal numbers by adjusting the exponent and mantissa
module fp16_ext (
    input  wire [15:0] in_e5m10,
    output wire [16:0] out_e6m10_ftz
);
  wire sign = in_e5m10[15];
  wire [10:0] shifted_mant_0 = {1'b0, in_e5m10[9:0]};
  wire leading_zero_1 = shifted_mant_0[10:3] == 0;
  wire [10:0] shifted_mant_1 = leading_zero_1 ? shifted_mant_0 << 8 : shifted_mant_0;
  wire leading_zero_2 = shifted_mant_1[10:7] == 0;
  wire [10:0] shifted_mant_2 = leading_zero_2 ? shifted_mant_1 << 4 : shifted_mant_1;
  wire leading_zero_3 = shifted_mant_2[10:9] == 0;
  wire [10:0] shifted_mant_3 = leading_zero_3 ? shifted_mant_2 << 2 : shifted_mant_2;
  wire leading_zero_4 = shifted_mant_3[10] == 0;
  wire [10:0] shifted_mant_4 = leading_zero_4 ? shifted_mant_3 << 1 : shifted_mant_3;
  wire [3:0] leading_zeros = {leading_zero_1, leading_zero_2, leading_zero_3, leading_zero_4};
  wire [5:0] exp = in_e5m10[14:10] != 0 ? {1'b1, in_e5m10[14:10]} : 33 - leading_zeros;
  wire [9:0] mant = in_e5m10[14:10] != 0 ? in_e5m10[9:0] : shifted_mant_4[9:0];
  assign out_e6m10_ftz = in_e5m10[14:0] == 0 ? 0 : {sign, exp, mant};
endmodule
