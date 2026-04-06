`timescale 1ns / 1ps

// Extend the 8-bit floating point number to 9 bits by adding 1 more exponent bit
// and convert subnormal numbers to normal numbers by adjusting the exponent and mantissa
module fp8_ext (
    input  wire [7:0] in_e4m3,
    output wire [8:0] out_e5m3_ftz
);
  wire sign = in_e4m3[7];
  wire [3:0] shifted_mant_0 = {1'b0, in_e4m3[2:0]};
  wire leading_zero_1 = shifted_mant_0[3:0] == 0;
  wire [3:0] shifted_mant_1 = leading_zero_1 ? shifted_mant_0 << 3 : shifted_mant_0;
  wire leading_zero_2 = shifted_mant_1[3:2] == 0;
  wire [3:0] shifted_mant_2 = leading_zero_2 ? shifted_mant_1 << 2 : shifted_mant_1;
  wire leading_zero_3 = shifted_mant_2[3] == 0;
  wire [3:0] shifted_mant_3 = leading_zero_3 ? shifted_mant_2 << 1 : shifted_mant_2;
  wire [2:0] leading_zeros = {leading_zero_1, leading_zero_2, leading_zero_3};
  wire [4:0] exp = in_e4m3[6:3] != 0 ? {1'b1, in_e4m3[6:3]} : 17 - leading_zeros;
  wire [2:0] mant = in_e4m3[6:3] != 0 ? in_e4m3[2:0] : shifted_mant_3[2:0];
  assign out_e5m3_ftz = in_e4m3[7:0] == 0 ? 0 : {sign, exp, mant};
endmodule
