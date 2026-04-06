`timescale 1ns / 1ps

module e4m3_ext17 (
    input wire [7:0] in_e4m3,  // exponent bias 7
    input wire [15:0] scale_ue6m10_ftz,
    output wire [16:0] out_e6m10_ftz
);
  wire sign = in_e4m3[7];
  wire [3:0] shifted_mant_0 = {1'b0, in_e4m3[2:0]};
  wire leading_zero_1 = shifted_mant_0[3:2] == 0;
  wire [3:0] shifted_mant_1 = leading_zero_1 ? shifted_mant_0 << 2 : shifted_mant_0;
  wire leading_zero_2 = shifted_mant_1[3] == 0;
  wire [3:0] shifted_mant_2 = leading_zero_2 ? shifted_mant_1 << 1 : shifted_mant_1;
  wire [1:0] leading_zeros = {leading_zero_1, leading_zero_2};
  wire [4:0] exp = in_e4m3[6:3] != 0 ? {1'b1, in_e4m3[6:3]} : 17 - leading_zeros;
  wire [2:0] mant = in_e4m3[6:3] != 0 ? in_e4m3[2:0] : shifted_mant_2[2:0];
  wire [14:0] mant_prod = {1'b1, mant} * {1'b1, scale_ue6m10_ftz[9:0]};
  wire [5:0] exp_prod = exp + scale_ue6m10_ftz[15:10] + mant_prod[14] - 23;
  wire [10:0] mant_final = mant_prod[14] ? mant_prod[13:3] : mant_prod[12:2];
  assign out_e6m10_ftz = in_e4m3[6:0] == 0 || exp_prod == 0 || scale_ue6m10_ftz == 0 ? 0 :
        {sign, exp_prod, mant_final[10:1]} + mant_final[1];
endmodule
