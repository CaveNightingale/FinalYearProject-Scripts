`timescale 1ns / 1ps

module e2m1_ext17 (
    input wire [3:0] in_e2m1,  // exponent bias 1
    input wire [15:0] scale_ue6m10_ftz,
    output wire [16:0] out_e6m10_ftz
);
  wire sign = in_e2m1[3];
  wire [10:0] mant_add = {1'b1, scale_ue6m10_ftz[9:1]} + scale_ue6m10_ftz[0];
  wire [10:0] mant = scale_ue6m10_ftz[9:0] + (in_e2m1[0] ? mant_add : 0);
  wire [6:0] exp_add = scale_ue6m10_ftz[15:10] + in_e2m1[2:1] + mant[10] - 1;
  wire [5:0] exp_0_5 = scale_ue6m10_ftz[15:10] - 1;
  wire [9:0] final_mant = mant[10] ? mant[9:1] : mant[9:0];
  assign out_e6m10_ftz = in_e2m1[2:0] == 0 || exp_add[6] || exp_add == 0 || scale_ue6m10_ftz == 0 ? 0 :  // ftz
      in_e2m1[2:0] == 1 ? {sign, exp_0_5, scale_ue6m10_ftz[9:0]} :  // 0.5
      {sign, exp_add[5:0], final_mant};  // normal
endmodule
