`timescale 1ns / 1ps

module e2m1_ext9 (
    input wire [3:0] in_e2m1,  // exponent bias 1
    input wire [7:0] scale_ue5m3_ftz,
    output wire [8:0] out_e5m3_ftz
);
  wire sign = in_e2m1[3];
  wire [3:0] mant_add = {1'b1, scale_ue5m3_ftz[2:1]} + scale_ue5m3_ftz[0];
  wire [3:0] mant = scale_ue5m3_ftz[2:0] + (in_e2m1[0] ? mant_add : 0);
  wire [5:0] exp_add = scale_ue5m3_ftz[7:3] + in_e2m1[2:1] + mant[3] - 1;
  wire [4:0] exp_0_5 = scale_ue5m3_ftz[7:3] - 1;
  wire [2:0] final_mant = mant[3] ? mant[2:1] : mant[2:0];
  assign out_e5m3_ftz = in_e2m1[2:0] == 0 || exp_add[5] || exp_add == 0 || scale_ue5m3_ftz == 0 ? 0 :  // ftz
      in_e2m1[2:0] == 1 ? {sign, exp_0_5, scale_ue5m3_ftz[2:0]} :  // 0.5
      {sign, exp_add[4:0], final_mant};  // normal
endmodule
