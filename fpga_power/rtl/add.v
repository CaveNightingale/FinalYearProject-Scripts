`timescale 1ns / 1ps
// No NaNs or Infs, no garanteed behavior for these cases
// No denormalized numbers, flushed to zero
// Rounding mode: truncate toward zero
module add (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] sum
);
  wire abs_a_greater = a[30:0] > b[30:0];
  wire [31:0] larger = abs_a_greater ? a : b;
  wire [31:0] smaller = abs_a_greater ? b : a;
  wire [7:0] exp_diff = larger[30:23] - smaller[30:23];
  wire [24:0] mant_larger = {1'b1, larger[22:0], 1'b0};
  wire [24:0] mant_smaller = exp_diff == 0 ? {1'b1, smaller[22:0], 1'b0} :
    ({1'b1, smaller[22:0], 1'b0} >> exp_diff);
  wire [25:0] mant_sum = a[31] == b[31] ? mant_larger + mant_smaller : mant_larger - mant_smaller;
  wire leading_1 = mant_sum[25:10] == 0;
  wire [25:0] mant_sum_shifted_1 = leading_1 ? mant_sum << 16 : mant_sum;
  wire leading_2 = mant_sum_shifted_1[25:18] == 0;
  wire [25:0] mant_sum_shifted_2 = leading_2 ? mant_sum_shifted_1 << 8 : mant_sum_shifted_1;
  wire leading_3 = mant_sum_shifted_2[25:22] == 0;
  wire [25:0] mant_sum_shifted_3 = leading_3 ? mant_sum_shifted_2 << 4 : mant_sum_shifted_2;
  wire leading_4 = mant_sum_shifted_3[25:24] == 0;
  wire [25:0] mant_sum_shifted_4 = leading_4 ? mant_sum_shifted_3 << 2 : mant_sum_shifted_3;
  wire leading_5 = mant_sum_shifted_4[25] == 0;
  wire [25:0] mant_sum_shifted_5 = leading_5 ? mant_sum_shifted_4 << 1 : mant_sum_shifted_4;
  wire [4:0] leading_zeros = {leading_1, leading_2, leading_3, leading_4, leading_5};
  wire [8:0] exp_sum = larger[30:23] + 1 - leading_zeros;
  wire [22:0] mant_sum_final = mant_sum_shifted_5[24:2];
  assign sum = a == 0 ? b : b == 0 ? a :
    (mant_sum == 0 || exp_sum[8]) ? 0 : {larger[31], exp_sum[7:0], mant_sum_final};
endmodule
