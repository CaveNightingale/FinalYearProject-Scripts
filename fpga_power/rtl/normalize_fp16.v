module normalize_fp16 (
    input wire sign,
    input wire [8:0] exp,
    input wire [25:0] mant,
    output wire [15:0] fp16
);
  wire leading_1 = mant[25:10] == 0;
  wire [25:0] mant_shifted_1 = leading_1 ? mant << 16 : mant;
  wire leading_2 = mant_shifted_1[25:18] == 0;
  wire [25:0] mant_shifted_2 = leading_2 ? mant_shifted_1 << 8 : mant_shifted_1;
  wire leading_3 = mant_shifted_2[25:22] == 0;
  wire [25:0] mant_shifted_3 = leading_3 ? mant_shifted_2 << 4 : mant_shifted_2;
  wire leading_4 = mant_shifted_3[25:24] == 0;
  wire [25:0] mant_shifted_4 = leading_4 ? mant_shifted_3 << 2 : mant_shifted_3;
  wire leading_5 = mant_shifted_4[25] == 0;
  wire [25:0] mant_shifted_5 = leading_5 ? mant_shifted_4 << 1 : mant_shifted_4;
  wire [4:0] leading_zeros = {leading_1, leading_2, leading_3, leading_4, leading_5};
  wire [8:0] exp_normalized = exp - leading_zeros - (127 - 15);
  wire [10:0] mant_subnormal = mant_shifted_5[25:15] >> (- exp_normalized);
  wire [15:0] out_abs = (exp_normalized == 0 || exp_normalized[8]) ?
    {5'b00000, mant_subnormal[10:1]} + mant_subnormal[0] :
    {exp_normalized[4:0], mant_shifted_5[24:15]} + mant_shifted_5[14];
  assign fp16 = mant == 0 ? 0 :
    !exp_normalized[8] && (exp_normalized >= 32 || out_abs[15]) ?
    {sign, 5'b11111, 10'b1111111111} : {sign, out_abs[14:0]};
endmodule
