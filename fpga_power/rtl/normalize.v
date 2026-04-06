module normalize (
    input wire sign,
    input wire [8:0] exp,
    input wire [25:0] mant,
    output wire [31:0] fp32
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
  wire [8:0] exp_normalized = exp - leading_zeros;
  assign fp32 = (exp_normalized[8] || mant == 0) ? 0 :
    {sign, exp_normalized[7:0], mant_shifted_5[24:2]};
endmodule
