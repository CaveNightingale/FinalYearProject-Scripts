`timescale 1ns / 1ps
// 1 more exponent bit than standard FP16, exponent bias is 47
// totally 17 bits, 1 sign bit, 6 exponent bits, 10 mantissa bits
// Processing Element for FP16 in systolic array
// Cycles delay for one multiplication:
//    1 cycle for Stage 1
//    2 cycles for Stage 2 & 3 (add_denormalize and normalize)
//  Additional 2 cycles for summing up even and odd cycles is placed outside of this PE

module pe #(
  parameter ActivationWidth = 16
)(
    input wire clk,
    input wire rst,
    input wire en,
    input wire clear_in,
    input wire [ActivationWidth:0] a_in,
    input wire [ActivationWidth:0] b_in,
    output reg clear_out,
    output reg [ActivationWidth:0] a_out,
    output reg [ActivationWidth:0] b_out,
    output reg [31:0] sum_out
);
  // Pipeline Stage 1: multiply a and b, get the product in FP32
  wire [31:0] add_denormalize_b_in;
  if (ActivationWidth == 16) begin : g_fp16 // e6m10 ftz
    wire sign_prod = a_in[16] ^ b_in[16];
    wire [10:0] mant_a = {1'b1, a_in[9:0]};
    wire [10:0] mant_b = {1'b1, b_in[9:0]};
    wire [21:0] mant_prod = mant_a * mant_b;  // 11 bits * 11 bits = 22 bits, always fit in 23 bits
    wire [7:0] exp_prod = a_in[15:10] + b_in[15:10] + 33 + mant_prod[21];  // 127 - 47 * 2 = 33
    wire [20:0] mant_prod_final = mant_prod[21] ? mant_prod[20:0] : {mant_prod[19:0], 1'b0};
    wire [29:0] prod = (a_in[15:10] == 0 || b_in[15:10] == 0) ? 0 :
                  {sign_prod, exp_prod, mant_prod_final};
    reg [29:0] prod_reg;
    assign add_denormalize_b_in = {prod_reg, 2'b0};
    always @(posedge clk or negedge rst) begin
      if (!rst) begin
        prod_reg <= 0;
      end else if (en) begin
        prod_reg <= prod;
      end
    end
  end else if (ActivationWidth == 8) begin : g_fp8 // e5m3 ftz
    wire sign_prod = a_in[8] ^ b_in[8];
    wire [3:0] mant_a = {1'b1, a_in[2:0]};
    wire [3:0] mant_b = {1'b1, b_in[2:0]};
    wire [7:0] mant_prod = mant_a * mant_b;
    wire [7:0] exp_prod = a_in[7:3] + b_in[7:3] + 81 + mant_prod[7];  // 127 - 23 * 2 = 81
    wire [6:0] mant_prod_final = mant_prod[7] ? mant_prod[6:0] : {mant_prod[5:0], 1'b0};
    wire [15:0] prod = (a_in[7:3] == 0 || b_in[7:3] == 0) ? 0 :
                  {sign_prod, exp_prod, mant_prod_final};
    reg [15:0] prod_reg;
    assign add_denormalize_b_in = {prod_reg, 16'b0};
    always @(posedge clk or negedge rst) begin
      if (!rst) begin
        prod_reg <= 0;
      end else if (en) begin
        prod_reg <= prod;
      end
    end
  end

  // Pipeline Stage 2 & 3: reduce the sum to 2 FP32 numbers, respectively the sum of odd cycles
  // and the sum of even cycles.
  reg clear_sum;
  wire add_denormalize_sign_out;
  wire [8:0] add_denormalize_exp_out;
  wire [25:0] add_denormalize_mant_out;
  reg normalize_sign_in;
  reg [8:0] normalize_exp_in;
  reg [25:0] normalize_mant_in;
  wire [31:0] normalize_fp32_out;
  add_denormalize add_denormalize_inst (
      .a(clear_sum ? 0 : sum_out),
      .b(add_denormalize_b_in),
      .sign(add_denormalize_sign_out),
      .exp(add_denormalize_exp_out),
      .mant(add_denormalize_mant_out)
  );
  normalize normalize_inst (
      .sign(normalize_sign_in),
      .exp (normalize_exp_in),
      .mant(normalize_mant_in),
      .fp32(normalize_fp32_out)
  );
  always @(posedge clk or negedge rst) begin
    if (!rst) begin
      normalize_sign_in <= 0;
      normalize_exp_in <= 0;
      normalize_mant_in <= 0;
      sum_out <= 0;
      clear_sum <= 0;
      a_out <= 0;
      b_out <= 0;
      clear_out <= 0;
    end else if (en) begin
      normalize_sign_in <= add_denormalize_sign_out;
      normalize_exp_in <= add_denormalize_exp_out;
      normalize_mant_in <= add_denormalize_mant_out;
      sum_out <= normalize_fp32_out;
      clear_sum <= clear_in || clear_out;
      a_out <= a_in;
      b_out <= b_in;
      clear_out <= clear_in;
    end
  end
endmodule
