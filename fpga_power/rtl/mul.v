`timescale 1ns / 1ps
// No NaNs or Infs, no garanteed behavior for these cases
// No denormalized numbers, flushed to zero
// Rounding Mode: Half Up
module mul (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] product
);
  localparam ExponentBias = 127;
  wire sign_prod = a[31] ^ b[31];
  wire [23:0] mant_a = {1'b1, a[22:0]};
  wire [23:0] mant_b = {1'b1, b[22:0]};
  wire [47:0] mant_prod = (mant_a * mant_b) + (1 << 22);  // +0.5 for rounding
  wire [8:0] exp_prod = a[30:23] + b[30:23] - ExponentBias + mant_prod[47];
  wire [22:0] mant_prod_final = mant_prod[47] ? mant_prod[46:24] : mant_prod[45:23];
  assign product = (a[30:23] == 0 || b[30:23] == 0 || exp_prod[8] || exp_prod == 0) ? 0 :
                 {sign_prod, exp_prod[7:0], mant_prod_final};
endmodule
