// Add two FP32 numbers, and denormalize numbers
// Which means the output is not normalized
module add_denormalize (
    input wire [31:0] a,
    input wire [31:0] b,
    output wire sign,
    output wire [8:0] exp,
    output wire [25:0] mant
);
  wire abs_a_greater = a[30:0] > b[30:0];
  wire [31:0] larger = abs_a_greater ? a : b;
  wire [31:0] smaller = abs_a_greater ? b : a;
  wire [7:0] exp_diff = larger[30:23] - smaller[30:23];
  wire [24:0] mant_larger = larger == 0 ? 0 : {1'b1, larger[22:0], 1'b0};
  wire [24:0] mant_smaller = smaller == 0 ? 0 : exp_diff == 0 ? {1'b1, smaller[22:0], 1'b0} :
    ({1'b1, smaller[22:0], 1'b0} >> exp_diff);
  wire [25:0] mant_sum = a[31] == b[31] ? mant_larger + mant_smaller : mant_larger - mant_smaller;
  assign sign = larger[31];
  assign exp = larger[30:23] + 1;
  assign mant = mant_sum;
endmodule
