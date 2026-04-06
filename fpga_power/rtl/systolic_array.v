`timescale 1ns / 1ps

// A systolic array for matrix multiplication of N x K and K x M matrices
// within K + (N + M - 1) cycles, where K is the inner dimension of the multiplication
// First full row of output is ready N cycles after the last row of input
module systolic_array #(
    parameter N = 4,
    parameter M = 16,
    parameter ActivationWidth = 16
) (
    input wire clk,
    input wire rst,
    input wire en,
    input wire clear,
    input wire [N*(ActivationWidth+1)-1:0] a,  // Flattened N x 1 vector
    input wire [M*(ActivationWidth+1)-1:0] b,  // Flattened 1 x M vector
    output wire [N*M*32-1:0] c  // Flattened N x M matrix
);
  // Internal wires to shift data between processing elements
  wire clear_wires[0:N-1][0:M];
  wire [ActivationWidth:0] a_wires[0:N-1][0:M];
  wire [ActivationWidth:0] b_wires[0:N][0:M-1];

  // Generate the processing elements and connect the wires
  genvar i, j;
  generate
    for (i = 0; i < N; i = i + 1) begin : g_pe_N
      for (j = 0; j < M; j = j + 1) begin : g_pe_M
        pe #(
            .ActivationWidth(ActivationWidth)
        ) pe_inst (
            .clk(clk),
            .rst(rst),
            .en(en),
            .clear_in(clear_wires[i][j]),
            .a_in(a_wires[i][j]),
            .b_in(b_wires[i][j]),
            .clear_out(clear_wires[i][j+1]),
            .a_out(a_wires[i][j+1]),
            .b_out(b_wires[i+1][j]),
            .sum_out(c[(i*M+j+1)*32-1:(i*M+j)*32])
        );
      end
    end
  endgenerate

  // Generate skewed input loading in N directions
  generate
    assign clear_wires[0][0] = clear;
    assign a_wires[0][0] = a[ActivationWidth:0];
    for (i = 1; i < N; i = i + 1) begin : g_input_N
      // Skew the a input for the i-th row by i cycles
      reg clear_skew[0:i-1];
      reg [ActivationWidth:0] a_skew[0:i-1];
      assign a_wires[i][0] = a_skew[0];
      assign clear_wires[i][0] = clear_skew[0];
      for (j = 0; j < i; j = j + 1) begin : g_skew_N
        if (j == i - 1) begin : g_load_N
          always @(posedge clk or negedge rst) begin
            if (!rst) begin
              clear_skew[j] <= 0;
              a_skew[j] <= 0;
            end else if (en) begin
              clear_skew[j] <= clear;
              a_skew[j] <= a[(i+1)*(ActivationWidth+1)-1:i*(ActivationWidth+1)];
            end
          end
        end else begin : g_shift_N
          always @(posedge clk or negedge rst) begin
            if (!rst) begin
              clear_skew[j] <= 0;
              a_skew[j] <= 0;
            end else if (en) begin
              clear_skew[j] <= clear_skew[j+1];
              a_skew[j] <= a_skew[j+1];
            end
          end
        end
      end
    end
  endgenerate

  // Generate skewed input loading in M directions
  generate
    assign b_wires[0][0] = b[ActivationWidth:0];
    for (j = 1; j < M; j = j + 1) begin : g_input_M
      // Skew the b input for the j-th column by j cycles
      reg [ActivationWidth:0] b_skew[0:j-1];
      assign b_wires[0][j] = b_skew[0];
      for (i = 0; i < j; i = i + 1) begin : g_skew_M
        if (i == j - 1) begin : g_load_M
          always @(posedge clk or negedge rst) begin
            if (!rst) begin
              b_skew[i] <= 0;
            end else if (en) begin
              b_skew[i] <= b[(j+1)*(ActivationWidth+1)-1:j*(ActivationWidth+1)];
            end
          end
        end else begin : g_shift_M
          always @(posedge clk or negedge rst) begin
            if (!rst) begin
              b_skew[i] <= 0;
            end else if (en) begin
              b_skew[i] <= b_skew[i+1];
            end
          end
        end
      end
    end
  endgenerate
endmodule
