`timescale 1ns / 1ps

// Pesudo code:
// systolic_array arr = systolic_array_empty();
// for (int block_m = 0; block_m < ceil(out_features / 16.0); block_m++) {
//   for (int block_n = 0; block_n < ceil(batch_size / 4.0); block_n++) {
//      for (int k = 0; k < in_features; k++) {
//          x, w = ? X[block_n * 4:block_n * 4 + 4, k], W^T[block_m * 16:block_m * 16 + 16, k]
//          systolic_array_feed(arr, x, w, k == 0);
//          if ((k >= 4 && k < 20) && (block_m != 0 || block_n != 0)) {
//              y = systolic_array_pull(arr);
//              ! y
//          }
//      }
//   }
// }
// for (int i = 0; i < 4; i++) {
//   ?
// }
// for (int k = 0; k < 16; k++) {
//   ?
//   y = systolic_array_pull(arr);
//   ! y
// }
// Where `?` means querying inputs from external SRAM and wait for the next cycle to get the data,
// and `!` means submitting outputs to external SRAM.
// M is iterated first since this will make the preparation of weight blocks more efficient, as weights
// must be loaded from DRAM to SRAM before computation, monotonically increasing the weight block index
// will result in a sequential access pattern in DRAM.

// A matrix multiplication module.
// Y = X W, X fp16(batch_size, in_features), W^T T(out_features, in_features), Y fp16(batch_size, out_features)
// Block size is 16 for weights, and 4 for activations
// Possible type of W:
//  8-bit quantized:
//    00: e4m3
//    01: int8
//  4-bit quantized:
//    10: e2m1
//    11: int4
// Full-precision W is equivent to 4-bit quantized with scale = abs(W) and quant = sign(W)
// Additionally, if no quantization type is enabled, we will directly use the input scale as the value.
// For quantized weights, the exponent range of the scale is -24 to 16 for fp16 activations, and -9 to 8 for fp8 activations, except for exact zero.
// Typically if the scale is converted from an e5m10, it will natrually fit in the exponent range.
// OVERFLOW AND UNDERFLOW BEHAVIOR IS BOTH UNDEFINED, NOT ZERO! Following the contract of exponent range of the scale will prevent underflow.
module gemm #(
    parameter N = 4,  // Activation block size
    parameter M = 16,  // Weight block size
    parameter ActivationWidth = 16,  // Input activation width, 16 or 8, do not affect output width
    parameter E4M3Enabled = 1,  // Whether to support e4m3 quantization for weights
    parameter Int8Enabled = 1,  // Whether to support int8 quantization for weights
    parameter E2M1Enabled = 1,  // Whether to support e2m1 quantization for weights
    parameter Int4Enabled = 1,  // Whether to support int4 quantization for weights
    parameter QueryInputCycle = 1  // Cycles queries must be issued before the data is ready
) (
    input wire clk,
    input wire rst,
    input wire en,
    input wire clear,  // Clear the internal state for the next matrix multiplication
    input wire [15:0] batch_size,
    input wire [23:0] in_features,
    input wire [15:0] out_features,
    input wire [N*ActivationWidth-1:0] x_data,  // N x fp16
    input wire [1:0] w_type,  // Data type of W
    input wire [M*(ActivationWidth/2)-1:0] w_quant_half,  // Input weight, quantized to half of the activation width
    input wire [M*(ActivationWidth/2)-1:0] w_zero_half,  // Zero point for quantization, same width as w_quant_half
    input wire [M*ActivationWidth-1:0] w_scale_uext_ftz,  // Scale for quantized W
    output reg [N*16-1:0] y_data,  // N x fp16
    output reg [15:0] query_n,
    output reg [15:0] query_m,
    output reg [23:0] query_k,
    output reg [15:0] submit_n,
    output reg [15:0] submit_m,
    output reg query_en,  // Indicates when the module wants more input data for X and W^T
    output reg submit_en,  // Indicates when the output y_data is valid and can be written to memory
    output reg done  // Indicates the completion of the matrix multiplication
);
  wire [2:0] tile_width = batch_size > N ? N : batch_size;
  // Weight dequant, targeting same format as activations
  reg [ActivationWidth:0] w_dequant_ext[0:M-1];
  genvar i, j;
  generate
    for (i = 0; i < M; i = i + 1) begin : g_w_dequant
      if (ActivationWidth == 16) begin : g_w_dequant_fp16
        wire [16:0] w_e4m3_ext;
        if (E4M3Enabled) begin : g_e4m3_fp16
          wire e4m3_enabled = w_type == 2'b00;
          e4m3_ext17 e4m3_ext_inst (
              .in_e4m3(w_quant_half[(i+1)*8-1:i*8] & {8{e4m3_enabled}}),
              .scale_ue6m10_ftz(w_scale_uext_ftz[(i+1)*16-1:i*16] & {16{e4m3_enabled}}),
              .out_e6m10_ftz(w_e4m3_ext)
          );
        end else begin : g_e4m3_disabled_fp16
          assign w_e4m3_ext = 0;
        end
        wire [16:0] w_int8_ext;
        if (Int8Enabled) begin : g_int8_fp16
          wire int8_enabled = w_type == 2'b01;
          int8_ext17 int8_ext_inst (
              .in_int8(w_quant_half[(i+1)*8-1:i*8] & {8{int8_enabled}}),
              .zero_int8(w_zero_half[(i+1)*8-1:i*8] & {8{int8_enabled}}),
              .scale_ue6m10_ftz(w_scale_uext_ftz[(i+1)*16-1:i*16] & {16{int8_enabled}}),
              .out_e6m10_ftz(w_int8_ext)
          );
        end else begin : g_int8_disabled_fp16
          assign w_int8_ext = 0;
        end
        wire [16:0] w_e2m1_ext;
        if (E2M1Enabled) begin : g_e2m1_fp16
          wire e2m1_enabled = w_type == 2'b10;
          e2m1_ext17 e2m1_ext_inst (
              .in_e2m1(w_quant_half[(i+1)*8-5:i*8] & {4{e2m1_enabled}}),
              .scale_ue6m10_ftz(w_scale_uext_ftz[(i+1)*16-1:i*16] & {16{e2m1_enabled}}),
              .out_e6m10_ftz(w_e2m1_ext)
          );
        end else begin : g_e2m1_disabled_fp16
          assign w_e2m1_ext = 0;
        end
        wire [16:0] w_int4_ext;
        if (Int4Enabled) begin : g_int4_fp16
          wire int4_enabled = w_type == 2'b11;
          int4_ext17 int4_ext_inst (
              .in_int4(w_quant_half[(i+1)*8-5:i*8] & {4{int4_enabled}}),
              .zero_int4(w_zero_half[(i+1)*8-5:i*8] & {4{int4_enabled}}),
              .scale_ue6m10_ftz(w_scale_uext_ftz[(i+1)*16-1:i*16] & {16{int4_enabled}}),
              .out_e6m10_ftz(w_int4_ext)
          );
        end else begin : g_int4_disabled_fp16
          assign w_int4_ext = 0;
        end
        wire [16:0] w_fp16_ext;
        if (!Int4Enabled && !E2M1Enabled && !Int8Enabled && !E4M3Enabled) begin : g_no_quant_fp16
          // Directly use scales as the value
          fp16_ext w_fp16_ext_inst (
              .in_e5m10(w_scale_uext_ftz[(i+1)*16-1:i*16]),
              .out_e6m10_ftz(w_fp16_ext)  // Any format works since they are the same for non-quantized values
          );
        end else begin : g_no_quant_fp16_disabled
          assign w_fp16_ext = 0;
        end
        always @(posedge clk or negedge rst) begin
          if (!rst) begin
            w_dequant_ext[i] <= 0;
          end else if (en) begin
            w_dequant_ext[i] <= w_e4m3_ext | w_int8_ext | w_e2m1_ext | w_int4_ext | w_fp16_ext;
          end
        end
      end else if (ActivationWidth == 8) begin : g_w_dequant_fp8
        // For fp8 activations, we only support int4 / e2m1 quantization
        wire [8:0] w_e2m1_ext;
        if (E2M1Enabled) begin : g_e2m1_fp8
          wire e2m1_enabled = w_type == 2'b10;
          e2m1_ext9 e2m1_ext_inst (
              .in_e2m1(w_quant_half[(i+1)*4-1:i*4] & {4{e2m1_enabled}}),
              .scale_ue5m3_ftz(w_scale_uext_ftz[(i+1)*8-1:i*8] & {8{e2m1_enabled}}),
              .out_e5m3_ftz(w_e2m1_ext)
          );
        end else begin : g_e2m1_fp8_disabled
          assign w_e2m1_ext = 0;
        end
        wire [8:0] w_int4_ext;
        if (Int4Enabled) begin : g_int4_fp8
          wire int4_enabled = w_type == 2'b11;
          int4_ext9 int4_ext_inst (
              .in_int4(w_quant_half[(i+1)*4-1:i*4] & {4{int4_enabled}}),
              .zero_int4(w_zero_half[(i+1)*4-1:i*4] & {4{int4_enabled}}),
              .scale_ue5m3_ftz(w_scale_uext_ftz[(i+1)*8-1:i*8] & {8{int4_enabled}}),
              .out_e5m3_ftz(w_int4_ext)
          );
        end else begin : g_int4_fp8_disabled
          assign w_int4_ext = 0;
        end
        wire [8:0] w_fp8_ext;
        if (!Int4Enabled && !E2M1Enabled) begin : g_no_quant_fp8
          // Directly use scales as the value
          fp8_ext w_fp8_ext_inst (
              .in_e4m3(w_scale_uext_ftz[(i+1)*8-1:i*8]),
              .out_e5m3_ftz(w_fp8_ext)
          );
        end else begin : g_no_quant_fp8_disabled
          assign w_fp8_ext = 0;
        end
        always @(posedge clk or negedge rst) begin
          if (!rst) begin
            w_dequant_ext[i] <= 0;
          end else if (en) begin
            w_dequant_ext[i] <= w_e2m1_ext | w_int4_ext | w_fp8_ext;
          end
        end
      end
    end
  endgenerate

  // Input dequant
  reg [ActivationWidth:0] x_dequant_ext[0:N-1];
  generate
    if (ActivationWidth == 16) begin : g_x_fp16
      for (i = 0; i < N; i = i + 1) begin : g_x_dequant
        wire [16:0] x_dequant_e6m10_ftz_element;  // Pipeline registers for dequantized X
        fp16_ext x_fp16_ext_inst (
            .in_e5m10(x_data[(i+1)*16-1:i*16]),
            .out_e6m10_ftz(x_dequant_e6m10_ftz_element)
        );
        always @(posedge clk or negedge rst) begin
          if (!rst) begin
            x_dequant_ext[i] <= 0;
          end else if (en) begin
            x_dequant_ext[i] <= x_dequant_e6m10_ftz_element;
          end
        end
      end
    end else if (ActivationWidth == 8) begin : g_x_fp8
      for (i = 0; i < N; i = i + 1) begin : g_x_dequant_fp8
        wire [8:0] x_dequant_e5m3_ftz_element;  // Pipeline registers for dequantized X
        fp8_ext x_fp8_ext_inst (
            .in_e4m3(x_data[(i+1)*8-1:i*8]),
            .out_e5m3_ftz(x_dequant_e5m3_ftz_element)
        );
        always @(posedge clk or negedge rst) begin
          if (!rst) begin
            x_dequant_ext[i] <= 0;
          end else if (en) begin
            x_dequant_ext[i] <= x_dequant_e5m3_ftz_element;
          end
        end
      end
    end
  endgenerate


  // IO Control
  wire [15:0] total_blocks_n = (batch_size + N - 1) >> $clog2(N);  // Activation block size N
  wire [15:0] total_blocks_m = (out_features + M - 1) >> $clog2(M);  // Weight block size M
  wire [15:0] total_k = in_features > N + M + 2 ? in_features : N + M + 2;
  reg [15:0] current_block_n;
  reg [15:0] current_block_m;
  reg [15:0] previous_block_n;
  reg [15:0] previous_block_m;
  reg [23:0] current_k;
  reg [$clog2(QueryInputCycle + 1):0] query_cycle;  // Counts cycles after issuing queries
  always @(posedge clk or negedge rst) begin
    if (!rst || clear) begin
      // Initial State: Querying X_blocks[0, 0] and W_blocks[0, 0] from SRAM
      current_block_n <= 0;
      current_block_m <= 0;
      previous_block_n <= 0;
      previous_block_m <= 0;
      current_k <= 0;
      query_n <= 0;
      query_k <= 0;
      query_m <= 0;
      query_en <= 1;
      submit_en <= 0;
      submit_n <= 0;
      submit_m <= 0;
      done <= 0;
      query_cycle <= QueryInputCycle + 1;  // 1 extra pipeline stage for dequantization of inputs
    end else if (en) begin
      // Querying:
      if (query_k != total_k - 1) begin
        query_k <= query_k + 1;
        if (query_k == in_features - 1) begin
          query_en <= 0;  // Stop issuing queries after the last column of W^T and X is queried
        end
      end else if (query_n != total_blocks_n - 1) begin
        query_k  <= 0;
        query_n  <= query_n + 1;
        query_en <= 1;
      end else if (query_m != total_blocks_m - 1) begin
        query_k  <= 0;
        query_m  <= query_m + 1;
        query_n  <= 0;
        query_en <= 1;
      end else begin
        query_en <= 0;
      end

      // Computation:
      if (current_block_m == total_blocks_m && current_block_n == 0 && current_k == N + M + 2) begin
        // Done issue after the last block of outputs is submitted
        done <= 1;
        submit_en <= 0;
      end else if (query_cycle > 0) begin
        // We haven't seen the first results yet
        // In such case, we should continue issuing queries but not advance the state
        query_cycle <= query_cycle - 1;
      end else if (current_k != total_k - 1) begin
        // current_m, current_n, and current_k are the indices currently ready for computation
        // We should advance the state for the next cycle of computation
        current_k <= current_k + 1;
        if ((current_k >= N + 2 && current_k <= N + M + 1)
          && (current_block_m != 0 || current_block_n != 0)) begin
          submit_en <= 1;
          submit_n  <= previous_block_n;
          submit_m  <= (previous_block_m << $clog2(M)) + ((current_k - N - 2) & (M - 1));
        end else begin
          submit_en <= 0;
        end
      end else if (current_block_n != total_blocks_n - 1) begin
        current_k <= 0;
        current_block_n <= current_block_n + 1;
        previous_block_n <= current_block_n;
        previous_block_m <= current_block_m;
        submit_en <= 0;
      end else if (current_block_m != total_blocks_m) begin
        // Go for one more block to submit the last few outputs
        current_k <= 0;
        current_block_m <= current_block_m + 1;
        current_block_n <= 0;
        previous_block_m <= current_block_m;
        previous_block_n <= current_block_n;
        submit_en <= 0;
      end
    end
  end

  // Output quant
  wire [N*M*32-1:0] y_fp32_ftz;  // N x M x fp32, in ftz format
  generate
    for (i = 0; i < N; i = i + 1) begin : g_y_quant
      // 2 level pipeline summing up 2 round outputs and converting to fp16
      reg [31:0] y_fp32_item_previous;
      always @(posedge clk) begin
        y_fp32_item_previous <= y_fp32_ftz >> ((i * M + ((current_k - i - 1) & (M - 1))) * 32);
      end
      wire [31:0] y_fp32_item = y_fp32_ftz >> ((i * M + ((current_k - i - 2) & (M - 1))) * 32);
      wire y_fp32_sum_add_denormalize_sign_out;
      wire [8:0] y_fp32_sum_add_denormalize_exp_out;
      wire [25:0] y_fp32_sum_add_denormalize_mant_out;
      reg y_fp32_sum_normalize_fp16_sign_in;
      reg [8:0] y_fp32_sum_normalize_fp16_exp_in;
      reg [25:0] y_fp32_sum_normalize_fp16_mant_in;
      wire [15:0] y_fp16_item;
      add_denormalize y_fp32_sum_add_denormalize_inst (
          .a(y_fp32_item_previous),
          .b(y_fp32_item),
          .sign(y_fp32_sum_add_denormalize_sign_out),
          .exp(y_fp32_sum_add_denormalize_exp_out),
          .mant(y_fp32_sum_add_denormalize_mant_out)
      );
      normalize_fp16 y_fp32_sum_normalize_fp16_inst (
          .sign(y_fp32_sum_normalize_fp16_sign_in),
          .exp (y_fp32_sum_normalize_fp16_exp_in),
          .mant(y_fp32_sum_normalize_fp16_mant_in),
          .fp16(y_fp16_item)
      );
      always @(posedge clk or negedge rst) begin
        if (!rst) begin
          y_fp32_sum_normalize_fp16_sign_in <= 0;
          y_fp32_sum_normalize_fp16_exp_in  <= 0;
          y_fp32_sum_normalize_fp16_mant_in <= 0;
        end else if (en) begin
          y_fp32_sum_normalize_fp16_sign_in <= y_fp32_sum_add_denormalize_sign_out;
          y_fp32_sum_normalize_fp16_exp_in  <= y_fp32_sum_add_denormalize_exp_out;
          y_fp32_sum_normalize_fp16_mant_in <= y_fp32_sum_add_denormalize_mant_out;
        end
      end

      // Skew the output for the i-th row by i cycles
      if (i == N - 1) begin : g_last_row
        always @(posedge clk or negedge rst) begin
          if (!rst) begin
            y_data[(i+1)*16-1:i*16] <= 0;
          end else if (en) begin
            y_data[(i+1)*16-1:i*16] <= y_fp16_item;
          end
        end
      end else begin : g_other_rows
        reg [15:0] y_skew[0:N-2-i];
        always @(posedge clk or negedge rst) begin
          if (!rst) begin
            y_data[(i+1)*16-1:i*16] <= 0;
          end else if (en) begin
            y_data[(i+1)*16-1:i*16] <= y_skew[0];
          end
        end
        for (j = 0; j < N - 2 - i; j = j + 1) begin : g_skew_y
          always @(posedge clk or negedge rst) begin
            if (!rst) begin
              y_skew[j] <= 0;
            end else if (en) begin
              y_skew[j] <= y_skew[j+1];
            end
          end
        end
        always @(posedge clk or negedge rst) begin
          if (!rst) begin
            y_skew[N-2-i] <= 0;
          end else if (en) begin
            y_skew[N-2-i] <= y_fp16_item;
          end
        end
      end
    end
  endgenerate

  // Systolic Array
  wire [N*(ActivationWidth+1)-1:0] x_flat;
  wire [M*(ActivationWidth+1)-1:0] w_flat;
  generate
    for (i = 0; i < N; i = i + 1) begin : g_x_flat
      assign x_flat[(i+1)*(ActivationWidth+1)-1:i*(ActivationWidth+1)] = x_dequant_ext[i];
    end
    for (i = 0; i < M; i = i + 1) begin : g_w_flat
      assign w_flat[(i+1)*(ActivationWidth+1)-1:i*(ActivationWidth+1)] = w_dequant_ext[i];
    end
  endgenerate
  systolic_array #(
      .N(N),
      .M(M),
      .ActivationWidth(ActivationWidth)
  ) systolic_array_inst (
      .clk(clk),
      .rst(rst),
      .en(en && query_cycle == 0),  // Enable the systolic array only when the first row is ready
      .clear(clear || current_k == 0),
      .width(tile_width),
      .a(current_k >= in_features ? 0 : x_flat),
      .b(current_k >= in_features ? 0 : w_flat),
      .c(y_fp32_ftz)
  );
endmodule
