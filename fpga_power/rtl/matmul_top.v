`ifndef MAX_BATCH_SIZE
`define MAX_BATCH_SIZE 24
`endif

`ifndef ACTIVATION_WIDTH
`define ACTIVATION_WIDTH 16
`endif

`ifndef WEIGHT_QUANTIZED
`define WEIGHT_QUANTIZED 1
`endif

module matmul_top #(
  parameter integer MaxBatchSize = `MAX_BATCH_SIZE,
  parameter integer ActivationWidth = `ACTIVATION_WIDTH,
  parameter integer WeightQuantized = `WEIGHT_QUANTIZED
) (
    input wire clk,
    input wire rst,   // negative reset
    input wire en,    // If the computation should work, after loading and before reading results
    input wire clear, // Clear the internal state of the computation module

    input wire [$clog2(MaxBatchSize)-1:0] batch_size,  // T dimension, max MaxBatchSize
    input wire [1:0] w_type,  // [1]: bitwidth (0:8bit, 1:4bit), [0]: asymmetric (0:symmetric, 1:asymmetric)

    // Flow control
    input wire w_write_en,
    input wire y_read_en,

    // Weight streaming interface, w_scales, w_zeros, w_quants
    // The size of the tile is 16 x 4096
    // Every tile starts with 64 lines of w_scales.
    // Quant lines depend on bitwidth: 8bit -> 2048 lines, 4bit -> 1024 lines.
    // If asymmetric quantization is used, zero lines are inserted between scale and quant:
    // 8bit -> 32 lines, 4bit -> 16 lines.
    // If a buffer has been filled, the w_ready signal will be set to 0, preventing further writes
    // until the computation module has consumed the data and swap the buffers.
    // If the buffer is not full when computation module needs to swap, it is starvation and
    // the computation module have to wait until w_ready is 1 again.
    // In the whole computation process, 1376 tiles will be consumed in total.
    input wire w_valid,  // Indicates in_bus should be written to the buffer
    output wire w_ready,  // Indicates a buffer can be written

    // Shared control/input bus (no conflict expected in system flow):
    // [63:0]   : x_data
    // [79:64]  : x_n
    // [95:80]  : x_k
    // [111:96] : y_n
    // [127:112]: y_m
    // [127:0] is also used by weight stream payload when w_valid=1
    // Weight stream protocol (DDR-style):
    //   negedge clk: in_bus carries weight upper 128 bits [255:128]
    //   posedge clk: in_bus carries weight lower 128 bits [127:0]
    input wire [127:0] in_bus,

    // Y output data bus (4 fp16 in one 64-bit word)
    output wire [63:0] out_bus,

    // Control and status
    output wire done
);
  localparam [23:0] InFeatures = 24'd4096;
  localparam [15:0] OutFeatures = 16'd22016;
  localparam [15:0] TileOutFeatures = 16'd16;
  localparam [15:0] TotalTilesM = 16'd1376;
  localparam integer N = 4;
  localparam integer M = 16;
  localparam integer XDataWidth = N * ActivationWidth;
  localparam integer WScaleWidth = M * ActivationWidth;
  localparam integer WHalfWidth = M * (ActivationWidth / 2);
  localparam integer WBufferDataWidth = M * ActivationWidth;
  localparam integer WBufferHalfWidth = WBufferDataWidth / 2;
  localparam QuantEnabled = (WeightQuantized != 0) ? 1'b1 : 1'b0;
  localparam integer WScaleDataCount = QuantEnabled ? 64 : 4096;
  localparam integer WScaleAddrWidth = $clog2(WScaleDataCount);
  localparam integer XDataCount = 1024 * MaxBatchSize;
  localparam integer YDataCount = 5504 * MaxBatchSize;

  localparam [12:0] TileLinesScaleQuant = 13'd64;
  localparam [12:0] TileLinesScaleFull = 13'd4096;
  localparam [12:0] TileLinesQuant8 = 13'd2048;
  localparam [12:0] TileLinesQuant4 = 13'd1024;
  localparam [12:0] TileLinesZero8 = 13'd32;
  localparam [12:0] TileLinesZero4 = 13'd16;

  wire [15:0] batch_size_full = {{(16-$clog2(MaxBatchSize)){1'b0}}, batch_size};
  wire [XDataWidth-1:0] x_data_packed = in_bus[XDataWidth-1:0];
  wire [15:0] x_n_packed = in_bus[79:64];
  wire [15:0] x_k_packed = in_bus[95:80];
  wire [15:0] y_n_packed = in_bus[111:96];
  wire [15:0] y_m_packed = in_bus[127:112];

  function [127:0] expand_nibbles_to_bytes;
    input [63:0] in64;
    integer idx;
    reg [127:0] out128;
    begin
      out128 = 128'd0;
      for (idx = 0; idx < 16; idx = idx + 1) begin
        out128[idx*8+:8] = {4'd0, in64[idx*4+:4]};
      end
      expand_nibbles_to_bytes = out128;
    end
  endfunction

  reg [15:0] write_tile;
  reg [12:0] write_head[0:1];
  reg last_query_m;

  wire [1:0] compute_dtype = QuantEnabled ? w_type : 2'b00;

  wire [15:0] x_query_addr = gemm_query_n * InFeatures[15:0] + gemm_query_k[15:0];
  wire [18:0] y_submit_addr_full = gemm_submit_n * OutFeatures + gemm_submit_m;
  wire [18:0] y_read_addr_full = y_n_packed * OutFeatures + y_m_packed;

  wire query_buf_sel = gemm_query_m[0];
  wire write_buf_sel = write_tile[0];
  wire gemm_step_en = en && ((write_head[gemm_query_m[0]] == fill_total_lines) || !gemm_query_en);
  wire query_tile_advanced = gemm_step_en && gemm_query_en &&
                             (gemm_query_m[0] != last_query_m);

  wire fill_is_4bit = w_type[1];
  wire fill_is_asym = w_type[0];
  wire [12:0] fill_scale_lines = QuantEnabled ? TileLinesScaleQuant : TileLinesScaleFull;
  wire [12:0] fill_quant_lines = !QuantEnabled ? 13'd0 :
                                 (fill_is_4bit ? (ActivationWidth == 8 ? TileLinesQuant8 : TileLinesQuant4)
                                               : TileLinesQuant8);
  wire [12:0] fill_zero_lines = (!QuantEnabled || !fill_is_asym) ? 13'd0 :
                                (fill_is_4bit ? (ActivationWidth == 8 ? TileLinesZero8 : TileLinesZero4)
                                              : TileLinesZero8);
  wire [12:0] fill_total_lines = fill_scale_lines + fill_zero_lines + fill_quant_lines;
  wire has_more_tiles_to_fill = write_tile < TotalTilesM;
  wire fill_accept = w_valid && w_ready;
  wire [WBufferDataWidth-1:0] fill_data = {
      w_upper_half[WBufferHalfWidth-1:0],
      in_bus[WBufferHalfWidth-1:0]
  };

  wire [12:0] fill_quant_base = fill_scale_lines + fill_zero_lines;
  wire [WScaleAddrWidth-1:0] fill_scale_addr = write_head[write_buf_sel][WScaleAddrWidth-1:0];
  wire [15:0] fill_zero_addr = write_head[write_buf_sel] - fill_scale_lines;
  wire [15:0] fill_quant_addr = write_head[write_buf_sel] - fill_quant_base;
  wire fill_write_scale = write_head[write_buf_sel] < fill_scale_lines;
  wire fill_write_zero = QuantEnabled && fill_is_asym &&
                         (write_head[write_buf_sel] < fill_quant_base) && !fill_write_scale;

  wire [WScaleAddrWidth-1:0] query_scale_addr =
      QuantEnabled ? gemm_query_k[11:6] : gemm_query_k[11:0];
  wire [15:0] query_zero_addr = !compute_dtype[0] ? 16'd0 :
                                (compute_dtype[1] ? (ActivationWidth == 8 ? gemm_query_k[11:7] : gemm_query_k[11:8])
                                                  : gemm_query_k[11:7]);
  wire [15:0] query_quant_addr = compute_dtype[1] ? (ActivationWidth == 8 ? gemm_query_k[11:1]
                                                                           : gemm_query_k[11:2])
                                                   : gemm_query_k[11:1];

  reg x_sram_en;
  reg x_sram_we;
  reg [15:0] x_sram_addr;
  reg [XDataWidth-1:0] x_sram_din;
  wire [XDataWidth-1:0] x_sram_dout;

  reg y_sram_en;
  reg y_sram_we;
  reg [18:0] y_sram_addr;
  reg [63:0] y_sram_din;
  wire [63:0] y_sram_dout;

  reg ws_en[0:1];
  reg ws_we[0:1];
  reg [WScaleAddrWidth-1:0] ws_addr[0:1];
  reg [WScaleWidth-1:0] ws_din[0:1];
  wire [WScaleWidth-1:0] ws_dout[0:1];

  reg wz_en[0:1];
  reg wz_we[0:1];
  reg [4:0] wz_addr[0:1];
  reg [WBufferDataWidth-1:0] wz_din[0:1];
  wire [WBufferDataWidth-1:0] wz_dout[0:1];

  reg wq_en[0:1];
  reg wq_we[0:1];
  reg [10:0] wq_addr[0:1];
  reg [WBufferDataWidth-1:0] wq_din[0:1];
  wire [WBufferDataWidth-1:0] wq_dout[0:1];

  wire [15:0] gemm_query_n;
  wire [15:0] gemm_query_m;
  wire [23:0] gemm_query_k;
  wire [15:0] gemm_submit_n;
  wire [15:0] gemm_submit_m;
  wire gemm_query_en;
  wire gemm_submit_en;
  wire gemm_done;
  wire [63:0] gemm_y_data;

  reg [WHalfWidth-1:0] gemm_w_quant_half;
  reg [WHalfWidth-1:0] gemm_w_zero_half;
  reg [WScaleWidth-1:0] gemm_w_scale;
  reg [127:0] w_upper_half;
  reg [23:0] gemm_query_k_d1;
  reg [15:0] gemm_query_m_d1;

  reg [XDataWidth-1:0] gemm_x_data_r;
  reg [WHalfWidth-1:0] gemm_w_quant_half_r;
  reg [WHalfWidth-1:0] gemm_w_zero_half_r;
  reg [WScaleWidth-1:0] gemm_w_scale_r;

  wire query_buf_sel_unpack = gemm_query_m_d1[0];
  wire [WScaleWidth-1:0] compute_w_scale = ws_dout[query_buf_sel_unpack];
  wire [WBufferDataWidth-1:0] compute_w_zero = wz_dout[query_buf_sel_unpack];
  wire [WBufferDataWidth-1:0] compute_w_quant = wq_dout[query_buf_sel_unpack];
  integer arb_i;

  assign done = gemm_done;
  assign out_bus = y_sram_dout;
  assign w_ready = has_more_tiles_to_fill && (write_head[write_buf_sel] < fill_total_lines);

  // Activation SRAM
  always @(*) begin
    x_sram_en   = 1'b0;
    x_sram_we   = 1'b0;
    x_sram_addr = 16'd0;
    x_sram_din  = {XDataWidth{1'b0}};
    if (w_write_en && !en) begin
      x_sram_en   = 1'b1;
      x_sram_we   = 1'b1;
      x_sram_addr = x_n_packed * InFeatures[15:0] + x_k_packed;
      x_sram_din  = x_data_packed;
    end else if (gemm_step_en && gemm_query_en) begin
      x_sram_en   = 1'b1;
      x_sram_addr = x_query_addr;
    end
  end

  // Output SRAM
  always @(*) begin
    y_sram_en   = 1'b0;
    y_sram_we   = 1'b0;
    y_sram_addr = 19'd0;
    y_sram_din  = 64'd0;
    if (gemm_submit_en && gemm_step_en) begin
      y_sram_en   = 1'b1;
      y_sram_we   = 1'b1;
      y_sram_addr = y_submit_addr_full;
      y_sram_din  = gemm_y_data;
    end else if (y_read_en) begin
      y_sram_en   = 1'b1;
      y_sram_we   = 1'b0;
      y_sram_addr = y_read_addr_full;
    end
  end

  // Weight SRAM arbitration
  always @(*) begin
    for (arb_i = 0; arb_i < 2; arb_i = arb_i + 1) begin
      ws_en[arb_i]   = 1'b0;
      ws_we[arb_i]   = 1'b0;
      ws_addr[arb_i] = {WScaleAddrWidth{1'b0}};
      ws_din[arb_i]  = {WScaleWidth{1'b0}};

      wz_en[arb_i]   = 1'b0;
      wz_we[arb_i]   = 1'b0;
      wz_addr[arb_i] = 5'd0;
      wz_din[arb_i]  = {WBufferDataWidth{1'b0}};

      wq_en[arb_i]   = 1'b0;
      wq_we[arb_i]   = 1'b0;
      wq_addr[arb_i] = 11'd0;
      wq_din[arb_i]  = {WBufferDataWidth{1'b0}};
    end

    if (gemm_step_en && gemm_query_en) begin
      ws_en[query_buf_sel]   = 1'b1;
      ws_addr[query_buf_sel] = query_scale_addr;
      if (QuantEnabled) begin
        wz_en[query_buf_sel]   = 1'b1;
        wz_addr[query_buf_sel] = query_zero_addr[4:0];
        wq_en[query_buf_sel]   = 1'b1;
        wq_addr[query_buf_sel] = query_quant_addr[10:0];
      end
    end

    if (fill_accept) begin
      if (fill_write_scale) begin
        ws_en[write_buf_sel]   = 1'b1;
        ws_we[write_buf_sel]   = 1'b1;
        ws_addr[write_buf_sel] = fill_scale_addr;
        ws_din[write_buf_sel]  = fill_data[WScaleWidth-1:0];
      end else if (fill_write_zero) begin
        wz_en[write_buf_sel]   = 1'b1;
        wz_we[write_buf_sel]   = 1'b1;
        wz_addr[write_buf_sel] = fill_zero_addr[4:0];
        wz_din[write_buf_sel]  = fill_data;
      end else if (QuantEnabled) begin
        wq_en[write_buf_sel]   = 1'b1;
        wq_we[write_buf_sel]   = 1'b1;
        wq_addr[write_buf_sel] = fill_quant_addr[10:0];
        wq_din[write_buf_sel]  = fill_data;
      end
    end
  end

  always @(negedge clk or negedge rst) begin
    if (!rst) begin
      w_upper_half <= 128'd0;
    end else if (clear) begin
      w_upper_half <= 128'd0;
    end else if (w_valid && w_ready) begin
      w_upper_half <= in_bus;
    end
  end

  // Weight unpack for GEMM
  always @(*) begin
    gemm_w_scale = compute_w_scale;
    gemm_w_quant_half = {WHalfWidth{1'b0}};
    gemm_w_zero_half = {WHalfWidth{1'b0}};
    if (!QuantEnabled) begin
      gemm_w_quant_half = {WHalfWidth{1'b0}};
      gemm_w_zero_half = {WHalfWidth{1'b0}};
    end else if (ActivationWidth == 16) begin
      if (compute_dtype[1]) begin
        case (gemm_query_k_d1[7:6])
          2'd0: gemm_w_zero_half = expand_nibbles_to_bytes(compute_w_zero[63:0]);
          2'd1: gemm_w_zero_half = expand_nibbles_to_bytes(compute_w_zero[127:64]);
          2'd2: gemm_w_zero_half = expand_nibbles_to_bytes(compute_w_zero[191:128]);
          default: gemm_w_zero_half = expand_nibbles_to_bytes(compute_w_zero[255:192]);
        endcase
        case (gemm_query_k_d1[1:0])
          2'd0: begin
            gemm_w_quant_half = expand_nibbles_to_bytes(compute_w_quant[63:0]);
          end
          2'd1: begin
            gemm_w_quant_half = expand_nibbles_to_bytes(compute_w_quant[127:64]);
          end
          2'd2: begin
            gemm_w_quant_half = expand_nibbles_to_bytes(compute_w_quant[191:128]);
          end
          default: begin
            gemm_w_quant_half = expand_nibbles_to_bytes(compute_w_quant[255:192]);
          end
        endcase
      end else begin
        if (gemm_query_k_d1[0]) begin
          gemm_w_quant_half = compute_w_quant[255:128];
        end else begin
          gemm_w_quant_half = compute_w_quant[127:0];
        end

        if (gemm_query_k_d1[6]) begin
          gemm_w_zero_half = compute_w_zero[255:128];
        end else begin
          gemm_w_zero_half = compute_w_zero[127:0];
        end
      end
    end else begin
      if (compute_dtype[1]) begin
        if (gemm_query_k_d1[0]) begin
          gemm_w_quant_half = compute_w_quant[WBufferDataWidth-1:WBufferHalfWidth];
        end else begin
          gemm_w_quant_half = compute_w_quant[WBufferHalfWidth-1:0];
        end

        if (gemm_query_k_d1[6]) begin
          gemm_w_zero_half = compute_w_zero[WBufferDataWidth-1:WBufferHalfWidth];
        end else begin
          gemm_w_zero_half = compute_w_zero[WBufferHalfWidth-1:0];
        end
      end
    end
  end

  always @(posedge clk) begin
    if (gemm_step_en) begin
      gemm_x_data_r <= x_sram_dout;
      gemm_w_quant_half_r <= gemm_w_quant_half;
      gemm_w_zero_half_r <= gemm_w_zero_half;
      gemm_w_scale_r <= gemm_w_scale;
    end
  end

  always @(posedge clk or negedge rst) begin
    if (!rst) begin
      write_tile <= 16'd0;
      write_head[0] <= 13'd0;
      write_head[1] <= 13'd0;
      last_query_m <= 16'd0;
      gemm_query_k_d1 <= 24'd0;
      gemm_query_m_d1 <= 16'd0;
    end else begin
      if (clear) begin
        write_tile <= 16'd0;
        write_head[0] <= 13'd0;
        write_head[1] <= 13'd0;
        last_query_m <= 16'd0;
        gemm_query_k_d1 <= 24'd0;
        gemm_query_m_d1 <= 16'd0;
      end else begin
        if (query_tile_advanced) begin
          write_head[last_query_m] <= 13'd0;
        end

        if (gemm_step_en && gemm_query_en) begin
          last_query_m <= gemm_query_m[0];
          gemm_query_k_d1 <= gemm_query_k;
          gemm_query_m_d1 <= gemm_query_m;
        end

        if (fill_accept) begin
          if (write_head[write_buf_sel] == fill_total_lines - 1) begin
            write_head[write_buf_sel] <= fill_total_lines;
            write_tile <= write_tile + 16'd1;
          end else begin
            write_head[write_buf_sel] <= write_head[write_buf_sel] + 13'd1;
          end
        end
      end
    end
  end

  // Activation SRAMs

  sram #(  // X shape [T, 4096], x_addr = n * 4096 + k. T <= MaxBatchSize.
      .DataWidth(XDataWidth),
      .DataCount(XDataCount)
  ) sram_x (
      .clk (clk),
      .en  (x_sram_en),
      .we  (x_sram_we),
      .addr(x_sram_addr),
      .din (x_sram_din),
      .dout(x_sram_dout)
  );

  sram #(  // Y shape [T, 22016], y_addr = n * 22016 + m. T <= MaxBatchSize.
      .DataWidth(64),
      .DataCount(YDataCount)
  ) sram_y (
      .clk (clk),
      .en  (y_sram_en),
      .we  (y_sram_we),
      .addr(y_sram_addr),
      .din (y_sram_din),
      .dout(y_sram_dout)
  );

  // Weight SRAMs, A/B buffering, the bus width is all 256-bit, to make streaming input easier,
  // and for the quant and zero buffer, we select higher 128 bits / lower 128 bits (8-bit quantization),
  // or higher 64 bits / middle-high 64 bits / middle-low 64 bits / lower 64 bits (4-bit quantization)
  // to feed the computation module
  sram #(  // w_scale_a shape [16, 64]
      .DataWidth(WScaleWidth),
      .DataCount(WScaleDataCount)
  ) sram_w_scale_a (
      .clk (clk),
      .en  (ws_en[0]),
      .we  (ws_we[0]),
      .addr(ws_addr[0]),
      .din (ws_din[0]),
      .dout(ws_dout[0])
  );

  sram #(  // w_scale_b shape [16, 64]
      .DataWidth(WScaleWidth),
      .DataCount(WScaleDataCount)
  ) sram_w_scale_b (
      .clk (clk),
      .en  (ws_en[1]),
      .we  (ws_we[1]),
      .addr(ws_addr[1]),
      .din (ws_din[1]),
      .dout(ws_dout[1])
  );
    generate
    if (QuantEnabled) begin : g_quant_sram
      sram #(  // w_zero_a shape [16, 32]
        .DataWidth(WBufferDataWidth),
        .DataCount(32)
      ) sram_w_zero_a (
        .clk (clk),
        .en  (wz_en[0]),
        .we  (wz_we[0]),
        .addr(wz_addr[0]),
        .din (wz_din[0]),
        .dout(wz_dout[0])
      );
      sram #(  // w_quant_a shape [16, 4096]
        .DataWidth(WBufferDataWidth),
        .DataCount(2048)
      ) sram_w_quant_a (
        .clk (clk),
        .en  (wq_en[0]),
        .we  (wq_we[0]),
        .addr(wq_addr[0]),
        .din (wq_din[0]),
        .dout(wq_dout[0])
      );

      sram #(  // w_zero_b shape [16, 32]
        .DataWidth(WBufferDataWidth),
        .DataCount(32)
      ) sram_w_zero_b (
        .clk (clk),
        .en  (wz_en[1]),
        .we  (wz_we[1]),
        .addr(wz_addr[1]),
        .din (wz_din[1]),
        .dout(wz_dout[1])
      );
      sram #(  // w_quant_b shape [16, 4096]
        .DataWidth(WBufferDataWidth),
        .DataCount(2048)
      ) sram_w_quant_b (
        .clk (clk),
        .en  (wq_en[1]),
        .we  (wq_we[1]),
        .addr(wq_addr[1]),
        .din (wq_din[1]),
        .dout(wq_dout[1])
      );
    end else begin : g_no_quant_sram
      assign wz_dout[0] = {WBufferDataWidth{1'b0}};
      assign wz_dout[1] = {WBufferDataWidth{1'b0}};
      assign wq_dout[0] = {WBufferDataWidth{1'b0}};
      assign wq_dout[1] = {WBufferDataWidth{1'b0}};
    end
    endgenerate

  // Computation module
  gemm #(
    .ActivationWidth(ActivationWidth),
    .E4M3Enabled(WeightQuantized),
    .Int8Enabled(WeightQuantized),
    .E2M1Enabled(WeightQuantized),
    .Int4Enabled(WeightQuantized),
    .QueryInputCycle(2)
  ) gemm_inst (
    .clk(clk),
    .rst(rst),
    .en(gemm_step_en),
    .clear(clear),
    .batch_size(batch_size_full),
    .in_features(InFeatures),
    .out_features(OutFeatures),
    .x_data(gemm_x_data_r),
    .w_type(compute_dtype),
    .w_quant_half(gemm_w_quant_half_r),
    .w_zero_half(gemm_w_zero_half_r),
    .w_scale_uext_ftz(gemm_w_scale_r),
    .y_data(gemm_y_data),
    .query_n(gemm_query_n),
    .query_m(gemm_query_m),
    .query_k(gemm_query_k),
    .submit_n(gemm_submit_n),
    .submit_m(gemm_submit_m),
    .query_en(gemm_query_en),
    .submit_en(gemm_submit_en),
    .done(gemm_done)
  );
endmodule
