`timescale 1ns / 1ps

module tb_matmul_top_wnam;

  localparam integer K = 4096;
  localparam integer OUT_FEATURES = 22016;
  localparam integer MAX_BATCH_SIZE = 24;
  localparam integer TOTAL_TILES = 1376;

`ifdef CASE_WE5M10_AE5M10
  localparam integer ACTIVATION_WIDTH = 16;
  localparam integer WEIGHT_QUANTIZED = 0;
  localparam integer WEIGHT_BITS = 16;
  localparam integer WEIGHT_IS_INT = 0;
  localparam [1:0] W_TYPE_SEL = 2'b00;
  localparam [8*32-1:0] EXPECTED_ACT_DTYPE = "e5m10";
  localparam [8*32-1:0] EXPECTED_WEIGHT_DTYPE = "e5m10";
  localparam [8*256-1:0] TEST_FILE = "c/test_we5m10_ae5m10.txt";
`elsif CASE_WINT8_AE5M10
  localparam integer ACTIVATION_WIDTH = 16;
  localparam integer WEIGHT_QUANTIZED = 1;
  localparam integer WEIGHT_BITS = 8;
  localparam integer WEIGHT_IS_INT = 1;
  localparam [1:0] W_TYPE_SEL = 2'b01;
  localparam [8*32-1:0] EXPECTED_ACT_DTYPE = "e5m10";
  localparam [8*32-1:0] EXPECTED_WEIGHT_DTYPE = "int8";
  localparam [8*256-1:0] TEST_FILE = "c/test_wint8_ae5m10.txt";
`elsif CASE_WE4M3_AE5M10
  localparam integer ACTIVATION_WIDTH = 16;
  localparam integer WEIGHT_QUANTIZED = 1;
  localparam integer WEIGHT_BITS = 8;
  localparam integer WEIGHT_IS_INT = 0;
  localparam [1:0] W_TYPE_SEL = 2'b00;
  localparam [8*32-1:0] EXPECTED_ACT_DTYPE = "e5m10";
  localparam [8*32-1:0] EXPECTED_WEIGHT_DTYPE = "e4m3";
  localparam [8*256-1:0] TEST_FILE = "c/test_we4m3_ae5m10.txt";
`elsif CASE_WINT4_AE5M10
  localparam integer ACTIVATION_WIDTH = 16;
  localparam integer WEIGHT_QUANTIZED = 1;
  localparam integer WEIGHT_BITS = 4;
  localparam integer WEIGHT_IS_INT = 1;
  localparam [1:0] W_TYPE_SEL = 2'b11;
  localparam [8*32-1:0] EXPECTED_ACT_DTYPE = "e5m10";
  localparam [8*32-1:0] EXPECTED_WEIGHT_DTYPE = "int4";
  localparam [8*256-1:0] TEST_FILE = "c/test_wint4_ae5m10.txt";
`elsif CASE_WE2M1_AE5M10
  localparam integer ACTIVATION_WIDTH = 16;
  localparam integer WEIGHT_QUANTIZED = 1;
  localparam integer WEIGHT_BITS = 4;
  localparam integer WEIGHT_IS_INT = 0;
  localparam [1:0] W_TYPE_SEL = 2'b10;
  localparam [8*32-1:0] EXPECTED_ACT_DTYPE = "e5m10";
  localparam [8*32-1:0] EXPECTED_WEIGHT_DTYPE = "e2m1";
  localparam [8*256-1:0] TEST_FILE = "c/test_we2m1_ae5m10.txt";
`elsif CASE_WE4M3_AE4M3
  localparam integer ACTIVATION_WIDTH = 8;
  localparam integer WEIGHT_QUANTIZED = 0;
  localparam integer WEIGHT_BITS = 8;
  localparam integer WEIGHT_IS_INT = 0;
  localparam [1:0] W_TYPE_SEL = 2'b00;
  localparam [8*32-1:0] EXPECTED_ACT_DTYPE = "e4m3";
  localparam [8*32-1:0] EXPECTED_WEIGHT_DTYPE = "e4m3";
  localparam [8*256-1:0] TEST_FILE = "c/test_we4m3_ae4m3.txt";
`elsif CASE_WINT4_AE4M3
  localparam integer ACTIVATION_WIDTH = 8;
  localparam integer WEIGHT_QUANTIZED = 1;
  localparam integer WEIGHT_BITS = 4;
  localparam integer WEIGHT_IS_INT = 1;
  localparam [1:0] W_TYPE_SEL = 2'b11;
  localparam [8*32-1:0] EXPECTED_ACT_DTYPE = "e4m3";
  localparam [8*32-1:0] EXPECTED_WEIGHT_DTYPE = "int4";
  localparam [8*256-1:0] TEST_FILE = "c/test_wint4_ae4m3.txt";
`elsif CASE_WE2M1_AE4M3
  localparam integer ACTIVATION_WIDTH = 8;
  localparam integer WEIGHT_QUANTIZED = 1;
  localparam integer WEIGHT_BITS = 4;
  localparam integer WEIGHT_IS_INT = 0;
  localparam [1:0] W_TYPE_SEL = 2'b10;
  localparam [8*32-1:0] EXPECTED_ACT_DTYPE = "e4m3";
  localparam [8*32-1:0] EXPECTED_WEIGHT_DTYPE = "e2m1";
  localparam [8*256-1:0] TEST_FILE = "c/test_we2m1_ae4m3.txt";
`else
  localparam integer ACTIVATION_WIDTH = 16;
  localparam integer WEIGHT_QUANTIZED = 1;
  localparam integer WEIGHT_BITS = 4;
  localparam integer WEIGHT_IS_INT = 1;
  localparam [1:0] W_TYPE_SEL = 2'b11;
  localparam [8*32-1:0] EXPECTED_ACT_DTYPE = "e5m10";
  localparam [8*32-1:0] EXPECTED_WEIGHT_DTYPE = "int4";
  localparam [8*256-1:0] TEST_FILE = "c/test_wint4_ae5m10.txt";
`endif

`ifdef BATCH1
  localparam integer BATCH_USED = 1;
`else
  localparam integer BATCH_USED = 24;
`endif

  localparam integer PACK_ROWS = (BATCH_USED + 3) / 4;
  localparam integer XDATA_WIDTH = 4 * ACTIVATION_WIDTH;
  localparam integer WBUF_WIDTH = 16 * ACTIVATION_WIDTH;
  localparam integer WBUF_HALF = WBUF_WIDTH / 2;
  localparam integer COMPARE_ROWS = (BATCH_USED == 1) ? 1 : 24;
  localparam Q_EN = (WEIGHT_QUANTIZED != 0) ? 1'b1 : 1'b0;
  localparam W_INT = (WEIGHT_IS_INT != 0) ? 1'b1 : 1'b0;

  integer fd;
  integer rc;
  integer i;
  integer j;
  integer k;
  integer tile_idx;
  integer sim_cyc;
  integer en_start_cyc;
  integer done_cyc;
  reg en_seen;
  reg done_seen;

  reg clk;
  reg rst;
  reg en;
  reg clear;

  reg [$clog2(MAX_BATCH_SIZE)-1:0] batch_size;
  reg [1:0] w_type;

  reg w_write_en;
  reg y_read_en;
  reg [127:0] in_bus;
  reg w_valid;

  wire [63:0] out_bus;
  wire w_ready;
  wire done;

  reg [8*32-1:0] tok;
  reg [8*32-1:0] tok2;
  reg [8*32-1:0] tok3;
  reg [255:0] line_buf;

  reg [15:0] xvals [0:23];
  reg [15:0] row16 [0:15];
  reg [15:0] y_ref [0:24*OUT_FEATURES-1];

  real tmp_real_a;
  real tmp_real_b;
  real tmp_real_c;

  real mae_acc;
  real mse_acc;
  real linf_abs;
  real eps;
  real max_relabs;
  integer compare_count;

  real top_score [0:15];
  integer top_n [0:15];
  integer top_m [0:15];
  reg [15:0] top_ref [0:15];
  reg [15:0] top_out [0:15];

  matmul_top #(
      .MaxBatchSize(MAX_BATCH_SIZE),
      .ActivationWidth(ACTIVATION_WIDTH),
      .WeightQuantized(WEIGHT_QUANTIZED)
  ) dut (
      .clk(clk),
      .rst(rst),
      .en(en),
      .clear(clear),
      .batch_size(batch_size),
      .w_type(w_type),
      .w_write_en(w_write_en),
      .y_read_en(y_read_en),
      .w_valid(w_valid),
      .w_ready(w_ready),
      .in_bus(in_bus),
      .out_bus(out_bus),
      .done(done)
  );

  always #2.5 clk = ~clk;

  always @(posedge clk or negedge rst) begin
    if (!rst) begin
      sim_cyc <= 0;
      en_start_cyc <= -1;
      done_cyc <= -1;
      en_seen <= 1'b0;
      done_seen <= 1'b0;
    end else begin
      sim_cyc <= sim_cyc + 1;
      if (en && !en_seen) begin
        en_seen <= 1'b1;
        en_start_cyc <= sim_cyc;
      end
      if (done && !done_seen) begin
        done_seen <= 1'b1;
        done_cyc <= sim_cyc;
      end
    end
  end

`ifdef DEBUG_WLOAD
  always @(posedge clk) begin
    if (rst && dut.fill_accept && dut.fill_write_scale && (dut.write_tile == 16'd2) && (dut.write_head[dut.write_buf_sel] < 13'd2)) begin
      $display("[DBG_WLOAD] tile=%0d head=%0d in_bus=%032h w_upper=%032h fill_data=%064h ws_din=%064h",
               dut.write_tile,
               dut.write_head[dut.write_buf_sel],
               dut.in_bus,
               dut.w_upper_half,
               dut.fill_data,
               dut.ws_din[dut.write_buf_sel]);
    end
  end
`endif

  function real abs_real;
    input real x;
    begin
      abs_real = (x < 0.0) ? -x : x;
    end
  endfunction

  function real max_real;
    input real a;
    input real b;
    begin
      max_real = (a > b) ? a : b;
    end
  endfunction

  function real fp16_to_real;
    input [15:0] h;
    integer e;
    integer m;
    real sign;
    real frac;
    begin
      sign = h[15] ? -1.0 : 1.0;
      e = h[14:10];
      m = h[9:0];
      if (e == 0) begin
        frac = $itor(m) / 1024.0;
        fp16_to_real = sign * frac * (2.0 ** -14);
      end else if (e == 31) begin
        fp16_to_real = sign * 65504.0;
      end else begin
        frac = 1.0 + ($itor(m) / 1024.0);
        fp16_to_real = sign * frac * (2.0 ** (e - 15));
      end
    end
  endfunction

  function real relative_or_absolute_error;
    input real y_out;
    input real y_ans;
    input real epsilon;
    real denom;
    begin
      denom = max_real(abs_real(y_ans), epsilon);
      relative_or_absolute_error = abs_real(y_out - y_ans) / denom;
    end
  endfunction

  task automatic pulse_x_write;
    begin
      w_write_en = 1'b1;
      @(posedge clk);
      #1;
      w_write_en = 1'b0;
    end
  endtask

  task automatic stream_weight_word;
    input [WBUF_WIDTH-1:0] word;
    reg [127:0] bus_word;
    integer wait_cyc;
    begin
      wait_cyc = 0;
      while (!w_ready && !done) begin
        @(posedge clk);
        wait_cyc = wait_cyc + 1;
      end
      if (!done) begin
        #1;
        bus_word = 128'd0;
        bus_word[WBUF_HALF-1:0] = word[WBUF_WIDTH-1:WBUF_HALF];
        in_bus = bus_word;
        w_valid = 1'b1;
        @(negedge clk);
        #1;

        bus_word = 128'd0;
        bus_word[WBUF_HALF-1:0] = word[WBUF_HALF-1:0];
        in_bus = bus_word;
        @(posedge clk);
        #1;
        w_valid = 1'b0;
      end
    end
  endtask

  task automatic stream_matrix_rows;
    input integer rows;
    input integer elem_bits;
    integer r;
    integer c;
    integer bit_pos;
    reg [WBUF_WIDTH-1:0] acc;
    begin
      acc = {WBUF_WIDTH{1'b0}};
      bit_pos = 0;
      for (r = 0; r < rows; r = r + 1) begin
        rc = $fscanf(fd,
                     "%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h",
                     row16[0], row16[1], row16[2], row16[3],
                     row16[4], row16[5], row16[6], row16[7],
                     row16[8], row16[9], row16[10], row16[11],
                     row16[12], row16[13], row16[14], row16[15]);
        if (rc != 16) begin
          $display("[TB_WNAM] parse matrix row failed rows=%0d r=%0d", rows, r);
          $finish;
        end
        for (c = 0; c < 16; c = c + 1) begin
          if (elem_bits == 16) begin
            acc[bit_pos +: 16] = row16[c];
            bit_pos = bit_pos + 16;
          end else if (elem_bits == 8) begin
            acc[bit_pos +: 8] = row16[c][7:0];
            bit_pos = bit_pos + 8;
          end else begin
            acc[bit_pos +: 4] = row16[c][3:0];
            bit_pos = bit_pos + 4;
          end

          if (bit_pos == WBUF_WIDTH) begin
            stream_weight_word(acc);
            acc = {WBUF_WIDTH{1'b0}};
            bit_pos = 0;
          end
        end
      end
      if (bit_pos != 0) begin
        $display("[TB_WNAM] matrix packing tail not empty: bit_pos=%0d", bit_pos);
        $finish;
      end
    end
  endtask

  task automatic parse_header;
    integer batch_in_file;
    begin
      rc = $fscanf(fd, "%s %s %s", tok, tok2, tok3);
      if (rc != 3 || tok != "pcg64_seed") begin
        $display("[TB_WNAM] bad header: pcg64_seed");
        $finish;
      end

      rc = $fscanf(fd, "%s %d", tok, batch_in_file);
      if (rc != 2 || tok != "batch_size" || batch_in_file != 24) begin
        $display("[TB_WNAM] bad header: batch_size");
        $finish;
      end

      rc = $fscanf(fd, "%s %s", tok, tok2);
      if (rc != 2 || tok != "input_dtype" || tok2 != EXPECTED_ACT_DTYPE) begin
        $display("[TB_WNAM] input_dtype mismatch, got=%0s expect=%0s", tok2, EXPECTED_ACT_DTYPE);
        $finish;
      end

      rc = $fscanf(fd, "%s %s", tok, tok2);
      if (rc != 2 || tok != "weight_dtype" || tok2 != EXPECTED_WEIGHT_DTYPE) begin
        $display("[TB_WNAM] weight_dtype mismatch, got=%0s expect=%0s", tok2, EXPECTED_WEIGHT_DTYPE);
        $finish;
      end

      rc = $fscanf(fd, "%s %s", tok, tok2);
      if (rc != 2 || tok != "output_dtype" || tok2 != "e5m10") begin
        $display("[TB_WNAM] output_dtype mismatch");
        $finish;
      end

      rc = $fscanf(fd, "%s %s %f %s %f %s %f",
                   tok, tok2, tmp_real_a, tok3, tmp_real_b, tok2, tmp_real_c);
      if (rc != 7 || tok != "input_activation_error") begin
        $display("[TB_WNAM] input_activation_error line parse failed");
        $finish;
      end

      rc = $fscanf(fd, "%s %s %f %s %f %s %f",
                   tok, tok2, tmp_real_a, tok3, tmp_real_b, tok2, tmp_real_c);
      if (rc != 7 || tok != "weight_error") begin
        $display("[TB_WNAM] weight_error line parse failed");
        $finish;
      end

      rc = $fscanf(fd, "%s %s %f %s %f %s %f",
                   tok, tok2, tmp_real_a, tok3, tmp_real_b, tok2, tmp_real_c);
      if (rc != 7 || tok != "output_activation_error") begin
        $display("[TB_WNAM] output_activation_error line parse failed");
        $finish;
      end

      rc = $fscanf(fd, "%s %s %s", tok, tok2, tok3);
      if (rc != 3 || tok != "input_activation" || tok2 != "column_major") begin
        $display("[TB_WNAM] input_activation header parse failed");
        $finish;
      end
    end
  endtask

  task automatic load_x_data;
    integer col;
    integer pack_idx;
    integer lane;
    reg [XDATA_WIDTH-1:0] x_word;
    begin
      for (col = 0; col < K; col = col + 1) begin
        rc = $fscanf(fd,
                     "%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h",
                     xvals[0], xvals[1], xvals[2], xvals[3],
                     xvals[4], xvals[5], xvals[6], xvals[7],
                     xvals[8], xvals[9], xvals[10], xvals[11],
                     xvals[12], xvals[13], xvals[14], xvals[15],
                     xvals[16], xvals[17], xvals[18], xvals[19],
                     xvals[20], xvals[21], xvals[22], xvals[23]);
        if (rc != 24) begin
          $display("[TB_WNAM] activation parse failed at k=%0d", col);
          $finish;
        end

        for (pack_idx = 0; pack_idx < PACK_ROWS; pack_idx = pack_idx + 1) begin
          x_word = {XDATA_WIDTH{1'b0}};
          for (lane = 0; lane < 4; lane = lane + 1) begin
            if (BATCH_USED == 1) begin
              if (lane == 0) begin
                if (ACTIVATION_WIDTH == 16) begin
                  x_word[lane*16 +: 16] = xvals[0];
                end else begin
                  x_word[lane*8 +: 8] = xvals[0][7:0];
                end
              end
            end else begin
              if (ACTIVATION_WIDTH == 16) begin
                x_word[lane*16 +: 16] = xvals[pack_idx*4 + lane];
              end else begin
                x_word[lane*8 +: 8] = xvals[pack_idx*4 + lane][7:0];
              end
            end
          end

          in_bus = 128'd0;
          in_bus[XDATA_WIDTH-1:0] = x_word;
          in_bus[79:64] = pack_idx[15:0];
          in_bus[95:80] = col[15:0];
          pulse_x_write();
        end
      end
    end
  endtask

  task automatic stream_all_weights;
    integer rows_scale;
    begin
      for (tile_idx = 0; tile_idx < TOTAL_TILES; tile_idx = tile_idx + 1) begin
        rc = $fscanf(fd, "%s %d", tok, i);
        if (rc != 2 || tok != "weight_tile" || i != tile_idx) begin
          $display("[TB_WNAM] weight_tile header parse failed at tile=%0d", tile_idx);
          $finish;
        end
        rc = $fgets(line_buf, fd);

        rows_scale = Q_EN ? 64 : 4096;
        stream_matrix_rows(rows_scale, ACTIVATION_WIDTH);
        if (Q_EN && W_INT) begin
          stream_matrix_rows(64, WEIGHT_BITS);
        end
        if (Q_EN) begin
          stream_matrix_rows(4096, WEIGHT_BITS);
        end
      end
    end
  endtask

  task automatic parse_reference_output;
    integer m;
    integer n;
    begin
      rc = $fscanf(fd, "%s %s %s", tok, tok2, tok3);
      if (rc != 3 || tok != "output_activation" || tok2 != "column_major") begin
        $display("[TB_WNAM] output_activation header parse failed");
        $finish;
      end

      for (m = 0; m < OUT_FEATURES; m = m + 1) begin
        rc = $fscanf(fd,
                     "%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h",
                     xvals[0], xvals[1], xvals[2], xvals[3],
                     xvals[4], xvals[5], xvals[6], xvals[7],
                     xvals[8], xvals[9], xvals[10], xvals[11],
                     xvals[12], xvals[13], xvals[14], xvals[15],
                     xvals[16], xvals[17], xvals[18], xvals[19],
                     xvals[20], xvals[21], xvals[22], xvals[23]);
        if (rc != 24) begin
          $display("[TB_WNAM] output_activation data parse failed at col=%0d", m);
          $finish;
        end
        for (n = 0; n < 24; n = n + 1) begin
          y_ref[n * OUT_FEATURES + m] = xvals[n];
        end
      end
    end
  endtask

  task automatic read_y_word;
    input [15:0] pack_n;
    input [15:0] m;
    output [63:0] word;
    begin
      in_bus[111:96] = pack_n;
      in_bus[127:112] = m;
      y_read_en = 1'b1;
      @(posedge clk);
      #1;
      word = out_bus;
      y_read_en = 1'b0;
    end
  endtask

  task automatic update_top16;
    input real score;
    input integer n;
    input integer m;
    input [15:0] ref_bits;
    input [15:0] out_bits;
    integer idx;
    integer min_idx;
    begin
      min_idx = 0;
      for (idx = 1; idx < 16; idx = idx + 1) begin
        if (top_score[idx] < top_score[min_idx]) begin
          min_idx = idx;
        end
      end
      if (score > top_score[min_idx]) begin
        top_score[min_idx] = score;
        top_n[min_idx] = n;
        top_m[min_idx] = m;
        top_ref[min_idx] = ref_bits;
        top_out[min_idx] = out_bits;
      end
    end
  endtask

  task automatic collect_and_compare;
    integer m;
    integer pack_n;
    integer lane;
    integer n;
    reg [63:0] word;
    reg [15:0] out_bits;
    reg [15:0] ref_bits;
    real y_out;
    real y_ans;
    real abs_err;
    real relabs_err;
    begin
      mae_acc = 0.0;
      mse_acc = 0.0;
      linf_abs = 0.0;
      max_relabs = 0.0;
      compare_count = 0;

      for (i = 0; i < 16; i = i + 1) begin
        top_score[i] = -1.0;
        top_n[i] = -1;
        top_m[i] = -1;
        top_ref[i] = 16'd0;
        top_out[i] = 16'd0;
      end

      eps = (ACTIVATION_WIDTH == 16) ? (1.0 / 16.0) : 8.0;

      for (m = 0; m < OUT_FEATURES; m = m + 1) begin
        for (pack_n = 0; pack_n < PACK_ROWS; pack_n = pack_n + 1) begin
          read_y_word(pack_n[15:0], m[15:0], word);
          for (lane = 0; lane < 4; lane = lane + 1) begin
            n = pack_n * 4 + lane;
            if (n < COMPARE_ROWS) begin
              out_bits = word[lane*16 +: 16];
              ref_bits = y_ref[n * OUT_FEATURES + m];
              y_out = fp16_to_real(out_bits);
              y_ans = fp16_to_real(ref_bits);
              abs_err = abs_real(y_out - y_ans);
              relabs_err = relative_or_absolute_error(y_out, y_ans, eps);

              mae_acc = mae_acc + abs_err;
              mse_acc = mse_acc + abs_err * abs_err;
              if (abs_err > linf_abs) begin
                linf_abs = abs_err;
              end
              if (relabs_err > max_relabs) begin
                max_relabs = relabs_err;
              end
              compare_count = compare_count + 1;

              update_top16(relabs_err, n, m, ref_bits, out_bits);
            end
          end
        end
      end

      mae_acc = mae_acc / compare_count;
      mse_acc = mse_acc / compare_count;
    end
  endtask

  initial begin
    clk = 1'b0;
    rst = 1'b1;
    en = 1'b0;
    clear = 1'b0;
    batch_size = BATCH_USED[$clog2(MAX_BATCH_SIZE)-1:0];
    w_type = W_TYPE_SEL;
    w_write_en = 1'b0;
    y_read_en = 1'b0;
    in_bus = 128'd0;
    w_valid = 1'b0;

    #1;
    rst = 1'b0;
    repeat (3) @(posedge clk);
    #1;
    rst = 1'b1;

    fd = $fopen(TEST_FILE, "r");
    if (fd == 0) begin
      $display("[TB_WNAM] cannot open %0s", TEST_FILE);
      $finish;
    end

    parse_header();
    load_x_data();

    en = 1'b1;
    stream_all_weights();
    parse_reference_output();

    i = 0;
    while (i < 50000000 && !done) begin
      @(posedge clk);
      i = i + 1;
    end

    if (!done) begin
      $display("[TB_WNAM] [FAIL] timeout waiting done");
      $finish;
    end

    if (!done_seen) begin
      done_cyc = sim_cyc;
    end

    collect_and_compare();

    $display("[TB_WNAM] CASE=%0s", TEST_FILE);
    $display("[TB_WNAM] BATCH_USED=%0d COMPARE_ROWS=%0d", BATCH_USED, COMPARE_ROWS);
    $display("[TB_WNAM] CYCLES en_to_done=%0d done_cycle=%0d en_start_cycle=%0d",
             done_cyc - en_start_cyc, done_cyc, en_start_cyc);
    $display("[TB_WNAM] MAE=%e MSE=%e Linf=%e", mae_acc, mse_acc, linf_abs);
    $display("[TB_WNAM] MAX_RELABS=%e THRESH=%f", max_relabs, 1);

    if (max_relabs < 1) begin
      $display("[PASS] max_relative_or_absolute_error=%e < 1", max_relabs);
    end else begin
      $display("[FAIL] max_relative_or_absolute_error=%e >= 1", max_relabs);
    end

    $display("[TB_WNAM] Top16 elements by relative_or_absolute_error:");
    for (i = 0; i < 16; i = i + 1) begin
      if (top_n[i] >= 0) begin
        $display("  idx=%0d pos=(n=%0d,m=%0d) score=%e ref=0x%04h(%e) out=0x%04h(%e)",
                 i,
                 top_n[i],
                 top_m[i],
                 top_score[i],
                 top_ref[i], fp16_to_real(top_ref[i]),
                 top_out[i], fp16_to_real(top_out[i]));
      end
    end

    $fclose(fd);
    #20;
    $finish;
  end

endmodule
