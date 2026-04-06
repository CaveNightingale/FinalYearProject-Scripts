`timescale 1ns / 1ps

module tb_systolic_array;

  localparam integer N = 4;
  localparam integer M = 16;
  localparam integer ExtW = 17;
  localparam integer OutW = 32;
  localparam integer Fp16W = 16;
  localparam integer PeOutLatency = 3;
  localparam integer PeOutLatencyBase = 2;
  localparam integer ExtraFlush = PeOutLatency - PeOutLatencyBase;
  localparam integer SumCycles = 2;
  localparam integer MaxK = 256;
  localparam integer CasesK1 = 16;
  localparam integer CasesK4 = 16;
  localparam integer CasesK16 = 16;
  localparam integer CasesK256 = 8;

  reg clk;
  reg rst;
  reg en;
  reg clear;
  reg [N*Fp16W-1:0] a_fp16;
  reg [M*Fp16W-1:0] b_fp16;
  wire [N*ExtW-1:0] a_ext;
  wire [M*ExtW-1:0] b_ext;
  wire [N*M*OutW-1:0] c;
  reg [N*M*OutW-1:0] c_prev;
  reg [N*M*OutW-1:0] c_curr;

  reg [N*MaxK*Fp16W-1:0] matrix_a;
  reg [MaxK*M*Fp16W-1:0] matrix_b;

  integer seed;
  integer total_cases;
  integer total_checks;
  integer fail_cnt;

  integer case_idx;
  integer lane_idx;

  real actual_value;
  real expected_value;
  real abs_err;
  real rel_err;
  real denom;
  real pos_abs_sum;
  real neg_abs_sum;
  real term_value;

  genvar ga;
  genvar gb;

  generate
    for (ga = 0; ga < N; ga = ga + 1) begin : g_fp16_ext_a
      fp16_ext u_fp16_ext_a (
          .in_e5m10 (a_fp16[ga*Fp16W+:Fp16W]),
          .out_e6m10_ftz(a_ext[ga*ExtW+:ExtW])
      );
    end

    for (gb = 0; gb < M; gb = gb + 1) begin : g_fp16_ext_b
      fp16_ext u_fp16_ext_b (
          .in_e5m10 (b_fp16[gb*Fp16W+:Fp16W]),
          .out_e6m10_ftz(b_ext[gb*ExtW+:ExtW])
      );
    end
  endgenerate

  systolic_array #(
      .N(N),
      .M(M)
  ) dut (
      .clk  (clk),
      .rst  (rst),
      .en   (en),
      .clear(clear),
      .width(3'd4),  // Use width=4 to enable all PEs
      .a    (a_ext),
      .b    (b_ext),
      .c    (c)
  );

  always #5 clk = ~clk;

  always @(posedge clk or negedge rst) begin
    if (!rst) begin
      c_prev <= {N * M * OutW{1'b0}};
      c_curr <= {N * M * OutW{1'b0}};
    end else if (en) begin
      c_prev <= c_curr;
      c_curr <= c;
    end
  end

  function automatic real rabs;
    input real x;
    begin
      if (x < 0.0) rabs = -x;
      else rabs = x;
    end
  endfunction

  function automatic real rmax;
    input real x;
    input real y;
    begin
      if (x > y) rmax = x;
      else rmax = y;
    end
  endfunction

  function automatic real fp32_to_real;
    input [31:0] bits;
    integer exp_unbiased;
    integer k;
    real mant;
    real value;
    begin
      if (bits[30:23] == 8'd0 && bits[22:0] == 23'd0) begin
        value = 0.0;
      end else if (bits[30:23] == 8'd0) begin
        mant = 0.0;
        for (k = 0; k < 23; k = k + 1) begin
          if (bits[22-k]) mant = mant + (2.0 ** (-(k + 1)));
        end
        value = mant * (2.0 ** (-126));
      end else begin
        exp_unbiased = bits[30:23] - 127;
        mant = 1.0;
        for (k = 0; k < 23; k = k + 1) begin
          if (bits[22-k]) mant = mant + (2.0 ** (-(k + 1)));
        end
        value = mant * (2.0 ** exp_unbiased);
      end

      if (bits[31]) fp32_to_real = -value;
      else fp32_to_real = value;
    end
  endfunction

  function automatic real fp16_to_real_std;
    input [15:0] bits;
    integer exp_unbiased;
    integer k;
    real frac;
    real value;
    begin
      if (bits[14:10] == 5'd0) begin
        if (bits[9:0] == 10'd0) begin
          value = 0.0;
        end else begin
          frac = 0.0;
          for (k = 0; k < 10; k = k + 1) begin
            if (bits[9-k]) frac = frac + (2.0 ** (-(k + 1)));
          end
          value = frac * (2.0 ** (-14));
        end
        end else if (bits[14:10] == 5'h1F) begin
          value = 65504.0;
      end else begin
        exp_unbiased = bits[14:10] - 15;
        frac = 1.0;
        for (k = 0; k < 10; k = k + 1) begin
          if (bits[9-k]) frac = frac + (2.0 ** (-(k + 1)));
        end
        value = frac * (2.0 ** exp_unbiased);
      end

      if (bits[15]) fp16_to_real_std = -value;
      else fp16_to_real_std = value;
    end
  endfunction

  function automatic [15:0] rand_finite_fp16;
    reg [15:0] bits;
    begin
      seed = seed * 1664525 + 1013904223;
      bits[15] = seed[31];
      bits[14:10] = 5'd14;
      bits[9:0] = seed[9:0];
      rand_finite_fp16 = bits;
    end
  endfunction

  task automatic apply_reset;
    begin
      rst = 1'b1;
      #1;
      rst = 1'b0;
      en = 1'b0;
      clear = 1'b0;
      a_fp16 = {N * Fp16W{1'b0}};
      b_fp16 = {M * Fp16W{1'b0}};
      @(posedge clk);
      #1;
      rst = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic generate_random_matrices;
    input integer k_len;
    integer row;
    integer col;
    integer kk;
    begin
      for (row = 0; row < N; row = row + 1) begin
        for (kk = 0; kk < k_len; kk = kk + 1) begin
          matrix_a[(row*MaxK+kk)*Fp16W+:Fp16W] = rand_finite_fp16();
        end
      end

      for (kk = 0; kk < k_len; kk = kk + 1) begin
        for (col = 0; col < M; col = col + 1) begin
          matrix_b[(kk*M+col)*Fp16W+:Fp16W] = rand_finite_fp16();
        end
      end
    end
  endtask

  task automatic drive_systolic_array;
    input integer k_len;
    integer cyc;
    integer row;
    integer col;
    integer flush;
    begin
      apply_reset();

      for (cyc = 0; cyc < k_len; cyc = cyc + 1) begin
        for (row = 0; row < N; row = row + 1) begin
          a_fp16[row*Fp16W+:Fp16W] = matrix_a[(row*MaxK+cyc)*Fp16W+:Fp16W];
        end

        for (col = 0; col < M; col = col + 1) begin
          b_fp16[col*Fp16W+:Fp16W] = matrix_b[(cyc*M+col)*Fp16W+:Fp16W];
        end

        en = 1'b1;
        clear = 1'b0;
        @(posedge clk);
        #1;
      end

      a_fp16 = {N * Fp16W{1'b0}};
      b_fp16 = {M * Fp16W{1'b0}};

      for (flush = 0; flush < (N + M - 1 + ExtraFlush + (SumCycles - 1)); flush = flush + 1) begin
        en = 1'b1;
        clear = 1'b0;
        @(posedge clk);
        #1;
      end

      en = 1'b0;
      clear = 1'b0;
      #1;
    end
  endtask

  task automatic check_outputs;
    input integer k_len;
    integer row;
    integer col;
    integer kk;
    integer out_idx;
    begin
      total_cases = total_cases + 1;

      for (row = 0; row < N; row = row + 1) begin
        for (col = 0; col < M; col = col + 1) begin
          expected_value = 0.0;
          pos_abs_sum = 0.0;
          neg_abs_sum = 0.0;

          for (kk = 0; kk < k_len; kk = kk + 1) begin
            term_value = fp16_to_real_std(matrix_a[(row*MaxK+kk)*Fp16W+:Fp16W])
              * fp16_to_real_std(matrix_b[(kk*M+col)*Fp16W+:Fp16W]);
            expected_value = expected_value + term_value;

            if (term_value >= 0.0) begin
              pos_abs_sum = pos_abs_sum + term_value;
            end else begin
              neg_abs_sum = neg_abs_sum + (-term_value);
            end
          end

          out_idx = row * M + col;
          actual_value = fp32_to_real(c_prev[out_idx*OutW+:OutW])
            + fp32_to_real(c_curr[out_idx*OutW+:OutW]);
          abs_err = rabs(actual_value - expected_value);
          denom = rmax(rmax(pos_abs_sum, neg_abs_sum), 1.0);
          rel_err = abs_err / denom;

          total_checks = total_checks + 1;

          if ((abs_err > 1.0e-3) && (rel_err > 1.0e-3)) begin
            fail_cnt = fail_cnt + 1;
            $display("[FAIL] K=%0d c=%0d r=%0d col=%0d out_prev=%h out_curr=%h a=%e e=%e ae=%e re=%e d=%e", k_len,
                     total_cases, row, col, c_prev[out_idx*OutW+:OutW], c_curr[out_idx*OutW+:OutW],
                     actual_value, expected_value, abs_err, rel_err, denom);
          end
        end
      end
    end
  endtask

  task automatic run_random_suite;
    input integer k_len;
    input integer case_count;
    integer local_case;
    begin
      for (local_case = 0; local_case < case_count; local_case = local_case + 1) begin
        generate_random_matrices(k_len);
        drive_systolic_array(k_len);
        check_outputs(k_len);
      end

      $display("[INFO] K=%0d done cases=%0d checks=%0d fail=%0d", k_len, case_count, total_checks,
               fail_cnt);
    end
  endtask

  initial begin
    clk = 1'b0;
    rst = 1'b0;
    en = 1'b0;
    clear = 1'b0;
    a_fp16 = {N * Fp16W{1'b0}};
    b_fp16 = {M * Fp16W{1'b0}};
    c_prev = {N * M * OutW{1'b0}};
    c_curr = {N * M * OutW{1'b0}};

    seed = 32'h4A3C91D2;
    total_cases = 0;
    total_checks = 0;
    fail_cnt = 0;

    for (case_idx = 0; case_idx < N * MaxK; case_idx = case_idx + 1) begin
      matrix_a[case_idx*Fp16W+:Fp16W] = 16'h0000;
    end
    for (lane_idx = 0; lane_idx < MaxK * M; lane_idx = lane_idx + 1) begin
      matrix_b[lane_idx*Fp16W+:Fp16W] = 16'h0000;
    end

    apply_reset();

    run_random_suite(1, CasesK1);
    run_random_suite(4, CasesK4);
    run_random_suite(16, CasesK16);
    run_random_suite(256, CasesK256);

    if (fail_cnt == 0) begin
      $display("[PASS] tb_systolic_array done. cases=%0d checks=%0d fail=%0d", total_cases,
               total_checks, fail_cnt);
    end else begin
      $display("[FAIL] tb_systolic_array done. cases=%0d checks=%0d fail=%0d", total_cases,
               total_checks, fail_cnt);
      $stop;
    end

    $finish;
  end

endmodule
