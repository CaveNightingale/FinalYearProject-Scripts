`timescale 1ns / 1ps

module tb_gemm_wnam;

  localparam integer BATCH = 8;
  localparam integer K = 64;
  localparam integer OUT = 32;
  localparam integer BN = BATCH / 4;
  localparam integer BM = OUT / 16;
  localparam integer KG = K / 16;
  localparam real N_STD = 0.1414213562373095;
  localparam real TOL = 1.0e-3;
  localparam real TOL_A8_Q = 5.0e-2;
  localparam integer DUMP_ALL_CASES = 1;

  integer i;
  integer j;
  integer k;
  integer lane;
  integer cyc;
  integer fail_cases;
  integer total_cycles;
  integer case_id;
  integer rand_seed;

  real x_real[0:BATCH*K-1];
  real w_real[0:K*OUT-1];
  real w_deq[0:K*OUT-1];
  real y_ref[0:BATCH*OUT-1];

  real y_hw_16_nq[0:BATCH*OUT-1];
  real y_hw_16_q[0:BATCH*OUT-1];
  real y_hw_8_nq[0:BATCH*OUT-1];
  real y_hw_8_q[0:BATCH*OUT-1];

  integer y_valid_16_nq[0:BATCH*OUT-1];
  integer y_valid_16_q[0:BATCH*OUT-1];
  integer y_valid_8_nq[0:BATCH*OUT-1];
  integer y_valid_8_q[0:BATCH*OUT-1];

  real scale_map[0:BM*KG*16-1];

  reg clk;
  reg rst;

  reg en16_nq, clear16_nq;
  reg [1:0] w_type16_nq;
  reg [63:0] x_feed16_nq;
  reg [127:0] wq_feed16_nq, wz_feed16_nq;
  reg [255:0] ws_feed16_nq;
  wire [63:0] y_data16_nq;
  wire [15:0] query_n16_nq, query_m16_nq;
  wire [23:0] query_k16_nq;
  wire [15:0] submit_n16_nq, submit_m16_nq;
  wire query_en16_nq, submit_en16_nq, done16_nq;

  reg en16_q, clear16_q;
  reg [1:0] w_type16_q;
  reg [63:0] x_feed16_q;
  reg [127:0] wq_feed16_q, wz_feed16_q;
  reg [255:0] ws_feed16_q;
  wire [63:0] y_data16_q;
  wire [15:0] query_n16_q, query_m16_q;
  wire [23:0] query_k16_q;
  wire [15:0] submit_n16_q, submit_m16_q;
  wire query_en16_q, submit_en16_q, done16_q;

  reg en8_nq, clear8_nq;
  reg [1:0] w_type8_nq;
  reg [31:0] x_feed8_nq;
  reg [63:0] wq_feed8_nq, wz_feed8_nq;
  reg [127:0] ws_feed8_nq;
  wire [63:0] y_data8_nq;
  wire [15:0] query_n8_nq, query_m8_nq;
  wire [23:0] query_k8_nq;
  wire [15:0] submit_n8_nq, submit_m8_nq;
  wire query_en8_nq, submit_en8_nq, done8_nq;

  reg en8_q, clear8_q;
  reg [1:0] w_type8_q;
  reg [31:0] x_feed8_q;
  reg [63:0] wq_feed8_q, wz_feed8_q;
  reg [127:0] ws_feed8_q;
  wire [63:0] y_data8_q;
  wire [15:0] query_n8_q, query_m8_q;
  wire [23:0] query_k8_q;
  wire [15:0] submit_n8_q, submit_m8_q;
  wire query_en8_q, submit_en8_q, done8_q;

  reg req_valid_q16_nq, req_valid_q16_q, req_valid_q8_nq, req_valid_q8_q;
  reg [15:0] req_n_q16_nq, req_m_q16_nq;
  reg [23:0] req_k_q16_nq;
  reg [15:0] req_n_q16_q, req_m_q16_q;
  reg [23:0] req_k_q16_q;
  reg [15:0] req_n_q8_nq, req_m_q8_nq;
  reg [23:0] req_k_q8_nq;
  reg [15:0] req_n_q8_q, req_m_q8_q;
  reg [23:0] req_k_q8_q;

  reg [63:0] x_mem16_nq[0:2047];
  reg [127:0] wq_mem16_nq[0:2047];
  reg [127:0] wz_mem16_nq[0:2047];
  reg [255:0] ws_mem16_nq[0:2047];

  reg [63:0] x_mem16_q[0:2047];
  reg [127:0] wq_mem16_q[0:2047];
  reg [127:0] wz_mem16_q[0:2047];
  reg [255:0] ws_mem16_q[0:2047];

  reg [31:0] x_mem8_nq[0:2047];
  reg [63:0] wq_mem8_nq[0:2047];
  reg [63:0] wz_mem8_nq[0:2047];
  reg [127:0] ws_mem8_nq[0:2047];

  reg [31:0] x_mem8_q[0:2047];
  reg [63:0] wq_mem8_q[0:2047];
  reg [63:0] wz_mem8_q[0:2047];
  reg [127:0] ws_mem8_q[0:2047];

  function automatic integer idx_x;
    input integer n;
    input integer kk;
    begin
      idx_x = n * K + kk;
    end
  endfunction

  function automatic integer idx_w;
    input integer kk;
    input integer m;
    begin
      idx_w = kk * OUT + m;
    end
  endfunction

  function automatic integer idx_y;
    input integer n;
    input integer m;
    begin
      idx_y = n * OUT + m;
    end
  endfunction

  function automatic integer idx_scale;
    input integer bm_i;
    input integer g_i;
    input integer lane_i;
    begin
      idx_scale = (bm_i * KG + g_i) * 16 + lane_i;
    end
  endfunction

  function automatic real fabs;
    input real v;
    begin
      fabs = (v < 0.0) ? -v : v;
    end
  endfunction

  function automatic real fmax3;
    input real a;
    input real b;
    input real c;
    real t;
    begin
      t = (a > b) ? a : b;
      fmax3 = (t > c) ? t : c;
    end
  endfunction

  function automatic real pow2i;
    input integer exp_v;
    integer t;
    real r;
    begin
      r = 1.0;
      if (exp_v >= 0) begin
        for (t = 0; t < exp_v; t = t + 1) r = r * 2.0;
      end else begin
        for (t = 0; t < -exp_v; t = t + 1) r = r / 2.0;
      end
      pow2i = r;
    end
  endfunction

  function automatic real clamp_scale_fp16;
    input real v;
    real min_v;
    real max_v;
    begin
      min_v = pow2i(-24);
      max_v = (2.0 - (1.0 / 1024.0)) * pow2i(16);
      if (v <= 0.0) clamp_scale_fp16 = 0.0;
      else if (v < min_v) clamp_scale_fp16 = min_v;
      else if (v > max_v) clamp_scale_fp16 = max_v;
      else clamp_scale_fp16 = v;
    end
  endfunction

  function automatic real clamp_scale_fp8;
    input real v;
    real min_v;
    real max_v;
    begin
      min_v = pow2i(-9);
      max_v = (2.0 - (1.0 / 8.0)) * pow2i(8);
      if (v <= 0.0) clamp_scale_fp8 = 0.0;
      else if (v < min_v) clamp_scale_fp8 = min_v;
      else if (v > max_v) clamp_scale_fp8 = max_v;
      else clamp_scale_fp8 = v;
    end
  endfunction

  function automatic integer round_nearest_int;
    input real v;
    begin
      if (v >= 0.0) round_nearest_int = $rtoi(v + 0.5);
      else round_nearest_int = -$rtoi(-v + 0.5);
    end
  endfunction

  function automatic real rand_u01;
    input integer dummy;
    integer r;
    begin
      rand_seed = rand_seed * 1103515245 + 12345;
      r = (rand_seed >> 1) & 31'h7fffffff;
      if (r < 0) r = -r;
      rand_u01 = (r % 1000000) / 1000000.0;
    end
  endfunction

  function automatic real randn_approx;
    input integer dummy;
    integer t;
    real s;
    begin
      s = 0.0;
      for (t = 0; t < 12; t = t + 1) s = s + rand_u01(0);
      randn_approx = (s - 6.0) * N_STD;
    end
  endfunction

  function automatic [15:0] fp16_from_real_sat;
    input real v;
    reg sign;
    real a;
    real frac;
    real mant_r;
    integer exp_unb;
    integer exp_field;
    integer mant_i;
    real max_abs;
    begin
      max_abs = (2.0 - (1.0 / 1024.0)) * pow2i(16);
      if (v == 0.0) begin
        fp16_from_real_sat = 16'd0;
      end else begin
        sign = (v < 0.0);
        a = sign ? -v : v;
        if (a > max_abs) begin
          fp16_from_real_sat = sign ? 16'hffff : 16'h7fff;
        end else begin
          exp_unb = 0;
          frac = a;
          while (frac >= 2.0) begin
            frac = frac / 2.0;
            exp_unb = exp_unb + 1;
          end
          while (frac < 1.0) begin
            frac = frac * 2.0;
            exp_unb = exp_unb - 1;
          end
          exp_field = exp_unb + 15;
          if (exp_field <= 0) begin
            mant_r = a / pow2i(1 - 15) * 1024.0;
            mant_i = $rtoi(mant_r + 0.5);
            if (mant_i > 1023) mant_i = 1023;
            fp16_from_real_sat = {sign, 5'd0, mant_i[9:0]};
          end else begin
            mant_r = (frac - 1.0) * 1024.0;
            mant_i = $rtoi(mant_r + 0.5);
            if (mant_i >= 1024) begin
              mant_i = 0;
              exp_field = exp_field + 1;
            end
            if (exp_field >= 32) fp16_from_real_sat = sign ? 16'hffff : 16'h7fff;
            else fp16_from_real_sat = {sign, exp_field[4:0], mant_i[9:0]};
          end
        end
      end
    end
  endfunction

  function automatic real fp16_to_real;
    input [15:0] x;
    integer exp_unb;
    real frac;
    begin
      if (x[14:10] == 0) begin
        frac = (x[9:0] * 1.0) / 1024.0;
        fp16_to_real = (x[15] ? -1.0 : 1.0) * pow2i(1 - 15) * frac;
      end else begin
        exp_unb = x[14:10] - 15;
        frac = 1.0 + (x[9:0] * 1.0) / 1024.0;
        fp16_to_real = (x[15] ? -1.0 : 1.0) * pow2i(exp_unb) * frac;
      end
    end
  endfunction

  function automatic [15:0] e6m10_from_real;
    input real v;
    real a;
    real frac;
    real mant_r;
    integer exp_unb;
    integer exp_field;
    integer mant_i;
    begin
      if (v <= 0.0) begin
        e6m10_from_real = 16'd0;
      end else begin
        a = v;
        exp_unb = 0;
        frac = a;
        while (frac >= 2.0) begin
          frac = frac / 2.0;
          exp_unb = exp_unb + 1;
        end
        while (frac < 1.0) begin
          frac = frac * 2.0;
          exp_unb = exp_unb - 1;
        end
        exp_field = exp_unb + 47;
        if (exp_field <= 0 || exp_field >= 64) begin
          e6m10_from_real = 16'd0;
        end else begin
          mant_r = (frac - 1.0) * 1024.0;
          mant_i = $rtoi(mant_r + 0.5);
          if (mant_i >= 1024) begin
            mant_i = 0;
            exp_field = exp_field + 1;
          end
          if (exp_field >= 64) e6m10_from_real = 16'd0;
          else e6m10_from_real = {exp_field[5:0], mant_i[9:0]};
        end
      end
    end
  endfunction

  function automatic real decode_e6m10;
    input [15:0] x;
    integer exp_unb;
    real frac;
    begin
      if (x[15:10] == 0) decode_e6m10 = 0.0;
      else begin
        exp_unb = x[15:10] - 47;
        frac = 1.0 + (x[9:0] * 1.0) / 1024.0;
        decode_e6m10 = pow2i(exp_unb) * frac;
      end
    end
  endfunction

  function automatic [7:0] e5m3_from_real;
    input real v;
    real a;
    real frac;
    real mant_r;
    integer exp_unb;
    integer exp_field;
    integer mant_i;
    begin
      if (v <= 0.0) begin
        e5m3_from_real = 8'd0;
      end else begin
        a = v;
        exp_unb = 0;
        frac = a;
        while (frac >= 2.0) begin
          frac = frac / 2.0;
          exp_unb = exp_unb + 1;
        end
        while (frac < 1.0) begin
          frac = frac * 2.0;
          exp_unb = exp_unb - 1;
        end
        exp_field = exp_unb + 23;
        if (exp_field <= 0 || exp_field >= 32) begin
          e5m3_from_real = 8'd0;
        end else begin
          mant_r = (frac - 1.0) * 8.0;
          mant_i = $rtoi(mant_r + 0.5);
          if (mant_i >= 8) begin
            mant_i = 0;
            exp_field = exp_field + 1;
          end
          if (exp_field >= 32) e5m3_from_real = 8'd0;
          else e5m3_from_real = {exp_field[4:0], mant_i[2:0]};
        end
      end
    end
  endfunction

  function automatic real decode_e5m3;
    input [7:0] x;
    integer exp_unb;
    real frac;
    begin
      if (x[7:3] == 0) decode_e5m3 = 0.0;
      else begin
        exp_unb = x[7:3] - 23;
        frac = 1.0 + (x[2:0] * 1.0) / 8.0;
        decode_e5m3 = pow2i(exp_unb) * frac;
      end
    end
  endfunction

  function automatic [7:0] encode_e4m3_fnuz;
    input real v;
    reg sign;
    real a;
    real frac;
    integer exp_unb;
    integer exp_field;
    integer mant_i;
    real mant_r;
    begin
      if (v == 0.0) begin
        encode_e4m3_fnuz = 8'd0;
      end else begin
        sign = (v < 0.0);
        a = sign ? -v : v;
        exp_unb = 0;
        frac = a;
        while (frac >= 2.0) begin
          frac = frac / 2.0;
          exp_unb = exp_unb + 1;
        end
        while (frac < 1.0) begin
          frac = frac * 2.0;
          exp_unb = exp_unb - 1;
        end
        exp_field = exp_unb + 7;
        if (exp_field <= 0) begin
          mant_r = a / pow2i(1 - 7) * 8.0;
          mant_i = $rtoi(mant_r + 0.5);
          if (mant_i > 7) mant_i = 7;
          encode_e4m3_fnuz = {sign, 4'd0, mant_i[2:0]};
        end else if (exp_field >= 16) begin
          encode_e4m3_fnuz = {sign, 4'hf, 3'h7};
        end else begin
          mant_r = (frac - 1.0) * 8.0;
          mant_i = $rtoi(mant_r + 0.5);
          if (mant_i >= 8) begin
            mant_i = 0;
            exp_field = exp_field + 1;
          end
          if (exp_field >= 16) encode_e4m3_fnuz = {sign, 4'hf, 3'h7};
          else encode_e4m3_fnuz = {sign, exp_field[3:0], mant_i[2:0]};
        end
      end
    end
  endfunction

  function automatic real decode_e4m3_fnuz;
    input [7:0] x;
    integer exp_unb;
    real frac;
    begin
      if (x[6:0] == 0) decode_e4m3_fnuz = 0.0;
      else if (x[6:3] == 0) begin
        frac = (x[2:0] * 1.0) / 8.0;
        decode_e4m3_fnuz = (x[7] ? -1.0 : 1.0) * pow2i(1 - 7) * frac;
      end else begin
        exp_unb = x[6:3] - 7;
        frac = 1.0 + (x[2:0] * 1.0) / 8.0;
        decode_e4m3_fnuz = (x[7] ? -1.0 : 1.0) * pow2i(exp_unb) * frac;
      end
    end
  endfunction

  function automatic [3:0] encode_e2m1_fnuz;
    input real v;
    reg sign;
    real a;
    real frac;
    integer exp_unb;
    integer exp_field;
    integer mant_i;
    real mant_r;
    begin
      if (v == 0.0) begin
        encode_e2m1_fnuz = 4'd0;
      end else begin
        sign = (v < 0.0);
        a = sign ? -v : v;
        exp_unb = 0;
        frac = a;
        while (frac >= 2.0) begin
          frac = frac / 2.0;
          exp_unb = exp_unb + 1;
        end
        while (frac < 1.0) begin
          frac = frac * 2.0;
          exp_unb = exp_unb - 1;
        end
        exp_field = exp_unb + 1;
        if (exp_field <= 0) begin
          mant_r = a / pow2i(1 - 1) * 2.0;
          mant_i = $rtoi(mant_r + 0.5);
          if (mant_i > 1) mant_i = 1;
          encode_e2m1_fnuz = {sign, 2'd0, mant_i[0]};
        end else if (exp_field >= 4) begin
          encode_e2m1_fnuz = {sign, 2'b11, 1'b1};
        end else begin
          mant_r = (frac - 1.0) * 2.0;
          mant_i = $rtoi(mant_r + 0.5);
          if (mant_i >= 2) begin
            mant_i = 0;
            exp_field = exp_field + 1;
          end
          if (exp_field >= 4) encode_e2m1_fnuz = {sign, 2'b11, 1'b1};
          else encode_e2m1_fnuz = {sign, exp_field[1:0], mant_i[0]};
        end
      end
    end
  endfunction

  function automatic real decode_e2m1_fnuz;
    input [3:0] x;
    integer exp_unb;
    real frac;
    begin
      if (x[2:0] == 0) decode_e2m1_fnuz = 0.0;
      else if (x[2:1] == 0) begin
        frac = x[0] ? 0.5 : 0.0;
        decode_e2m1_fnuz = (x[3] ? -1.0 : 1.0) * pow2i(1 - 1) * frac;
      end else begin
        exp_unb = x[2:1] - 1;
        frac = 1.0 + (x[0] ? 0.5 : 0.0);
        decode_e2m1_fnuz = (x[3] ? -1.0 : 1.0) * pow2i(exp_unb) * frac;
      end
    end
  endfunction

  function automatic [127:0] set_byte128;
    input [127:0] vec;
    input integer idx;
    input [7:0] b;
    reg [127:0] t;
    begin
      t = vec;
      t[idx*8 +: 8] = b;
      set_byte128 = t;
    end
  endfunction

  function automatic [63:0] set_nibble64;
    input [63:0] vec;
    input integer idx;
    input [3:0] n;
    reg [63:0] t;
    begin
      t = vec;
      t[idx*4 +: 4] = n;
      set_nibble64 = t;
    end
  endfunction

  function automatic [255:0] set_half256;
    input [255:0] vec;
    input integer idx;
    input [15:0] h;
    reg [255:0] t;
    begin
      t = vec;
      t[idx*16 +: 16] = h;
      set_half256 = t;
    end
  endfunction

  function automatic [15:0] get_lane16_64;
    input [63:0] v;
    input integer l;
    begin
      case (l)
        0: get_lane16_64 = v[15:0];
        1: get_lane16_64 = v[31:16];
        2: get_lane16_64 = v[47:32];
        default: get_lane16_64 = v[63:48];
      endcase
    end
  endfunction

  task automatic gen_random_xy;
    integer n;
    integer m;
    integer kk;
    begin
      for (n = 0; n < BATCH; n = n + 1)
        for (kk = 0; kk < K; kk = kk + 1)
          x_real[idx_x(n, kk)] = randn_approx(0);

      for (kk = 0; kk < K; kk = kk + 1)
        for (m = 0; m < OUT; m = m + 1)
          w_real[idx_w(kk, m)] = randn_approx(0);
    end
  endtask

  task automatic reset_capture_16_nq;
    integer t;
    begin
      for (t = 0; t < BATCH * OUT; t = t + 1) begin
        y_hw_16_nq[t] = 0.0;
        y_valid_16_nq[t] = 0;
      end
    end
  endtask

  task automatic reset_capture_16_q;
    integer t;
    begin
      for (t = 0; t < BATCH * OUT; t = t + 1) begin
        y_hw_16_q[t] = 0.0;
        y_valid_16_q[t] = 0;
      end
    end
  endtask

  task automatic reset_capture_8_nq;
    integer t;
    begin
      for (t = 0; t < BATCH * OUT; t = t + 1) begin
        y_hw_8_nq[t] = 0.0;
        y_valid_8_nq[t] = 0;
      end
    end
  endtask

  task automatic reset_capture_8_q;
    integer t;
    begin
      for (t = 0; t < BATCH * OUT; t = t + 1) begin
        y_hw_8_q[t] = 0.0;
        y_valid_8_q[t] = 0;
      end
    end
  endtask

  task automatic preload_x16_nq;
    integer bn_i;
    integer kk;
    integer nbase;
    reg [63:0] pack_x;
    begin
      for (bn_i = 0; bn_i < BN; bn_i = bn_i + 1) begin
        nbase = bn_i * 4;
        for (kk = 0; kk < K; kk = kk + 1) begin
          pack_x = {
            fp16_from_real_sat(x_real[idx_x(nbase+3, kk)]),
            fp16_from_real_sat(x_real[idx_x(nbase+2, kk)]),
            fp16_from_real_sat(x_real[idx_x(nbase+1, kk)]),
            fp16_from_real_sat(x_real[idx_x(nbase+0, kk)])
          };
          x_mem16_nq[bn_i * K + kk] = pack_x;
        end
      end
    end
  endtask

  task automatic preload_x16_q;
    integer bn_i;
    integer kk;
    integer nbase;
    reg [63:0] pack_x;
    begin
      for (bn_i = 0; bn_i < BN; bn_i = bn_i + 1) begin
        nbase = bn_i * 4;
        for (kk = 0; kk < K; kk = kk + 1) begin
          pack_x = {
            fp16_from_real_sat(x_real[idx_x(nbase+3, kk)]),
            fp16_from_real_sat(x_real[idx_x(nbase+2, kk)]),
            fp16_from_real_sat(x_real[idx_x(nbase+1, kk)]),
            fp16_from_real_sat(x_real[idx_x(nbase+0, kk)])
          };
          x_mem16_q[bn_i * K + kk] = pack_x;
        end
      end
    end
  endtask

  task automatic preload_x8_nq;
    integer bn_i;
    integer kk;
    integer nbase;
    reg [31:0] pack_x;
    begin
      for (bn_i = 0; bn_i < BN; bn_i = bn_i + 1) begin
        nbase = bn_i * 4;
        for (kk = 0; kk < K; kk = kk + 1) begin
          pack_x = {
            encode_e4m3_fnuz(x_real[idx_x(nbase+3, kk)]),
            encode_e4m3_fnuz(x_real[idx_x(nbase+2, kk)]),
            encode_e4m3_fnuz(x_real[idx_x(nbase+1, kk)]),
            encode_e4m3_fnuz(x_real[idx_x(nbase+0, kk)])
          };
          x_mem8_nq[bn_i * K + kk] = pack_x;
        end
      end
    end
  endtask

  task automatic preload_x8_q;
    integer bn_i;
    integer kk;
    integer nbase;
    reg [31:0] pack_x;
    begin
      for (bn_i = 0; bn_i < BN; bn_i = bn_i + 1) begin
        nbase = bn_i * 4;
        for (kk = 0; kk < K; kk = kk + 1) begin
          pack_x = {
            encode_e4m3_fnuz(x_real[idx_x(nbase+3, kk)]),
            encode_e4m3_fnuz(x_real[idx_x(nbase+2, kk)]),
            encode_e4m3_fnuz(x_real[idx_x(nbase+1, kk)]),
            encode_e4m3_fnuz(x_real[idx_x(nbase+0, kk)])
          };
          x_mem8_q[bn_i * K + kk] = pack_x;
        end
      end
    end
  endtask

  task automatic quantize_w16_noquant;
    integer bm_i, g_i, kk, k0, m;
    reg [255:0] pack_s;
    reg [15:0] w_h;
    begin
      for (bm_i = 0; bm_i < BM; bm_i = bm_i + 1) begin
        for (kk = 0; kk < K; kk = kk + 1) begin
          pack_s = 256'd0;
          for (lane = 0; lane < 16; lane = lane + 1) begin
            m = bm_i * 16 + lane;
            w_h = fp16_from_real_sat(w_real[idx_w(kk, m)]);
            pack_s = set_half256(pack_s, lane, w_h);
            w_deq[idx_w(kk, m)] = fp16_to_real(w_h);
          end
          ws_mem16_nq[bm_i * K + kk] = pack_s;
        end
        for (kk = 0; kk < K; kk = kk + 1) begin
          wq_mem16_nq[bm_i * K + kk] = 0;
          wz_mem16_nq[bm_i * K + kk] = 0;
        end
      end
    end
  endtask

  task automatic quantize_w16_quant;
    input [1:0] dtype;
    integer bm_i, g_i, kk, m;
    real max_abs;
    real scale_sel;
    real scale_r;
    real v;
    real vn;
    integer q_i;
    integer z_i;
    reg [7:0] q_b;
    reg [7:0] z_b;
    reg [127:0] pack_q;
    reg [127:0] pack_z;
    reg [255:0] pack_s;
    reg [15:0] s_h;
    begin
      for (bm_i = 0; bm_i < BM; bm_i = bm_i + 1) begin
        for (g_i = 0; g_i < KG; g_i = g_i + 1) begin
          for (lane = 0; lane < 16; lane = lane + 1) begin
            m = bm_i * 16 + lane;
            max_abs = 0.0;
            for (kk = g_i * 16; kk < g_i * 16 + 16; kk = kk + 1) begin
              v = fabs(w_real[idx_w(kk, m)]);
              if (v > max_abs) max_abs = v;
            end
            if (max_abs < 1.0e-12) begin
              scale_sel = 0.0;
            end else if (dtype == 2'b00) begin
              scale_sel = clamp_scale_fp16(max_abs / 480.0);
            end else if (dtype == 2'b01) begin
              scale_sel = clamp_scale_fp16(max_abs / 127.0);
            end else if (dtype == 2'b10) begin
              scale_sel = clamp_scale_fp16(max_abs / 6.0);
            end else begin
              scale_sel = clamp_scale_fp16(max_abs / 7.0);
            end
            scale_map[idx_scale(bm_i, g_i, lane)] = scale_sel;
          end
        end
      end

      for (bm_i = 0; bm_i < BM; bm_i = bm_i + 1) begin
        for (g_i = 0; g_i < KG; g_i = g_i + 1) begin
          pack_s = 256'd0;
          for (lane = 0; lane < 16; lane = lane + 1) begin
            s_h = e6m10_from_real(scale_map[idx_scale(bm_i, g_i, lane)]);
            pack_s = set_half256(pack_s, lane, s_h);
          end
          ws_mem16_q[bm_i * KG + g_i] = pack_s;
        end

        for (kk = 0; kk < K; kk = kk + 1) begin
          g_i = kk >> 4;
          pack_q = 128'd0;
          pack_z = 128'd0;
          for (lane = 0; lane < 16; lane = lane + 1) begin
            m = bm_i * 16 + lane;
            scale_r = decode_e6m10(e6m10_from_real(scale_map[idx_scale(bm_i, g_i, lane)]));
            if (scale_r < 1.0e-12) scale_r = 1.0e-12;
            v = w_real[idx_w(kk, m)];
            vn = v / scale_r;

            if (dtype == 2'b00) begin
              q_b = encode_e4m3_fnuz(vn);
              z_b = 8'd0;
              w_deq[idx_w(kk, m)] = decode_e4m3_fnuz(q_b) * scale_r;
            end else if (dtype == 2'b10) begin
              q_b = {4'd0, encode_e2m1_fnuz(vn)};
              z_b = 8'd0;
              w_deq[idx_w(kk, m)] = decode_e2m1_fnuz(q_b[3:0]) * scale_r;
            end else if (dtype == 2'b01) begin
              z_i = 128;
              q_i = round_nearest_int(v / scale_r) + z_i;
              if (q_i < 0) q_i = 0;
              if (q_i > 255) q_i = 255;
              q_b = q_i[7:0];
              z_b = z_i[7:0];
              w_deq[idx_w(kk, m)] = (q_i - z_i) * scale_r;
            end else begin
              z_i = 8;
              q_i = round_nearest_int(v / scale_r) + z_i;
              if (q_i < 0) q_i = 0;
              if (q_i > 15) q_i = 15;
              q_b = q_i[7:0];
              z_b = z_i[7:0];
              w_deq[idx_w(kk, m)] = (q_i - z_i) * scale_r;
            end

            pack_q = set_byte128(pack_q, lane, q_b);
            pack_z = set_byte128(pack_z, lane, z_b);
          end
          wq_mem16_q[bm_i * K + kk] = pack_q;
          wz_mem16_q[bm_i * K + kk] = pack_z;
        end
      end
    end
  endtask

  task automatic quantize_w8_noquant;
    integer bm_i, kk, m;
    reg [127:0] pack_s;
    reg [7:0] w_b;
    begin
      for (bm_i = 0; bm_i < BM; bm_i = bm_i + 1) begin
        for (kk = 0; kk < K; kk = kk + 1) begin
          pack_s = 128'd0;
          for (lane = 0; lane < 16; lane = lane + 1) begin
            m = bm_i * 16 + lane;
            w_b = encode_e4m3_fnuz(w_real[idx_w(kk, m)]);
            pack_s = set_byte128(pack_s, lane, w_b);
            w_deq[idx_w(kk, m)] = decode_e4m3_fnuz(w_b);
          end
          ws_mem8_nq[bm_i * K + kk] = pack_s;
        end
        for (kk = 0; kk < K; kk = kk + 1) begin
          wq_mem8_nq[bm_i * K + kk] = 0;
          wz_mem8_nq[bm_i * K + kk] = 0;
        end
      end
    end
  endtask

  task automatic quantize_w8_quant;
    input [1:0] dtype;
    integer bm_i, g_i, kk, m;
    real max_abs;
    real scale_sel;
    real scale_r;
    real v;
    real vn;
    integer q_i;
    integer z_i;
    reg [3:0] q_n;
    reg [3:0] z_n;
    reg [63:0] pack_q;
    reg [63:0] pack_z;
    reg [127:0] pack_s;
    reg [7:0] s_b;
    begin
      for (bm_i = 0; bm_i < BM; bm_i = bm_i + 1) begin
        for (g_i = 0; g_i < KG; g_i = g_i + 1) begin
          for (lane = 0; lane < 16; lane = lane + 1) begin
            m = bm_i * 16 + lane;
            max_abs = 0.0;
            for (kk = g_i * 16; kk < g_i * 16 + 16; kk = kk + 1) begin
              v = fabs(w_real[idx_w(kk, m)]);
              if (v > max_abs) max_abs = v;
            end
            if (max_abs < 1.0e-12) begin
              scale_sel = 0.0;
            end else if (dtype == 2'b10) begin
              scale_sel = clamp_scale_fp8(max_abs / 6.0);
            end else begin
              scale_sel = clamp_scale_fp8(max_abs / 7.0);
            end
            scale_map[idx_scale(bm_i, g_i, lane)] = scale_sel;
          end
        end
      end

      for (bm_i = 0; bm_i < BM; bm_i = bm_i + 1) begin
        for (g_i = 0; g_i < KG; g_i = g_i + 1) begin
          pack_s = 128'd0;
          for (lane = 0; lane < 16; lane = lane + 1) begin
            s_b = e5m3_from_real(scale_map[idx_scale(bm_i, g_i, lane)]);
            pack_s = set_byte128(pack_s, lane, s_b);
          end
          ws_mem8_q[bm_i * KG + g_i] = pack_s;
        end

        for (kk = 0; kk < K; kk = kk + 1) begin
          g_i = kk >> 4;
          pack_q = 64'd0;
          pack_z = 64'd0;
          for (lane = 0; lane < 16; lane = lane + 1) begin
            m = bm_i * 16 + lane;
            scale_r = decode_e5m3(e5m3_from_real(scale_map[idx_scale(bm_i, g_i, lane)]));
            if (scale_r < 1.0e-12) scale_r = 1.0e-12;
            v = w_real[idx_w(kk, m)];
            vn = v / scale_r;

            if (dtype == 2'b10) begin
              q_n = encode_e2m1_fnuz(vn);
              z_n = 4'd0;
              w_deq[idx_w(kk, m)] = decode_e2m1_fnuz(q_n) * scale_r;
            end else begin
              z_i = 8;
              q_i = round_nearest_int(v / scale_r) + z_i;
              if (q_i < 0) q_i = 0;
              if (q_i > 15) q_i = 15;
              q_n = q_i[3:0];
              z_n = z_i[3:0];
              w_deq[idx_w(kk, m)] = (q_i - z_i) * scale_r;
            end

            pack_q = set_nibble64(pack_q, lane, q_n);
            pack_z = set_nibble64(pack_z, lane, z_n);
          end
          wq_mem8_q[bm_i * K + kk] = pack_q;
          wz_mem8_q[bm_i * K + kk] = pack_z;
        end
      end
    end
  endtask

  task automatic compute_ref;
    input integer act8;
    integer n, m, kk;
    real s;
    real x_ref_v;
    begin
      for (n = 0; n < BATCH; n = n + 1) begin
        for (m = 0; m < OUT; m = m + 1) begin
          s = 0.0;
          for (kk = 0; kk < K; kk = kk + 1) begin
            if (act8 != 0) x_ref_v = decode_e4m3_fnuz(encode_e4m3_fnuz(x_real[idx_x(n, kk)]));
            else x_ref_v = fp16_to_real(fp16_from_real_sat(x_real[idx_x(n, kk)]));
            s = s + x_ref_v * w_deq[idx_w(kk, m)];
          end
          y_ref[idx_y(n, m)] = s;
        end
      end
    end
  endtask

  task automatic report_16_nq;
    input [127:0] tag;
    real err_abs, pos_sum, neg_sum, denom, rel_err;
    integer n, m, t;
    begin
      compute_ref(0);
      err_abs = 0.0;
      pos_sum = 0.0;
      neg_sum = 0.0;
      for (n = 0; n < BATCH; n = n + 1) begin
        for (m = 0; m < OUT; m = m + 1) begin
          t = idx_y(n, m);
          err_abs = err_abs + fabs(y_hw_16_nq[t] - y_ref[t]);
          if (y_ref[t] > 0.0) pos_sum = pos_sum + y_ref[t];
          if (y_ref[t] < 0.0) neg_sum = neg_sum + y_ref[t];
          if (y_valid_16_nq[t] == 0) fail_cases = fail_cases + 1;
        end
      end
      denom = fmax3(fabs(pos_sum), fabs(neg_sum), 1.0);
      rel_err = err_abs / denom;
      $display("[ERR][tb_gemm_wnam][%0s] abs_err=%e rel_err=%e denom=%e tol=%e", tag, err_abs, rel_err, denom, TOL);
      if (!((err_abs <= TOL) || (rel_err <= TOL))) begin
        dump_case_debug(tag, 0);
        fail_cases = fail_cases + 1;
      end else if (DUMP_ALL_CASES != 0) begin
        dump_case_debug(tag, 0);
      end
    end
  endtask

  task automatic report_16_q;
    input [127:0] tag;
    real err_abs, pos_sum, neg_sum, denom, rel_err;
    integer n, m, t;
    begin
      compute_ref(0);
      err_abs = 0.0;
      pos_sum = 0.0;
      neg_sum = 0.0;
      for (n = 0; n < BATCH; n = n + 1) begin
        for (m = 0; m < OUT; m = m + 1) begin
          t = idx_y(n, m);
          err_abs = err_abs + fabs(y_hw_16_q[t] - y_ref[t]);
          if (y_ref[t] > 0.0) pos_sum = pos_sum + y_ref[t];
          if (y_ref[t] < 0.0) neg_sum = neg_sum + y_ref[t];
          if (y_valid_16_q[t] == 0) fail_cases = fail_cases + 1;
        end
      end
      denom = fmax3(fabs(pos_sum), fabs(neg_sum), 1.0);
      rel_err = err_abs / denom;
      $display("[ERR][tb_gemm_wnam][%0s] abs_err=%e rel_err=%e denom=%e tol=%e", tag, err_abs, rel_err, denom, TOL);
      if (!((err_abs <= TOL) || (rel_err <= TOL))) begin
        dump_case_debug(tag, 1);
        fail_cases = fail_cases + 1;
      end else if (DUMP_ALL_CASES != 0) begin
        dump_case_debug(tag, 1);
      end
    end
  endtask

  task automatic report_8_nq;
    input [127:0] tag;
    real err_abs, pos_sum, neg_sum, denom, rel_err;
    integer n, m, t;
    begin
      compute_ref(1);
      err_abs = 0.0;
      pos_sum = 0.0;
      neg_sum = 0.0;
      for (n = 0; n < BATCH; n = n + 1) begin
        for (m = 0; m < OUT; m = m + 1) begin
          t = idx_y(n, m);
          err_abs = err_abs + fabs(y_hw_8_nq[t] - y_ref[t]);
          if (y_ref[t] > 0.0) pos_sum = pos_sum + y_ref[t];
          if (y_ref[t] < 0.0) neg_sum = neg_sum + y_ref[t];
          if (y_valid_8_nq[t] == 0) fail_cases = fail_cases + 1;
        end
      end
      denom = fmax3(fabs(pos_sum), fabs(neg_sum), 1.0);
      rel_err = err_abs / denom;
      $display("[ERR][tb_gemm_wnam][%0s] abs_err=%e rel_err=%e denom=%e tol=%e", tag, err_abs, rel_err, denom, TOL);
      if (!((err_abs <= TOL) || (rel_err <= TOL))) begin
        dump_case_debug(tag, 2);
        fail_cases = fail_cases + 1;
      end else if (DUMP_ALL_CASES != 0) begin
        dump_case_debug(tag, 2);
      end
    end
  endtask

  task automatic report_8_q;
    input [127:0] tag;
    real err_abs, pos_sum, neg_sum, denom, rel_err, tol_case;
    integer n, m, t;
    begin
      compute_ref(1);
      tol_case = TOL_A8_Q;
      err_abs = 0.0;
      pos_sum = 0.0;
      neg_sum = 0.0;
      for (n = 0; n < BATCH; n = n + 1) begin
        for (m = 0; m < OUT; m = m + 1) begin
          t = idx_y(n, m);
          err_abs = err_abs + fabs(y_hw_8_q[t] - y_ref[t]);
          if (y_ref[t] > 0.0) pos_sum = pos_sum + y_ref[t];
          if (y_ref[t] < 0.0) neg_sum = neg_sum + y_ref[t];
          if (y_valid_8_q[t] == 0) fail_cases = fail_cases + 1;
        end
      end
      denom = fmax3(fabs(pos_sum), fabs(neg_sum), 1.0);
      rel_err = err_abs / denom;
      $display("[ERR][tb_gemm_wnam][%0s] abs_err=%e rel_err=%e denom=%e tol=%e", tag, err_abs, rel_err, denom, tol_case);
      if (!((err_abs <= tol_case) || (rel_err <= tol_case))) begin
        dump_case_debug(tag, 3);
        fail_cases = fail_cases + 1;
      end else if (DUMP_ALL_CASES != 0) begin
        dump_case_debug(tag, 3);
      end
    end
  endtask

  task automatic dump_case_debug;
    input [127:0] tag;
    input integer mode;
    integer kk, n, m, idx;
    real hw_v;
    integer valid_v;
    reg [15:0] xh16;
    reg [7:0] xh8;
    begin
      $display("[DBG][%0s] ===== FEED SNAPSHOT (all k=0..%0d) =====", tag, K-1);
      for (kk = 0; kk < K; kk = kk + 1) begin
        if (mode == 0) begin
          $display("[DBG][%0s][feed16_nq] k=%0d x_mem=%h wq=%h wz=%h ws(k=%0d)=%h",
              tag, kk, x_mem16_nq[kk], wq_mem16_nq[kk], wz_mem16_nq[kk], kk, ws_mem16_nq[kk]);
        end else if (mode == 1) begin
          $display("[DBG][%0s][feed16_q ] k=%0d x_mem=%h wq=%h wz=%h ws(g=%0d)=%h",
              tag, kk, x_mem16_q[kk], wq_mem16_q[kk], wz_mem16_q[kk], (kk >> 4), ws_mem16_q[(kk >> 4)]);
        end else if (mode == 2) begin
          $display("[DBG][%0s][feed8_nq ] k=%0d x_mem=%h wq=%h wz=%h ws(k=%0d)=%h",
              tag, kk, x_mem8_nq[kk], wq_mem8_nq[kk], wz_mem8_nq[kk], kk, ws_mem8_nq[kk]);
        end else begin
          $display("[DBG][%0s][feed8_q  ] k=%0d x_mem=%h wq=%h wz=%h ws(g=%0d)=%h",
              tag, kk, x_mem8_q[kk], wq_mem8_q[kk], wz_mem8_q[kk], (kk >> 4), ws_mem8_q[(kk >> 4)]);
        end
      end

      $display("[DBG][%0s] ===== RAW INPUT (all n,k) =====", tag);
      for (n = 0; n < BATCH; n = n + 1)
        for (kk = 0; kk < K; kk = kk + 1) begin
          xh16 = fp16_from_real_sat(x_real[idx_x(n, kk)]);
          xh8 = encode_e4m3_fnuz(x_real[idx_x(n, kk)]);
          $display("[DBG][%0s][x_real] n=%0d k=%0d val=%e x_fp16_hex=%h x_e4m3_hex=%h",
              tag, n, kk, x_real[idx_x(n, kk)], xh16, xh8);
        end

      $display("[DBG][%0s] ===== RAW/DEQ WEIGHT (all k,m) =====", tag);
      for (kk = 0; kk < K; kk = kk + 1)
        for (m = 0; m < OUT; m = m + 1)
          $display("[DBG][%0s][w] k=%0d m=%0d raw=%e deq=%e", tag, kk, m, w_real[idx_w(kk, m)], w_deq[idx_w(kk, m)]);

      $display("[DBG][%0s] ===== OUTPUT vs EXPECTED (all n,m) =====", tag);
      for (n = 0; n < BATCH; n = n + 1) begin
        for (m = 0; m < OUT; m = m + 1) begin
          idx = idx_y(n, m);
          if (mode == 0) begin
            hw_v = y_hw_16_nq[idx];
            valid_v = y_valid_16_nq[idx];
          end else if (mode == 1) begin
            hw_v = y_hw_16_q[idx];
            valid_v = y_valid_16_q[idx];
          end else if (mode == 2) begin
            hw_v = y_hw_8_nq[idx];
            valid_v = y_valid_8_nq[idx];
          end else begin
            hw_v = y_hw_8_q[idx];
            valid_v = y_valid_8_q[idx];
          end
          $display("[DBG][%0s][out] n=%0d m=%0d hw=%e exp=%e abs=%e valid=%0d",
              tag, n, m, hw_v, y_ref[idx], fabs(hw_v - y_ref[idx]), valid_v);
        end
      end
    end
  endtask

  task automatic run_case_16_nq;
    input [127:0] tag;
    begin
      case_id = case_id + 1;
      gen_random_xy();
      preload_x16_nq();
      quantize_w16_noquant();
      reset_capture_16_nq();
      en16_nq = 0;
      w_type16_nq = 2'b00;
      clear16_nq = 1;
      @(posedge clk);
      clear16_nq = 0;
      @(negedge clk);
      en16_nq = 1;
      cyc = 0;
      while (!done16_nq && cyc < 8000) begin
        @(posedge clk);
        cyc = cyc + 1;
      end
      total_cycles = total_cycles + cyc;
      en16_nq = 0;
      if (!done16_nq) begin
        $display("[FAIL][tb_gemm_wnam][%0s] timeout", tag);
        fail_cases = fail_cases + 1;
      end else begin
        report_16_nq(tag);
      end
    end
  endtask

  task automatic run_case_16_q;
    input [127:0] tag;
    input [1:0] dtype;
    begin
      case_id = case_id + 1;
      gen_random_xy();
      preload_x16_q();
      quantize_w16_quant(dtype);
      reset_capture_16_q();
      en16_q = 0;
      w_type16_q = dtype;
      clear16_q = 1;
      @(posedge clk);
      clear16_q = 0;
      @(negedge clk);
      en16_q = 1;
      cyc = 0;
      while (!done16_q && cyc < 8000) begin
        @(posedge clk);
        cyc = cyc + 1;
      end
      total_cycles = total_cycles + cyc;
      en16_q = 0;
      if (!done16_q) begin
        $display("[FAIL][tb_gemm_wnam][%0s] timeout", tag);
        fail_cases = fail_cases + 1;
      end else begin
        report_16_q(tag);
      end
    end
  endtask

  task automatic run_case_8_nq;
    input [127:0] tag;
    begin
      case_id = case_id + 1;
      gen_random_xy();
      preload_x8_nq();
      quantize_w8_noquant();
      reset_capture_8_nq();
      en8_nq = 0;
      w_type8_nq = 2'b00;
      clear8_nq = 1;
      @(posedge clk);
      clear8_nq = 0;
      @(negedge clk);
      en8_nq = 1;
      cyc = 0;
      while (!done8_nq && cyc < 8000) begin
        @(posedge clk);
        cyc = cyc + 1;
      end
      total_cycles = total_cycles + cyc;
      en8_nq = 0;
      if (!done8_nq) begin
        $display("[FAIL][tb_gemm_wnam][%0s] timeout", tag);
        fail_cases = fail_cases + 1;
      end else begin
        report_8_nq(tag);
      end
    end
  endtask

  task automatic run_case_8_q;
    input [127:0] tag;
    input [1:0] dtype;
    begin
      case_id = case_id + 1;
      gen_random_xy();
      preload_x8_q();
      quantize_w8_quant(dtype);
      reset_capture_8_q();
      en8_q = 0;
      w_type8_q = dtype;
      clear8_q = 1;
      @(posedge clk);
      clear8_q = 0;
      @(negedge clk);
      en8_q = 1;
      cyc = 0;
      while (!done8_q && cyc < 8000) begin
        @(posedge clk);
        cyc = cyc + 1;
      end
      total_cycles = total_cycles + cyc;
      en8_q = 0;
      if (!done8_q) begin
        $display("[FAIL][tb_gemm_wnam][%0s] timeout", tag);
        fail_cases = fail_cases + 1;
      end else begin
        report_8_q(tag);
      end
    end
  endtask

  gemm #(
      .ActivationWidth(16),
      .E4M3Enabled(0),
      .Int8Enabled(0),
      .E2M1Enabled(0),
      .Int4Enabled(0)
  ) dut16_nq (
      .clk(clk), .rst(rst), .en(en16_nq), .clear(clear16_nq),
      .batch_size(BATCH), .in_features(K), .out_features(OUT),
      .x_data(x_feed16_nq), .w_type(w_type16_nq), .w_quant_half(wq_feed16_nq),
      .w_zero_half(wz_feed16_nq), .w_scale_uext_ftz(ws_feed16_nq), .y_data(y_data16_nq),
      .query_n(query_n16_nq), .query_m(query_m16_nq), .query_k(query_k16_nq),
      .submit_n(submit_n16_nq), .submit_m(submit_m16_nq),
      .query_en(query_en16_nq), .submit_en(submit_en16_nq), .done(done16_nq)
  );

  gemm #(
      .ActivationWidth(16),
      .E4M3Enabled(1),
      .Int8Enabled(1),
      .E2M1Enabled(1),
      .Int4Enabled(1)
  ) dut16_q (
      .clk(clk), .rst(rst), .en(en16_q), .clear(clear16_q),
      .batch_size(BATCH), .in_features(K), .out_features(OUT),
      .x_data(x_feed16_q), .w_type(w_type16_q), .w_quant_half(wq_feed16_q),
      .w_zero_half(wz_feed16_q), .w_scale_uext_ftz(ws_feed16_q), .y_data(y_data16_q),
      .query_n(query_n16_q), .query_m(query_m16_q), .query_k(query_k16_q),
      .submit_n(submit_n16_q), .submit_m(submit_m16_q),
      .query_en(query_en16_q), .submit_en(submit_en16_q), .done(done16_q)
  );

  gemm #(
      .ActivationWidth(8),
      .E2M1Enabled(0),
      .Int4Enabled(0)
  ) dut8_nq (
      .clk(clk), .rst(rst), .en(en8_nq), .clear(clear8_nq),
      .batch_size(BATCH), .in_features(K), .out_features(OUT),
      .x_data(x_feed8_nq), .w_type(w_type8_nq), .w_quant_half(wq_feed8_nq),
      .w_zero_half(wz_feed8_nq), .w_scale_uext_ftz(ws_feed8_nq), .y_data(y_data8_nq),
      .query_n(query_n8_nq), .query_m(query_m8_nq), .query_k(query_k8_nq),
      .submit_n(submit_n8_nq), .submit_m(submit_m8_nq),
      .query_en(query_en8_nq), .submit_en(submit_en8_nq), .done(done8_nq)
  );

  gemm #(
      .ActivationWidth(8),
      .E2M1Enabled(1),
      .Int4Enabled(1)
  ) dut8_q (
      .clk(clk), .rst(rst), .en(en8_q), .clear(clear8_q),
      .batch_size(BATCH), .in_features(K), .out_features(OUT),
      .x_data(x_feed8_q), .w_type(w_type8_q), .w_quant_half(wq_feed8_q),
      .w_zero_half(wz_feed8_q), .w_scale_uext_ftz(ws_feed8_q), .y_data(y_data8_q),
      .query_n(query_n8_q), .query_m(query_m8_q), .query_k(query_k8_q),
      .submit_n(submit_n8_q), .submit_m(submit_m8_q),
      .query_en(query_en8_q), .submit_en(submit_en8_q), .done(done8_q)
  );

  always @(negedge clk) begin
    if (!rst || clear16_nq) begin
      req_valid_q16_nq <= 0; req_n_q16_nq <= 0; req_m_q16_nq <= 0; req_k_q16_nq <= 0;
    end else begin
      req_valid_q16_nq <= query_en16_nq && (query_k16_nq < K);
      if (query_en16_nq && (query_k16_nq < K)) begin
        req_n_q16_nq <= query_n16_nq;
        req_m_q16_nq <= query_m16_nq;
        req_k_q16_nq <= query_k16_nq;
      end
    end

    if (!rst || clear16_q) begin
      req_valid_q16_q <= 0; req_n_q16_q <= 0; req_m_q16_q <= 0; req_k_q16_q <= 0;
    end else begin
      req_valid_q16_q <= query_en16_q && (query_k16_q < K);
      if (query_en16_q && (query_k16_q < K)) begin
        req_n_q16_q <= query_n16_q;
        req_m_q16_q <= query_m16_q;
        req_k_q16_q <= query_k16_q;
      end
    end

    if (!rst || clear8_nq) begin
      req_valid_q8_nq <= 0; req_n_q8_nq <= 0; req_m_q8_nq <= 0; req_k_q8_nq <= 0;
    end else begin
      req_valid_q8_nq <= query_en8_nq && (query_k8_nq < K);
      if (query_en8_nq && (query_k8_nq < K)) begin
        req_n_q8_nq <= query_n8_nq;
        req_m_q8_nq <= query_m8_nq;
        req_k_q8_nq <= query_k8_nq;
      end
    end

    if (!rst || clear8_q) begin
      req_valid_q8_q <= 0; req_n_q8_q <= 0; req_m_q8_q <= 0; req_k_q8_q <= 0;
    end else begin
      req_valid_q8_q <= query_en8_q && (query_k8_q < K);
      if (query_en8_q && (query_k8_q < K)) begin
        req_n_q8_q <= query_n8_q;
        req_m_q8_q <= query_m8_q;
        req_k_q8_q <= query_k8_q;
      end
    end
  end

  always @(posedge clk) begin
    if (!rst || clear16_nq) begin
      x_feed16_nq <= 0; wq_feed16_nq <= 0; wz_feed16_nq <= 0; ws_feed16_nq <= 0;
    end else if (req_valid_q16_nq) begin
      x_feed16_nq <= x_mem16_nq[req_n_q16_nq * K + req_k_q16_nq];
      wq_feed16_nq <= wq_mem16_nq[req_m_q16_nq * K + req_k_q16_nq];
      wz_feed16_nq <= wz_mem16_nq[req_m_q16_nq * K + req_k_q16_nq];
      ws_feed16_nq <= ws_mem16_nq[req_m_q16_nq * K + req_k_q16_nq];
    end else begin
      x_feed16_nq <= 0; wq_feed16_nq <= 0; wz_feed16_nq <= 0; ws_feed16_nq <= 0;
    end

    if (!rst || clear16_q) begin
      x_feed16_q <= 0; wq_feed16_q <= 0; wz_feed16_q <= 0; ws_feed16_q <= 0;
    end else if (req_valid_q16_q) begin
      x_feed16_q <= x_mem16_q[req_n_q16_q * K + req_k_q16_q];
      wq_feed16_q <= wq_mem16_q[req_m_q16_q * K + req_k_q16_q];
      wz_feed16_q <= wz_mem16_q[req_m_q16_q * K + req_k_q16_q];
      ws_feed16_q <= ws_mem16_q[req_m_q16_q * KG + (req_k_q16_q >> 4)];
    end else begin
      x_feed16_q <= 0; wq_feed16_q <= 0; wz_feed16_q <= 0; ws_feed16_q <= 0;
    end

    if (!rst || clear8_nq) begin
      x_feed8_nq <= 0; wq_feed8_nq <= 0; wz_feed8_nq <= 0; ws_feed8_nq <= 0;
    end else if (req_valid_q8_nq) begin
      x_feed8_nq <= x_mem8_nq[req_n_q8_nq * K + req_k_q8_nq];
      wq_feed8_nq <= wq_mem8_nq[req_m_q8_nq * K + req_k_q8_nq];
      wz_feed8_nq <= wz_mem8_nq[req_m_q8_nq * K + req_k_q8_nq];
      ws_feed8_nq <= ws_mem8_nq[req_m_q8_nq * K + req_k_q8_nq];
    end else begin
      x_feed8_nq <= 0; wq_feed8_nq <= 0; wz_feed8_nq <= 0; ws_feed8_nq <= 0;
    end

    if (!rst || clear8_q) begin
      x_feed8_q <= 0; wq_feed8_q <= 0; wz_feed8_q <= 0; ws_feed8_q <= 0;
    end else if (req_valid_q8_q) begin
      x_feed8_q <= x_mem8_q[req_n_q8_q * K + req_k_q8_q];
      wq_feed8_q <= wq_mem8_q[req_m_q8_q * K + req_k_q8_q];
      wz_feed8_q <= wz_mem8_q[req_m_q8_q * K + req_k_q8_q];
      ws_feed8_q <= ws_mem8_q[req_m_q8_q * KG + (req_k_q8_q >> 4)];
    end else begin
      x_feed8_q <= 0; wq_feed8_q <= 0; wz_feed8_q <= 0; ws_feed8_q <= 0;
    end
  end

  always @(posedge clk) begin
    integer nbase;
    integer col;
    integer idx0;
    if (submit_en16_nq) begin
      nbase = submit_n16_nq * 4;
      col = submit_m16_nq;
      if (col < OUT) begin
        for (lane = 0; lane < 4; lane = lane + 1) begin
          if (nbase + lane < BATCH) begin
            idx0 = idx_y(nbase + lane, col);
            y_hw_16_nq[idx0] = fp16_to_real(get_lane16_64(y_data16_nq, lane));
            y_valid_16_nq[idx0] = 1;
          end
        end
      end
    end

    if (submit_en16_q) begin
      nbase = submit_n16_q * 4;
      col = submit_m16_q;
      if (col < OUT) begin
        for (lane = 0; lane < 4; lane = lane + 1) begin
          if (nbase + lane < BATCH) begin
            idx0 = idx_y(nbase + lane, col);
            y_hw_16_q[idx0] = fp16_to_real(get_lane16_64(y_data16_q, lane));
            y_valid_16_q[idx0] = 1;
          end
        end
      end
    end

    if (submit_en8_nq) begin
      nbase = submit_n8_nq * 4;
      col = submit_m8_nq;
      if (col < OUT) begin
        for (lane = 0; lane < 4; lane = lane + 1) begin
          if (nbase + lane < BATCH) begin
            idx0 = idx_y(nbase + lane, col);
            y_hw_8_nq[idx0] = fp16_to_real(get_lane16_64(y_data8_nq, lane));
            y_valid_8_nq[idx0] = 1;
          end
        end
      end
    end

    if (submit_en8_q) begin
      nbase = submit_n8_q * 4;
      col = submit_m8_q;
      if (col < OUT) begin
        for (lane = 0; lane < 4; lane = lane + 1) begin
          if (nbase + lane < BATCH) begin
            idx0 = idx_y(nbase + lane, col);
            y_hw_8_q[idx0] = fp16_to_real(get_lane16_64(y_data8_q, lane));
            y_valid_8_q[idx0] = 1;
          end
        end
      end
    end
  end

  initial begin
    $dumpfile("tb_gemm_wnam.vcd");
    $dumpvars(0, tb_gemm_wnam);

    clk = 0;
    rst = 0;
    en16_nq = 0; clear16_nq = 0; w_type16_nq = 0;
    en16_q = 0; clear16_q = 0; w_type16_q = 0;
    en8_nq = 0; clear8_nq = 0; w_type8_nq = 0;
    en8_q = 0; clear8_q = 0; w_type8_q = 0;

    req_valid_q16_nq = 0; req_n_q16_nq = 0; req_m_q16_nq = 0; req_k_q16_nq = 0;
    req_valid_q16_q = 0; req_n_q16_q = 0; req_m_q16_q = 0; req_k_q16_q = 0;
    req_valid_q8_nq = 0; req_n_q8_nq = 0; req_m_q8_nq = 0; req_k_q8_nq = 0;
    req_valid_q8_q = 0; req_n_q8_q = 0; req_m_q8_q = 0; req_k_q8_q = 0;

    x_feed16_nq = 0; wq_feed16_nq = 0; wz_feed16_nq = 0; ws_feed16_nq = 0;
    x_feed16_q = 0; wq_feed16_q = 0; wz_feed16_q = 0; ws_feed16_q = 0;
    x_feed8_nq = 0; wq_feed8_nq = 0; wz_feed8_nq = 0; ws_feed8_nq = 0;
    x_feed8_q = 0; wq_feed8_q = 0; wz_feed8_q = 0; ws_feed8_q = 0;

    fail_cases = 0;
    total_cycles = 0;
    case_id = 0;
    rand_seed = 32'h1badf00d;

    repeat (4) @(posedge clk);
    rst = 1;

    run_case_16_nq("act16_w_e5m10");
    run_case_16_q("act16_w_int8", 2'b01);
    run_case_16_q("act16_w_e4m3", 2'b00);
    run_case_16_q("act16_w_int4", 2'b11);
    run_case_16_q("act16_w_e2m1", 2'b10);

    run_case_8_nq("act8_w_e4m3");
    run_case_8_q("act8_w_int4", 2'b11);
    run_case_8_q("act8_w_e2m1", 2'b10);

    if (fail_cases == 0) begin
      $display("[PASS] tb_gemm_wnam passed 8 groups with abs/rel error checks, total_cycles=%0d", total_cycles);
    end else begin
      $display("[FAIL] tb_gemm_wnam fail_cases=%0d total_cycles=%0d", fail_cases, total_cycles);
      $fatal;
    end

    #20;
    $finish;
  end

  always #5 clk = ~clk;

endmodule
