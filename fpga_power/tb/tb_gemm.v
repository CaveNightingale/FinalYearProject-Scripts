`timescale 1ns / 1ps

module tb_gemm;

  localparam integer BATCH = 8;
  localparam integer K = 64;
  localparam integer OUT = 32;
  localparam integer BN = BATCH / 4;
  localparam integer BM = OUT / 16;
  localparam integer KG = K / 16;
  localparam real N_VAR = 0.02;
  localparam real N_STD = 0.1414213562373095;
  localparam real TOL = 1.0e-3;

  integer i;
  integer j;
  integer k;
  integer g;
  integer lane;
  integer cyc;
  integer fail_cases;
  integer cap_nbase;
  integer cap_col;
  integer cap_idx0;
  integer trace_fd;
  integer trace_cycle;
  integer trace_case_id;
  integer total_cycles;
  integer rand_seed;
  integer fail_snapshot_taken;

  reg trace_active;

  real x_real[0:BATCH*K-1];
  real w_real[0:K*OUT-1];
  real w_deq[0:K*OUT-1];
  real y_ref[0:BATCH*OUT-1];
  real y_hw[0:BATCH*OUT-1];
  integer y_valid[0:BATCH*OUT-1];
  real scale_map[0:BM*KG*16-1];

  reg clk;
  reg rst;
  reg en;
  reg clear;

  reg [15:0] batch_size;
  reg [23:0] in_features;
  reg [15:0] out_features;
  reg [1:0] w_type;

  wire [63:0] x_data;
  wire [127:0] w_quant_8bit;
  wire [127:0] w_zero_8bit;
  wire [255:0] w_scale_e6m10_ftz;

  wire [63:0] y_data;
  wire [15:0] query_n;
  wire [15:0] query_m;
  wire [15:0] query_k;
  wire [15:0] submit_n;
  wire [15:0] submit_m;
  wire query_en;
  wire submit_en;
  wire done;

  reg x_sram_en;
  reg x_sram_we;
  reg [15:0] x_sram_addr;
  reg [63:0] x_sram_din;
  wire [63:0] x_sram_dout;

  reg wq_sram_en;
  reg wq_sram_we;
  reg [15:0] wq_sram_addr;
  reg [127:0] wq_sram_din;
  wire [127:0] wq_sram_dout;

  reg wz_sram_en;
  reg wz_sram_we;
  reg [15:0] wz_sram_addr;
  reg [127:0] wz_sram_din;
  wire [127:0] wz_sram_dout;

  reg ws_sram_en;
  reg ws_sram_we;
  reg [15:0] ws_sram_addr;
  reg [255:0] ws_sram_din;
  wire [255:0] ws_sram_dout;

  reg y_sram_en;
  reg y_sram_we;
  reg [15:0] y_sram_addr;
  reg [63:0] y_sram_din;
  wire [63:0] y_sram_dout;

  reg [63:0] x_feed;
  reg [127:0] wq_feed;
  reg [127:0] wz_feed;
  reg [255:0] ws_feed;

  reg req_valid_q;
  reg [15:0] req_n_q;
  reg [15:0] req_m_q;
  reg [15:0] req_k_q;

  reg [63:0] x_model_mem [0:65535];
  reg [127:0] wq_model_mem [0:65535];
  reg [127:0] wz_model_mem [0:65535];
  reg [255:0] ws_model_mem [0:65535];

  gemm dut (
      .clk(clk),
      .rst(rst),
      .en(en),
      .clear(clear),
      .batch_size(batch_size),
      .in_features(in_features),
      .out_features(out_features),
      .x_data(x_data),
      .w_type(w_type),
      .w_quant_half(w_quant_8bit),
      .w_zero_half(w_zero_8bit),
      .w_scale_uext_ftz(w_scale_e6m10_ftz),
      .y_data(y_data),
      .query_n(query_n),
      .query_m(query_m),
      .query_k(query_k),
      .submit_n(submit_n),
      .submit_m(submit_m),
      .query_en(query_en),
      .submit_en(submit_en),
      .done(done)
  );

  sram #(.DataWidth(64), .DataCount(65536)) x_sram (
      .clk(clk), .en(x_sram_en), .we(x_sram_we), .addr(x_sram_addr), .din(x_sram_din), .dout(x_sram_dout)
  );
  sram #(.DataWidth(128), .DataCount(65536)) wq_sram (
      .clk(clk), .en(wq_sram_en), .we(wq_sram_we), .addr(wq_sram_addr), .din(wq_sram_din), .dout(wq_sram_dout)
  );
  sram #(.DataWidth(128), .DataCount(65536)) wz_sram (
      .clk(clk), .en(wz_sram_en), .we(wz_sram_we), .addr(wz_sram_addr), .din(wz_sram_din), .dout(wz_sram_dout)
  );
  sram #(.DataWidth(256), .DataCount(65536)) ws_sram (
      .clk(clk), .en(ws_sram_en), .we(ws_sram_we), .addr(ws_sram_addr), .din(ws_sram_din), .dout(ws_sram_dout)
  );
  sram #(.DataWidth(64), .DataCount(65536)) y_sram (
      .clk(clk), .en(y_sram_en), .we(y_sram_we), .addr(y_sram_addr), .din(y_sram_din), .dout(y_sram_dout)
  );

  assign x_data = x_feed;
  assign w_quant_8bit = wq_feed;
  assign w_zero_8bit = wz_feed;
  assign w_scale_e6m10_ftz = ws_feed;

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
      case (idx)
        0: t[7:0] = b;
        1: t[15:8] = b;
        2: t[23:16] = b;
        3: t[31:24] = b;
        4: t[39:32] = b;
        5: t[47:40] = b;
        6: t[55:48] = b;
        7: t[63:56] = b;
        8: t[71:64] = b;
        9: t[79:72] = b;
        10: t[87:80] = b;
        11: t[95:88] = b;
        12: t[103:96] = b;
        13: t[111:104] = b;
        14: t[119:112] = b;
        15: t[127:120] = b;
        default: ;
      endcase
      set_byte128 = t;
    end
  endfunction

  function automatic [255:0] set_half256;
    input [255:0] vec;
    input integer idx;
    input [15:0] h;
    reg [255:0] t;
    begin
      t = vec;
      case (idx)
        0: t[15:0] = h;
        1: t[31:16] = h;
        2: t[47:32] = h;
        3: t[63:48] = h;
        4: t[79:64] = h;
        5: t[95:80] = h;
        6: t[111:96] = h;
        7: t[127:112] = h;
        8: t[143:128] = h;
        9: t[159:144] = h;
        10: t[175:160] = h;
        11: t[191:176] = h;
        12: t[207:192] = h;
        13: t[223:208] = h;
        14: t[239:224] = h;
        15: t[255:240] = h;
        default: ;
      endcase
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

  function automatic [7:0] get_byte128;
    input [127:0] v;
    input integer l;
    begin
      case (l)
        0: get_byte128 = v[7:0];
        1: get_byte128 = v[15:8];
        2: get_byte128 = v[23:16];
        3: get_byte128 = v[31:24];
        4: get_byte128 = v[39:32];
        5: get_byte128 = v[47:40];
        6: get_byte128 = v[55:48];
        7: get_byte128 = v[63:56];
        8: get_byte128 = v[71:64];
        9: get_byte128 = v[79:72];
        10: get_byte128 = v[87:80];
        11: get_byte128 = v[95:88];
        12: get_byte128 = v[103:96];
        13: get_byte128 = v[111:104];
        14: get_byte128 = v[119:112];
        default: get_byte128 = v[127:120];
      endcase
    end
  endfunction

  function automatic [15:0] get_half16_256;
    input [255:0] v;
    input integer l;
    begin
      case (l)
        0: get_half16_256 = v[15:0];
        1: get_half16_256 = v[31:16];
        2: get_half16_256 = v[47:32];
        3: get_half16_256 = v[63:48];
        4: get_half16_256 = v[79:64];
        5: get_half16_256 = v[95:80];
        6: get_half16_256 = v[111:96];
        7: get_half16_256 = v[127:112];
        8: get_half16_256 = v[143:128];
        9: get_half16_256 = v[159:144];
        10: get_half16_256 = v[175:160];
        11: get_half16_256 = v[191:176];
        12: get_half16_256 = v[207:192];
        13: get_half16_256 = v[223:208];
        14: get_half16_256 = v[239:224];
        default: get_half16_256 = v[255:240];
      endcase
    end
  endfunction

  task automatic wr_x;
    input [15:0] addr;
    input [63:0] data;
    begin
      x_model_mem[addr] = data;
      @(negedge clk);
      x_sram_en = 1'b1;
      x_sram_we = 1'b1;
      x_sram_addr = addr;
      x_sram_din = data;
      @(posedge clk);
    end
  endtask

  task automatic wr_wq;
    input [15:0] addr;
    input [127:0] data;
    begin
      wq_model_mem[addr] = data;
      @(negedge clk);
      wq_sram_en = 1'b1;
      wq_sram_we = 1'b1;
      wq_sram_addr = addr;
      wq_sram_din = data;
      @(posedge clk);
    end
  endtask

  task automatic wr_wz;
    input [15:0] addr;
    input [127:0] data;
    begin
      wz_model_mem[addr] = data;
      @(negedge clk);
      wz_sram_en = 1'b1;
      wz_sram_we = 1'b1;
      wz_sram_addr = addr;
      wz_sram_din = data;
      @(posedge clk);
    end
  endtask

  task automatic wr_ws;
    input [15:0] addr;
    input [255:0] data;
    begin
      ws_model_mem[addr] = data;
      @(negedge clk);
      ws_sram_en = 1'b1;
      ws_sram_we = 1'b1;
      ws_sram_addr = addr;
      ws_sram_din = data;
      @(posedge clk);
    end
  endtask

  task automatic gen_random_xy;
    integer n;
    integer m;
    integer kk;
    integer hx;
    integer hw;
    begin
      for (n = 0; n < BATCH; n = n + 1) begin
        for (kk = 0; kk < K; kk = kk + 1) begin
          hx = (n * 131 + kk * 17 + 23) % 101;
          x_real[idx_x(n, kk)] = (hx - 50) * 0.02;
        end
      end

      for (kk = 0; kk < K; kk = kk + 1) begin
        for (m = 0; m < OUT; m = m + 1) begin
          hw = (kk * 29 + m * 37 + 11) % 127;
          w_real[idx_w(kk, m)] = (hw - 63) * 0.015625;
        end
      end
    end
  endtask

  task automatic preload_x_to_sram;
    integer bn_i;
    integer kk;
    integer nbase;
    reg [63:0] packed_1;
    begin
      for (bn_i = 0; bn_i < BN; bn_i = bn_i + 1) begin
        nbase = bn_i * 4;
        for (kk = 0; kk < K; kk = kk + 1) begin
          packed_1 = {
            fp16_from_real_sat(x_real[idx_x(nbase+3, kk)]),
            fp16_from_real_sat(x_real[idx_x(nbase+2, kk)]),
            fp16_from_real_sat(x_real[idx_x(nbase+1, kk)]),
            fp16_from_real_sat(x_real[idx_x(nbase+0, kk)])
          };
          wr_x(bn_i * K + kk, packed_1);
        end
      end
    end
  endtask

  task automatic quantize_and_preload_w;
    input [1:0] dtype;
    integer bm_i;
    integer g_i;
    integer kk;
    integer k0;
    integer m;
    real max_abs;
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
            if (max_abs < 1.0e-8) max_abs = 1.0e-8;
            scale_map[idx_scale(bm_i, g_i, lane)] = max_abs;
          end
        end
      end

      for (bm_i = 0; bm_i < BM; bm_i = bm_i + 1) begin
        for (g_i = 0; g_i < KG; g_i = g_i + 1) begin
          pack_s = 256'd0;
          for (lane = 0; lane < 16; lane = lane + 1) begin
            scale_r = scale_map[idx_scale(bm_i, g_i, lane)];
            s_h = e6m10_from_real(scale_r);
            pack_s = set_half256(pack_s, lane, s_h);
          end
          wr_ws(bm_i * KG + g_i, pack_s);
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
              q_i = $rtoi(vn * 127.0 + z_i + 0.5);
              if (q_i < 0) q_i = 0;
              if (q_i > 255) q_i = 255;
              q_b = q_i[7:0];
              z_b = z_i[7:0];
              w_deq[idx_w(kk, m)] = (q_i - z_i) * scale_r;
            end else begin
              z_i = 8;
              q_i = $rtoi(vn * 7.0 + z_i + 0.5);
              if (q_i < 0) q_i = 0;
              if (q_i > 15) q_i = 15;
              q_b = q_i[7:0];
              z_b = z_i[7:0];
              w_deq[idx_w(kk, m)] = (q_i - z_i) * scale_r;
            end

            pack_q = set_byte128(pack_q, lane, q_b);
            pack_z = set_byte128(pack_z, lane, z_b);
          end
          wr_wq(bm_i * K + kk, pack_q);
          wr_wz(bm_i * K + kk, pack_z);
        end
      end
    end
  endtask

  task automatic reset_y_capture;
    integer t;
    begin
      for (t = 0; t < BATCH * OUT; t = t + 1) begin
        y_hw[t] = 0.0;
        y_valid[t] = 0;
      end
    end
  endtask

  task automatic compute_ref;
    integer n;
    integer m;
    integer kk;
    real s;
    real x_q;
    begin
      for (n = 0; n < BATCH; n = n + 1) begin
        for (m = 0; m < OUT; m = m + 1) begin
          s = 0.0;
          for (kk = 0; kk < K; kk = kk + 1) begin
            x_q = fp16_to_real(fp16_from_real_sat(x_real[idx_x(n, kk)]));
            s = s + x_q * w_deq[idx_w(kk, m)];
          end
          y_ref[idx_y(n, m)] = fp16_to_real(fp16_from_real_sat(s));
        end
      end
    end
  endtask

  task automatic report_and_assert;
    input [1:0] dtype;
    real err_abs;
    real pos_sum;
    real neg_sum;
    real denom;
    real rel_err;
    real elem_abs;
    real elem_rel;
    integer n;
    integer m;
    integer t;
    integer bad_elem_count;
    integer snap_fd;
    begin
      compute_ref();
      err_abs = 0.0;
      pos_sum = 0.0;
      neg_sum = 0.0;
      bad_elem_count = 0;

      for (n = 0; n < BATCH; n = n + 1) begin
        for (m = 0; m < OUT; m = m + 1) begin
          t = idx_y(n, m);
          err_abs = err_abs + fabs(y_hw[t] - y_ref[t]);
          if (y_ref[t] > 0.0) pos_sum = pos_sum + y_ref[t];
          if (y_ref[t] < 0.0) neg_sum = neg_sum + y_ref[t];
        end
      end

      denom = fmax3(fabs(pos_sum), fabs(neg_sum), 1.0);
      rel_err = err_abs / denom;
      $display("[ERR][dtype=%0d] abs_err=%e rel_err=%e denom=%e tol=%e", dtype, err_abs, rel_err, denom, TOL);

      if (!((err_abs <= TOL) || (rel_err <= TOL))) begin
        $display("[FAIL][tb_gemm] dtype=%0d abs_err=%e rel_err=%e (tol=%e)", dtype, err_abs, rel_err, TOL);
        for (n = 0; n < BATCH; n = n + 1) begin
          for (m = 0; m < OUT; m = m + 1) begin
            t = idx_y(n, m);
            elem_abs = fabs(y_hw[t] - y_ref[t]);
            elem_rel = elem_abs / denom;
            if (elem_abs > 0.0 || !y_valid[t]) begin
              bad_elem_count = bad_elem_count + 1;
              if (bad_elem_count <= 64) begin
                $display(
                    "[BAD_ELEM][dtype=%0d] n=%0d m=%0d hw=%e ref=%e abs_err=%e norm_err=%e valid=%0d",
                    dtype, n, m, y_hw[t], y_ref[t], elem_abs, elem_rel, y_valid[t]
                );
              end
            end
          end
        end
        $display("[BAD_ELEM_CNT][dtype=%0d] count=%0d", dtype, bad_elem_count);
        if (!fail_snapshot_taken) begin
          snap_fd = $fopen("tb_gemm_fail_snapshot.txt", "w");
          if (snap_fd != 0) begin
            $fdisplay(snap_fd, "dtype=%0d", dtype);
            $fdisplay(snap_fd, "case_id=%0d", trace_case_id);
            $fdisplay(snap_fd, "seed=%0d", rand_seed);
            $fdisplay(snap_fd, "BATCH=%0d K=%0d OUT=%0d", BATCH, K, OUT);
            for (n = 0; n < BATCH; n = n + 1) begin
              for (k = 0; k < K; k = k + 1) begin
                $fdisplay(snap_fd, "X %0d %0d %.17e", n, k, x_real[idx_x(n, k)]);
              end
            end
            for (k = 0; k < K; k = k + 1) begin
              for (m = 0; m < OUT; m = m + 1) begin
                $fdisplay(snap_fd, "W %0d %0d %.17e", k, m, w_real[idx_w(k, m)]);
              end
            end
            $fclose(snap_fd);
            $display("[INFO][tb_gemm] wrote reproducible fail snapshot to tb_gemm_fail_snapshot.txt");
          end
          fail_snapshot_taken = 1;
        end
        fail_cases = fail_cases + 1;
      end
    end
  endtask

  task automatic run_case;
    input [1:0] dtype;
    begin
      gen_random_xy();
      preload_x_to_sram();
      quantize_and_preload_w(dtype);
      reset_y_capture();

      en = 1'b0;
      w_type = dtype;
      trace_case_id = trace_case_id + 1;
      trace_cycle = 0;
      trace_active = 1'b1;
      $fdisplay(trace_fd, "=== BEGIN_CASE case=%0d dtype=%0d time=%0t ===", trace_case_id, dtype, $time);
      clear = 1'b1;
      @(posedge clk);
      clear = 1'b0;

      // Prime one half-cycle so the external one-cycle-latency memory model
      // can capture the initial q=(n0,m0,k0) request before compute consumes data.
      @(negedge clk);
      en = 1'b1;

      cyc = 0;
      while (!done && cyc < 6000) begin
        @(posedge clk);
        cyc = cyc + 1;
      end
      total_cycles = total_cycles + cyc;

      if (!done) begin
        $display("[FAIL][tb_gemm] timeout dtype=%0d cycles=%0d", dtype, cyc);
        fail_cases = fail_cases + 1;
      end else begin
        report_and_assert(dtype);
      end

      $fdisplay(trace_fd, "=== END_CASE case=%0d dtype=%0d cycles=%0d done=%0d time=%0t ===", trace_case_id, dtype, trace_cycle, done, $time);
      en = 1'b0;
      trace_active = 1'b0;
    end
  endtask

  always @(negedge clk) begin
    if (!rst || clear) begin
      req_valid_q <= 1'b0;
      req_n_q <= 16'd0;
      req_m_q <= 16'd0;
      req_k_q <= 16'd0;
    end else begin
      req_valid_q <= query_en && (query_k < K);
      if (query_en && (query_k < K)) begin
        req_n_q <= query_n;
        req_m_q <= query_m;
        req_k_q <= query_k;
      end
    end
  end

  always @(posedge clk) begin
    if (!rst || clear) begin
      x_feed <= 64'd0;
      wq_feed <= 128'd0;
      wz_feed <= 128'd0;
      ws_feed <= 256'd0;
    end else if (req_valid_q) begin
      x_feed <= x_model_mem[req_n_q * K + req_k_q];
      wq_feed <= wq_model_mem[req_m_q * K + req_k_q];
      wz_feed <= wz_model_mem[req_m_q * K + req_k_q];
      ws_feed <= ws_model_mem[req_m_q * KG + (req_k_q >> 4)];
    end else begin
      x_feed <= 64'd0;
      wq_feed <= 128'd0;
      wz_feed <= 128'd0;
      ws_feed <= 256'd0;
    end
  end

  always @(*) begin
    x_sram_en = 1'b1;
    x_sram_we = 1'b0;
    x_sram_addr = query_n * K + query_k;
    x_sram_din = 64'd0;

    wq_sram_en = 1'b1;
    wq_sram_we = 1'b0;
    wq_sram_addr = query_m * K + query_k;
    wq_sram_din = 128'd0;

    wz_sram_en = 1'b1;
    wz_sram_we = 1'b0;
    wz_sram_addr = query_m * K + query_k;
    wz_sram_din = 128'd0;

    ws_sram_en = 1'b1;
    ws_sram_we = 1'b0;
    ws_sram_addr = query_m * KG + query_k[15:4];
    ws_sram_din = 256'd0;

    y_sram_en = submit_en;
    y_sram_we = submit_en;
    y_sram_addr = submit_n * OUT + submit_m;
    y_sram_din = y_data;
  end

  always @(posedge clk) begin
    if (submit_en) begin
      cap_nbase = submit_n * 4;
      cap_col = submit_m;
      if (cap_col < OUT) begin
        for (lane = 0; lane < 4; lane = lane + 1) begin
          if (cap_nbase + lane < BATCH) begin
            cap_idx0 = idx_y(cap_nbase + lane, cap_col);
            y_hw[cap_idx0] = fp16_to_real(get_lane16_64(y_data, lane));
            y_valid[cap_idx0] = 1;
          end
        end
      end
    end
  end

  initial begin
    $dumpfile("tb_gemm.vcd");
    $dumpvars(0, tb_gemm);
    trace_fd = $fopen("tb_gemm_trace.txt", "w");
    if (trace_fd == 0) begin
      $display("[FAIL][tb_gemm] cannot open tb_gemm_trace.txt");
      $fatal;
    end
    $fdisplay(trace_fd, "tb_gemm cycle trace");
    $fdisplay(trace_fd, "fields: cycle time rst clear en dtype query_en query_n query_m query_k x_addr wq_addr wz_addr ws_addr x_in wq_in wz_in ws_in cur_bn cur_bm cur_k qdelay y_row submit_en submit_n submit_m y_addr y_out done");

    clk = 1'b0;
    rst = 1'b0;
    en = 1'b0;
    clear = 1'b0;
    batch_size = BATCH;
    in_features = K;
    out_features = OUT;
    w_type = 2'b00;
    fail_cases = 0;
    total_cycles = 0;
    trace_cycle = 0;
    trace_case_id = 0;
    trace_active = 1'b0;
    rand_seed = 32'h1badf00d;
    fail_snapshot_taken = 0;
    req_valid_q = 1'b0;
    req_n_q = 16'd0;
    req_m_q = 16'd0;
    req_k_q = 16'd0;
    x_feed = 64'd0;
    wq_feed = 128'd0;
    wz_feed = 128'd0;
    ws_feed = 256'd0;

    repeat (4) @(posedge clk);
    rst = 1'b1;

    run_case(2'b10);  // e2m1
    run_case(2'b11);  // int4
    run_case(2'b00);  // e4m3
    run_case(2'b01);  // int8

    if (fail_cases == 0) begin
      $display("[PASS] tb_gemm passed all dtype cases with tol=%e total_cycles=%0d", TOL, total_cycles);
    end else begin
      $display("[FAIL] tb_gemm fail_cases=%0d total_cycles=%0d", fail_cases, total_cycles);
      $fatal;
    end

    #20;
    $fclose(trace_fd);
    $finish;
  end

  always #5 clk = ~clk;

endmodule
