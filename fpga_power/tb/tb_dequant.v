`timescale 1ns / 1ps

module tb_dequant;

  integer i;
  integer si;
  integer zi;
  integer test_count;
  integer error_count;
  integer error_fp16;
  integer error_e4m3;
  integer error_e2m1;
  integer error_int8;
  integer error_int4;
  integer error_e2m1_9;
  integer error_int4_9;

  real true_val;
  real dut_val;
  real abs_err;
  real ulp;
  reg [16:0] ref_bits;

  localparam integer ScaleN = 8;
  localparam integer Z8N = 4;
  localparam integer Z4N = 4;

  reg [15:0] scales[0:ScaleN-1];
  reg [7:0] zeros8[0:Z8N-1];
  reg [3:0] zeros4[0:Z4N-1];

  reg [15:0] fp16_in;
  wire [16:0] fp16_out;

  reg [7:0] e4m3_in;
  reg [15:0] e4m3_scale;
  wire [16:0] e4m3_out;

  reg [3:0] e2m1_in;
  reg [15:0] e2m1_scale;
  wire [16:0] e2m1_out;

  reg [7:0] int8_in;
  reg [7:0] int8_zero;
  reg [15:0] int8_scale;
  wire [16:0] int8_out;

  reg [3:0] int4_in;
  reg [3:0] int4_zero;
  reg [15:0] int4_scale;
  wire [16:0] int4_out;

  reg [3:0] e2m1_9_in;
  reg [7:0] e2m1_9_scale;
  wire [8:0] e2m1_9_out;

  reg [3:0] int4_9_in;
  reg [3:0] int4_9_zero;
  reg [7:0] int4_9_scale;
  wire [8:0] int4_9_out;

  fp16_ext u_fp16_ext (
      .in_e5m10 (fp16_in),
      .out_e6m10_ftz(fp16_out)
  );

  e4m3_ext17 u_e4m3_ext (
      .in_e4m3(e4m3_in),
      .scale_ue6m10_ftz(e4m3_scale),
      .out_e6m10_ftz(e4m3_out)
  );

  e2m1_ext17 u_e2m1_ext (
      .in_e2m1(e2m1_in),
      .scale_ue6m10_ftz(e2m1_scale),
      .out_e6m10_ftz(e2m1_out)
  );

  int8_ext17 u_int8_ext (
      .in_int8(int8_in),
      .zero_int8(int8_zero),
      .scale_ue6m10_ftz(int8_scale),
      .out_e6m10_ftz(int8_out)
  );

  int4_ext17 u_int4_ext (
      .in_int4(int4_in),
      .zero_int4(int4_zero),
      .scale_ue6m10_ftz(int4_scale),
      .out_e6m10_ftz(int4_out)
  );

    e2m1_ext9 u_e2m1_ext9 (
      .in_e2m1(e2m1_9_in),
      .scale_ue5m3_ftz(e2m1_9_scale),
      .out_e5m3_ftz(e2m1_9_out)
    );

    int4_ext9 u_int4_ext9 (
      .in_int4(int4_9_in),
      .zero_int4(int4_9_zero),
      .scale_ue5m3_ftz(int4_9_scale),
      .out_e5m3_ftz(int4_9_out)
    );

  function automatic real fabs;
    input real x;
    begin
      fabs = (x < 0.0) ? -x : x;
    end
  endfunction

  function automatic real pow2i;
    input integer exp_v;
    integer k;
    real r;
    begin
      r = 1.0;
      if (exp_v >= 0) begin
        for (k = 0; k < exp_v; k = k + 1) r = r * 2.0;
      end else begin
        for (k = 0; k < -exp_v; k = k + 1) r = r / 2.0;
      end
      pow2i = r;
    end
  endfunction

  function automatic real decode_fp17;
    input [16:0] x;
    integer exp_unbiased;
    real frac;
    begin
      if (x[15:10] == 0) begin
        decode_fp17 = 0.0;
      end else begin
        exp_unbiased = x[15:10] - 47;
        frac = 1.0 + (x[9:0] * 1.0) / 1024.0;
        decode_fp17 = (x[16] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
      end
    end
  endfunction

  function automatic real decode_ue6m10;
    input [15:0] x;
    integer exp_unbiased;
    real frac;
    begin
      if (x[15:10] == 0) begin
        decode_ue6m10 = 0.0;
      end else begin
        exp_unbiased = x[15:10] - 47;
        frac = 1.0 + (x[9:0] * 1.0) / 1024.0;
        decode_ue6m10 = pow2i(exp_unbiased) * frac;
      end
    end
  endfunction

  function automatic real decode_ue5m3;
    input [7:0] x;
    integer exp_unbiased;
    real frac;
    begin
      if (x[7:3] == 0) begin
        decode_ue5m3 = 0.0;
      end else begin
        exp_unbiased = x[7:3] - 23;
        frac = 1.0 + (x[2:0] * 1.0) / 8.0;
        decode_ue5m3 = pow2i(exp_unbiased) * frac;
      end
    end
  endfunction

  function automatic real decode_fp16_e5m10_fnuz;
    input [15:0] x;
    integer exp_field;
    integer exp_unbiased;
    real frac;
    begin
      exp_field = x[14:10];
      if (x[14:0] == 0) begin
        decode_fp16_e5m10_fnuz = 0.0;
      end else if (exp_field == 0) begin
        exp_unbiased = 1 - 15;
        frac = (x[9:0] * 1.0) / 1024.0;
        decode_fp16_e5m10_fnuz = (x[15] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
      end else begin
        exp_unbiased = exp_field - 15;
        frac = 1.0 + (x[9:0] * 1.0) / 1024.0;
        decode_fp16_e5m10_fnuz = (x[15] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
      end
    end
  endfunction

  function automatic real decode_e4m3_fnuz;
    input [7:0] x;
    integer exp_field;
    integer exp_unbiased;
    real frac;
    begin
      exp_field = x[6:3];
      if (x[6:0] == 0) begin
        decode_e4m3_fnuz = 0.0;
      end else if (exp_field == 0) begin
        exp_unbiased = 1 - 7;
        frac = (x[2:0] * 1.0) / 8.0;
        decode_e4m3_fnuz = (x[7] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
      end else begin
        exp_unbiased = exp_field - 7;
        frac = 1.0 + (x[2:0] * 1.0) / 8.0;
        decode_e4m3_fnuz = (x[7] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
      end
    end
  endfunction

  function automatic real decode_e2m1_fnuz;
    input [3:0] x;
    integer exp_field;
    integer exp_unbiased;
    real frac;
    begin
      exp_field = x[2:1];
      if (x[2:0] == 0) begin
        decode_e2m1_fnuz = 0.0;
      end else if (exp_field == 0) begin
        exp_unbiased = 1 - 1;
        frac = (x[0] * 1.0) / 2.0;
        decode_e2m1_fnuz = (x[3] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
      end else begin
        exp_unbiased = exp_field - 1;
        frac = 1.0 + (x[0] * 1.0) / 2.0;
        decode_e2m1_fnuz = (x[3] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
      end
    end
  endfunction

  function automatic integer uint8;
    input [7:0] x;
    begin
      uint8 = x;
    end
  endfunction

  function automatic integer uint4;
    input [3:0] x;
    begin
      uint4 = x;
    end
  endfunction

  function automatic real decode_fp9_e5m3;
    input [8:0] x;
    integer exp_unbiased;
    real frac;
    begin
      if (x[7:3] == 0) begin
        decode_fp9_e5m3 = 0.0;
      end else begin
        exp_unbiased = x[7:3] - 23;
        frac = 1.0 + (x[2:0] * 1.0) / 8.0;
        decode_fp9_e5m3 = (x[8] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
      end
    end
  endfunction

  function automatic [8:0] encode_fp9_e5m3_from_real;
    input real v;
    reg sign;
    real a;
    real frac;
    real mant_r;
    integer exp_unbiased;
    integer exp_field;
    integer mant_i;
    begin
      if (v == 0.0) begin
        encode_fp9_e5m3_from_real = 9'd0;
      end else begin
        sign = (v < 0.0);
        a = sign ? -v : v;
        exp_unbiased = 0;
        frac = a;
        while (frac >= 2.0) begin
          frac = frac / 2.0;
          exp_unbiased = exp_unbiased + 1;
        end
        while (frac < 1.0) begin
          frac = frac * 2.0;
          exp_unbiased = exp_unbiased - 1;
        end

        exp_field = exp_unbiased + 23;
        if (exp_field <= 0 || exp_field >= 32) begin
          encode_fp9_e5m3_from_real = 9'd0;
        end else begin
          mant_r = (frac - 1.0) * 8.0;
          mant_i = $rtoi(mant_r + 0.5);
          if (mant_i >= 8) begin
            mant_i = 0;
            exp_field = exp_field + 1;
          end
          if (exp_field >= 32) begin
            encode_fp9_e5m3_from_real = 9'd0;
          end else begin
            encode_fp9_e5m3_from_real = {sign, exp_field[4:0], mant_i[2:0]};
          end
        end
      end
    end
  endfunction

  function automatic real ulp_of_fp9_e5m3;
    input [8:0] x;
    integer exp_unbiased;
    begin
      if (x[7:3] == 0) begin
        ulp_of_fp9_e5m3 = pow2i(1 - 23 - 3);
      end else begin
        exp_unbiased = x[7:3] - 23;
        ulp_of_fp9_e5m3 = pow2i(exp_unbiased - 3);
      end
    end
  endfunction

  function automatic [16:0] encode_fp17_from_real;
    input real v;
    reg sign;
    real a;
    real frac;
    real mant_r;
    integer exp_unbiased;
    integer exp_field;
    integer mant_i;
    begin
      if (v == 0.0) begin
        encode_fp17_from_real = 17'd0;
      end else begin
        sign = (v < 0.0);
        a = sign ? -v : v;
        exp_unbiased = 0;
        frac = a;
        while (frac >= 2.0) begin
          frac = frac / 2.0;
          exp_unbiased = exp_unbiased + 1;
        end
        while (frac < 1.0) begin
          frac = frac * 2.0;
          exp_unbiased = exp_unbiased - 1;
        end

        exp_field = exp_unbiased + 47;
        if (exp_field <= 0 || exp_field >= 64) begin
          encode_fp17_from_real = 17'd0;
        end else begin
          mant_r = (frac - 1.0) * 1024.0;
          mant_i = $rtoi(mant_r + 0.5);
          if (mant_i >= 1024) begin
            mant_i = 0;
            exp_field = exp_field + 1;
          end
          if (exp_field >= 64) begin
            encode_fp17_from_real = 17'd0;
          end else begin
            encode_fp17_from_real = {sign, exp_field[5:0], mant_i[9:0]};
          end
        end
      end
    end
  endfunction

  function automatic real ulp_of_fp17;
    input [16:0] x;
    integer exp_unbiased;
    begin
      if (x[15:10] == 0) begin
        ulp_of_fp17 = pow2i(1 - 47 - 10);
      end else begin
        exp_unbiased = x[15:10] - 47;
        ulp_of_fp17  = pow2i(exp_unbiased - 10);
      end
    end
  endfunction

  task automatic check_1ulp;
    input [16:0] dut_bits;
    input real ideal;
    input [255:0] tag;
    input [31:0] in_bits;
    input integer dtype_id;
    begin
      test_count = test_count + 1;
      ref_bits = encode_fp17_from_real(ideal);
      dut_val = decode_fp17(dut_bits);
      ulp = ulp_of_fp17(ref_bits);
      abs_err = fabs(dut_val - ideal);
      if (abs_err > (ulp + 1.0e-18)) begin
        error_count = error_count + 1;
        case (dtype_id)
          0: error_fp16 = error_fp16 + 1;
          1: error_e4m3 = error_e4m3 + 1;
          2: error_e2m1 = error_e2m1 + 1;
          3: error_int8 = error_int8 + 1;
          4: error_int4 = error_int4 + 1;
          default: ;
        endcase
        if (error_count <= 40) begin
          case (dtype_id)
            0: begin
              $display(
                  "[FAIL][fp16_ext] quant=0x%04h scale=N/A zero=N/A output=0x%05h expected=0x%05h error=%e ulp=%e",
                  in_bits[15:0], dut_bits, ref_bits, abs_err, ulp);
            end
            1: begin
              $display(
                  "[FAIL][e4m3_ext] quant=0x%02h scale=0x%04h zero=N/A output=0x%05h expected=0x%05h error=%e ulp=%e",
                  in_bits[23:16], in_bits[15:0], dut_bits, ref_bits, abs_err, ulp);
            end
            2: begin
              $display(
                  "[FAIL][e2m1_ext] quant=0x%01h scale=0x%04h zero=N/A output=0x%05h expected=0x%05h error=%e ulp=%e",
                  in_bits[19:16], in_bits[15:0], dut_bits, ref_bits, abs_err, ulp);
            end
            3: begin
              $display(
              "[FAIL][int8_ext] quant=%0d(0x%02h) scale=0x%04h zero=%0d(0x%02h) output=0x%05h expected=0x%05h error=%e ulp=%e",
              uint8(in_bits[23:16]), in_bits[23:16], in_bits[15:0], uint8(in_bits[15:8]),
                  in_bits[15:8], dut_bits, ref_bits, abs_err, ulp);
            end
            4: begin
              $display(
              "[FAIL][int4_ext] quant=%0d(0x%01h) scale=0x%04h zero=%0d(0x%01h) output=0x%05h expected=0x%05h error=%e ulp=%e",
              uint4(in_bits[23:20]), in_bits[23:20], in_bits[15:0], uint4(in_bits[19:16]),
                  in_bits[19:16], dut_bits, ref_bits, abs_err, ulp);
            end
            default: begin
              $display(
                  "[FAIL][%0s] quant=0x%0h scale=N/A zero=N/A output=0x%05h expected=0x%05h error=%e ulp=%e",
                  tag, in_bits, dut_bits, ref_bits, abs_err, ulp);
            end
          endcase
          $display("             ideal=%e output_real=%e", ideal, dut_val);
        end
      end
    end
  endtask

  task automatic check_abs_tol;
    input [16:0] dut_bits;
    input real ideal;
    input real tol;
    input [255:0] tag;
    input [31:0] in_bits;
    input integer dtype_id;
    real rel_err;
    begin
      test_count = test_count + 1;
      dut_val = decode_fp17(dut_bits);
      abs_err = fabs(dut_val - ideal);
      rel_err = (fabs(ideal) > 1.0e-12) ? (abs_err / fabs(ideal)) : abs_err;
      if ((abs_err > tol) && (rel_err > tol)) begin
        error_count = error_count + 1;
        case (dtype_id)
          0: error_fp16 = error_fp16 + 1;
          1: error_e4m3 = error_e4m3 + 1;
          2: error_e2m1 = error_e2m1 + 1;
          3: error_int8 = error_int8 + 1;
          4: error_int4 = error_int4 + 1;
          default: ;
        endcase
        if (error_count <= 40) begin
          case (dtype_id)
            1: begin
              $display(
                  "[FAIL][e4m3_ext] quant=0x%02h scale=0x%04h zero=N/A output=0x%05h abs_err=%e rel_err=%e tol=%e",
                  in_bits[23:16], in_bits[15:0], dut_bits, abs_err, rel_err, tol);
            end
            default: begin
              $display(
                  "[FAIL][%0s] quant=0x%0h output=0x%05h abs_err=%e rel_err=%e tol=%e",
                  tag, in_bits, dut_bits, abs_err, rel_err, tol);
            end
          endcase
          $display("             ideal=%e output_real=%e", ideal, dut_val);
        end
      end
    end
  endtask

  task automatic check_1ulp_fp9;
    input [8:0] dut_bits;
    input real ideal;
    input [255:0] tag;
    input [31:0] in_bits;
    input integer dtype_id;
    reg [8:0] ref_bits9;
    real dut_val9;
    real ulp9;
    begin
      test_count = test_count + 1;
      ref_bits9 = encode_fp9_e5m3_from_real(ideal);
      dut_val9 = decode_fp9_e5m3(dut_bits);
      ulp9 = ulp_of_fp9_e5m3(ref_bits9);
      abs_err = fabs(dut_val9 - ideal);
      if (abs_err > (ulp9 + 1.0e-18)) begin
        error_count = error_count + 1;
        case (dtype_id)
          5: error_e2m1_9 = error_e2m1_9 + 1;
          6: error_int4_9 = error_int4_9 + 1;
          default: ;
        endcase
        if (error_count <= 40) begin
          case (dtype_id)
            5: $display(
                "[FAIL][e2m1_ext9] quant=0x%01h scale=0x%02h output=0x%03h expected=0x%03h error=%e ulp=%e",
                in_bits[19:16], in_bits[15:8], dut_bits, ref_bits9, abs_err, ulp9);
            6: $display(
                "[FAIL][int4_ext9] quant=0x%01h scale=0x%02h zero=0x%01h output=0x%03h expected=0x%03h error=%e ulp=%e",
                in_bits[23:20], in_bits[15:8], in_bits[19:16], dut_bits, ref_bits9, abs_err, ulp9);
            default: $display(
                "[FAIL][%0s] output=0x%03h expected=0x%03h error=%e ulp=%e",
                tag, dut_bits, ref_bits9, abs_err, ulp9);
          endcase
          $display("             ideal=%e output_real=%e", ideal, dut_val9);
        end
      end
    end
  endtask

  initial begin
    scales[0] = 16'h0000;
    scales[1] = 16'hbc00;  // 1.0
    scales[2] = 16'hb800;  // 0.5
    scales[3] = 16'hc000;  // 2.0
    scales[4] = 16'hc3ca;
    scales[5] = 16'hbe00;  // 1.5
    scales[6] = 16'hac00;  // 2^-4
    scales[7] = 16'hdc00;  // 256, avoid overflow-undefined combos

    zeros8[0] = 8'd0;
    zeros8[1] = 8'd64;
    zeros8[2] = 8'd128;
    zeros8[3] = 8'd255;

    zeros4[0] = 4'd0;
    zeros4[1] = 4'd4;
    zeros4[2] = 4'd8;
    zeros4[3] = 4'd15;

    fp16_in = 16'd0;
    e4m3_in = 8'd0;
    e4m3_scale = 16'd0;
    e2m1_in = 4'd0;
    e2m1_scale = 16'd0;
    int8_in = 8'd0;
    int8_zero = 8'd0;
    int8_scale = 16'd0;
    int4_in = 4'd0;
    int4_zero = 4'd0;
    int4_scale = 16'd0;
    e2m1_9_in = 4'd0;
    e2m1_9_scale = 8'd0;
    int4_9_in = 4'd0;
    int4_9_zero = 4'd0;
    int4_9_scale = 8'd0;

    test_count = 0;
    error_count = 0;
    error_fp16 = 0;
    error_e4m3 = 0;
    error_e2m1 = 0;
    error_int8 = 0;
    error_int4 = 0;
    error_e2m1_9 = 0;
    error_int4_9 = 0;

    #1;

    $display("[TB] fp16_ext exhaustive check (e5m10fnuz -> fp17)");
    for (i = 0; i < 65536; i = i + 1) begin
      fp16_in = i[15:0];
      #1;
      true_val = decode_fp16_e5m10_fnuz(fp16_in);
      check_1ulp(fp16_out, true_val, "fp16_ext", fp16_in, 0);
    end

    $display("[TB] e2m1_ext exhaustive check per scale (e2m1fnuz * scale -> fp17)");
    for (si = 0; si < ScaleN; si = si + 1) begin
      e2m1_scale = scales[si];
      for (i = 0; i < 16; i = i + 1) begin
        e2m1_in = i[3:0];
        #1;
        true_val = decode_e2m1_fnuz(e2m1_in) * decode_ue6m10(e2m1_scale);
        check_1ulp(e2m1_out, true_val, "e2m1_ext", {12'd0, e2m1_in, e2m1_scale}, 2);
      end
    end
    $display("[TB] int4_ext exhaustive check per scale/zero");
    for (si = 0; si < ScaleN; si = si + 1) begin
      int4_scale = scales[si];
      for (zi = 0; zi < Z4N; zi = zi + 1) begin
        int4_zero = zeros4[zi];
        for (i = 0; i < 16; i = i + 1) begin
          int4_in = i[3:0];
          #1;
          true_val = (uint4(int4_in) - uint4(int4_zero)) * decode_ue6m10(int4_scale);
          check_1ulp(int4_out, true_val, "int4_ext", {12'd0, int4_in, int4_zero, int4_scale}, 4);
        end
      end
    end

    $display("[TB] e4m3_ext exhaustive check per scale (e4m3fnuz * scale -> fp17)");
    for (si = 0; si < ScaleN; si = si + 1) begin
      e4m3_scale = scales[si];
      for (i = 0; i < 256; i = i + 1) begin
        e4m3_in = i[7:0];
        #1;
        true_val = decode_e4m3_fnuz(e4m3_in) * decode_ue6m10(e4m3_scale);
        check_abs_tol(e4m3_out, true_val, 1.0e-1, "e4m3_ext", {16'd0, e4m3_in, e4m3_scale}, 1);
      end
    end

    $display("[TB] int8_ext exhaustive check per scale/zero");
    for (si = 0; si < ScaleN; si = si + 1) begin
      int8_scale = scales[si];
      for (zi = 0; zi < Z8N; zi = zi + 1) begin
        int8_zero = zeros8[zi];
        for (i = 0; i < 256; i = i + 1) begin
          int8_in = i[7:0];
          #1;
          true_val = (uint8(int8_in) - uint8(int8_zero)) * decode_ue6m10(int8_scale);
          check_1ulp(int8_out, true_val, "int8_ext", {8'd0, int8_in, int8_zero, int8_scale}, 3);
        end
      end
    end

    $display("[TB] e2m1_ext9 exhaustive check per scale (bit-accurate)");
    for (si = 0; si < ScaleN; si = si + 1) begin
      e2m1_9_scale = scales[si][7:0];
      for (i = 0; i < 16; i = i + 1) begin
        e2m1_9_in = i[3:0];
        #1;
        true_val = decode_e2m1_fnuz(e2m1_9_in) * decode_ue5m3(e2m1_9_scale);
        check_1ulp_fp9(e2m1_9_out, true_val, "e2m1_ext9", {12'd0, e2m1_9_in, e2m1_9_scale, 8'd0}, 5);
      end
    end

    $display("[TB] int4_ext9 exhaustive check per scale/zero (bit-accurate)");
    for (si = 0; si < ScaleN; si = si + 1) begin
      int4_9_scale = scales[si][7:0];
      for (zi = 0; zi < Z4N; zi = zi + 1) begin
        int4_9_zero = zeros4[zi];
        for (i = 0; i < 16; i = i + 1) begin
          int4_9_in = i[3:0];
          #1;
          true_val = (uint4(int4_9_in) - uint4(int4_9_zero)) * decode_ue5m3(int4_9_scale);
          check_1ulp_fp9(int4_9_out, true_val, "int4_ext9", {8'd0, int4_9_in, int4_9_zero, int4_9_scale, 8'd0}, 6);
        end
      end
    end

    if (error_count == 0) begin
      $display("[PASS] tb_dequant passed. tests=%0d", test_count);
    end else begin
      $display("[FAIL] tb_dequant failed. tests=%0d errors=%0d", test_count, error_count);
    end
    $display("[FAIL_CNT] fp16_ext=%0d e4m3_ext=%0d e2m1_ext=%0d int8_ext=%0d int4_ext=%0d e2m1_ext9=%0d int4_ext9=%0d",
         error_fp16, error_e4m3, error_e2m1, error_int8, error_int4, error_e2m1_9,
         error_int4_9);

    #10;
    $finish;
  end

endmodule
