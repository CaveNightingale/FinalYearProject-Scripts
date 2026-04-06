`timescale 1ns / 1ps
module tb_add;

  reg [31:0] a;
  reg [31:0] b;
  wire [31:0] sum;

  integer total_cnt;
  integer fail_cnt;
  integer i;
  integer seed;

  real a_real;
  real b_real;
  real verilog_sum;
  real c_sum;
  real abs_err;
  real abs_tol;
  real abs_ref;
  real rel_err;

  add dut (
      .a  (a),
      .b  (b),
      .sum(sum)
  );

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

  task automatic run_case;
    input [31:0] a_bits;
    input [31:0] b_bits;
    real denom;
    begin
      a = a_bits;
      b = b_bits;
      #1;

      a_real = fp32_to_real(a_bits);
      b_real = fp32_to_real(b_bits);
      verilog_sum = fp32_to_real(sum);
      c_sum = a_real + b_real;

      denom = rmax(rabs(c_sum), 1.0);
      abs_err = rabs(verilog_sum - c_sum);
      abs_ref = rmax(rmax(rabs(a_real), rabs(b_real)), rmax(rabs(c_sum), 1.0));
      abs_tol = abs_ref * (2.0 ** (-24));
      rel_err = rabs(verilog_sum - c_sum) / denom;

      total_cnt = total_cnt + 1;

      if (abs_err > abs_tol) begin
        fail_cnt = fail_cnt + 1;
        if (fail_cnt <= 50) begin
          $display(
              "[FAIL] a=%h b=%h sum=%h verilog_sum=%e c_sum=%e abs_err=%e abs_tol=%e rel_err=%e",
              a_bits, b_bits, sum, verilog_sum, c_sum, abs_err, abs_tol, rel_err);
        end
      end
    end
  endtask

  function automatic [31:0] rand_normal_fp32;
    integer mant;
    integer exp;
    integer sgn;
    begin
      seed = seed * 1664525 + 1013904223;
      sgn = seed & 1;
      seed = seed * 1664525 + 1013904223;
      exp = (seed & 8'h7F) + 8'd64;
      if (exp > 8'd190) exp = 8'd190;
      if (exp < 8'd1) exp = 8'd1;
      seed = seed * 1664525 + 1013904223;
      mant = seed & 23'h7FFFFF;
      rand_normal_fp32 = {sgn[0], exp[7:0], mant[22:0]};
    end
  endfunction

  initial begin
    total_cnt = 0;
    fail_cnt = 0;
    seed = 32'h13572468;

    run_case(32'h3F800000, 32'h40000000);
    run_case(32'hBF800000, 32'h3F000000);
    run_case(32'h41200000, 32'hC0800000);
    run_case(32'h3EAAAAAB, 32'h3F2AAAAB);
    run_case(32'h00000000, 32'h3F800000);
    run_case(32'h80000000, 32'h3F800000);
    run_case(32'hBF800000, 32'h3FC00000);
    run_case(32'h3f800000, 32'h40000000);

    for (i = 0; i < 10000000; i = i + 1) begin
      run_case(rand_normal_fp32(), rand_normal_fp32());
    end


    if (fail_cnt == 0) begin
      $display("[PASS] tb_add done. total=%0d fail=%0d", total_cnt, fail_cnt);
    end else begin
      $display("[FAIL] tb_add done. total=%0d fail=%0d", total_cnt, fail_cnt);
      $stop;
    end

    $finish;
  end


endmodule
