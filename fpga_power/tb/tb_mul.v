`timescale 1ns / 1ps
module tb_mul;

  reg [31:0] a;
  reg [31:0] b;
  wire [31:0] product;

  integer total_cnt;
  integer fail_cnt;
  integer i;
  integer seed;

  real a_real;
  real b_real;
  real verilog_product;
  real c_product;
  real rel_err;
  real abs_product;
  real denom;

  mul dut (
      .a      (a),
      .b      (b),
      .product(product)
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

  task automatic run_case_exact;
    input [31:0] a_bits;
    input [31:0] b_bits;
    input [31:0] expected_bits;
    begin
      a = a_bits;
      b = b_bits;
      #1;

      total_cnt = total_cnt + 1;

      if (product !== expected_bits) begin
        fail_cnt = fail_cnt + 1;
        $display("[FAIL] a=%h b=%h product=%h expected=%h", a_bits, b_bits, product, expected_bits);
      end
    end
  endtask

  task automatic run_case;
    input [31:0] a_bits;
    input [31:0] b_bits;
    real min_normal;
    begin
      a = a_bits;
      b = b_bits;
      #1;

      a_real = fp32_to_real(a_bits);
      b_real = fp32_to_real(b_bits);
      verilog_product = fp32_to_real(product);
      c_product = a_real * b_real;
      abs_product = rabs(c_product);
      min_normal = 2.0 ** (-126);

      total_cnt = total_cnt + 1;

      if (a_bits[30:23] == 8'd0 || b_bits[30:23] == 8'd0 || abs_product < min_normal) begin
        if (product !== 32'h00000000) begin
          fail_cnt = fail_cnt + 1;
          $display("[FAIL] a=%h b=%h product=%h expected=00000000", a_bits, b_bits, product);
        end
      end else begin
        denom   = rmax(abs_product, 1.0);
        rel_err = rabs(verilog_product - c_product) / denom;

        if (rel_err > 1.0e-7) begin
          fail_cnt = fail_cnt + 1;
          $display("[FAIL] a=%h b=%h product=%h vp=%e cp=%e rel_err=%e", a_bits, b_bits, product,
                   verilog_product, c_product, rel_err);
        end
      end
    end
  endtask

  function automatic [31:0] rand_normal_fp32_mul;
    integer mant;
    integer exp;
    integer sgn;
    begin
      seed = seed * 1664525 + 1013904223;
      sgn = seed & 1;
      seed = seed * 1664525 + 1013904223;
      exp = (seed % 61) + 97;
      seed = seed * 1664525 + 1013904223;
      mant = seed & 23'h7FFFFF;
      rand_normal_fp32_mul = {sgn[0], exp[7:0], mant[22:0]};
    end
  endfunction

  initial begin
    total_cnt = 0;
    fail_cnt = 0;
    seed = 32'h24681357;

    run_case_exact(32'h3F800000, 32'h40000000, 32'h40000000);
    run_case_exact(32'hBF800000, 32'h3F000000, 32'hBF000000);
    run_case_exact(32'h41200000, 32'hC0800000, 32'hC2200000);
    run_case_exact(32'h00000000, 32'h3F800000, 32'h00000000);
    run_case_exact(32'h80000000, 32'h3F800000, 32'h00000000);
    run_case_exact(32'h00800000, 32'h3F000000, 32'h00000000);

    run_case(32'h3EAAAAAB, 32'h3F2AAAAB);
    run_case(32'h40400000, 32'h3F400000);
    run_case(32'hBE800000, 32'hBF800000);

    for (i = 0; i < 1000000; i = i + 1) begin
      run_case(rand_normal_fp32_mul(), rand_normal_fp32_mul());
    end

    if (fail_cnt == 0) begin
      $display("[PASS] tb_mul done. total=%0d fail=%0d", total_cnt, fail_cnt);
    end else begin
      $display("[FAIL] tb_mul done. total=%0d fail=%0d", total_cnt, fail_cnt);
      $stop;
    end

    $finish;
  end

endmodule
