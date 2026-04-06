`timescale 1ns / 1ps

module tb_pe;

  reg clk;
  reg rst;
  reg en;
  reg clear_in;
  reg [16:0] a_in;
  reg [16:0] b_in;

  wire clear_out;
  wire [16:0] a_out;
  wire [16:0] b_out;
  wire [31:0] sum_out;

  localparam [16:0] FP17_1_0 = 17'h0BC00;
  localparam [16:0] FP17_2_0 = 17'h0C000;
  localparam [16:0] FP17_3_0 = 17'h0C200;
  localparam [16:0] FP17_4_0 = 17'h0C400;
  localparam [16:0] FP17_5_0 = 17'h0C500;
  localparam [16:0] FP17_6_0 = 17'h0C600;
  localparam [16:0] FP17_7_0 = 17'h0C700;
  localparam [16:0] FP17_8_0 = 17'h0C800;
  localparam [16:0] FP17_0_0 = 17'h00000;

  localparam [31:0] FP32_14_0 = 32'h41600000;
  localparam [31:0] FP32_86_0 = 32'h42AC0000;

  integer cyc;
  integer fail_cnt;

  pe dut (
      .clk(clk),
      .rst(rst),
      .en(en),
      .clear_in(clear_in),
      .a_in(a_in),
      .b_in(b_in),
      .clear_out(clear_out),
      .a_out(a_out),
      .b_out(b_out),
      .sum_out(sum_out)
  );

  always #5 clk = ~clk;

  always @(posedge clk or negedge rst) begin
    if (!rst) begin
      cyc <= 0;
    end else begin
      cyc <= cyc + 1;
      $display("[TB] cyc=%0d clear_in=%0b a_in=%h b_in=%h sum_out=%h", cyc, clear_in, a_in, b_in, sum_out);
    end
  end

  task automatic run_round;
    input integer rid;
    input [16:0] a_val;
    input [16:0] b_val;
    input clear_val;
    input check_en;
    input [31:0] expected;
    begin
      @(negedge clk);
      a_in = a_val;
      b_in = b_val;
      clear_in = clear_val;

      @(posedge clk);
      #1;
      if (check_en) begin
        if (sum_out !== expected) begin
          fail_cnt = fail_cnt + 1;
          $display("[FAIL] round=%0d expected=%h got=%h", rid, expected, sum_out);
        end else begin
          $display("[PASS] round=%0d expected=%h got=%h", rid, expected, sum_out);
        end
      end
    end
  endtask

  initial begin
    $dumpfile("tb_pe.vcd");
    $dumpvars(0, tb_pe);

    clk = 0;
    rst = 0;
    en = 0;
    clear_in = 0;
    a_in = 17'd0;
    b_in = 17'd0;
    fail_cnt = 0;

    repeat (2) @(posedge clk);
    rst = 1;
    en = 1;

    run_round(1, FP17_1_0, FP17_2_0, 1'b1, 1'b0, 32'd0);
    run_round(2, FP17_3_0, FP17_4_0, 1'b0, 1'b0, 32'd0);
    run_round(3, FP17_5_0, FP17_6_0, 1'b1, 1'b0, 32'd0);
    run_round(4, FP17_7_0, FP17_8_0, 1'b0, 1'b0, 32'd0);
    run_round(5, FP17_0_0, FP17_0_0, 1'b0, 1'b0, 32'd0);
    run_round(6, FP17_0_0, FP17_0_0, 1'b0, 1'b1, FP32_14_0);
    run_round(7, FP17_0_0, FP17_0_0, 1'b0, 1'b0, 32'd0);
    run_round(8, FP17_0_0, FP17_0_0, 1'b0, 1'b1, FP32_86_0);

    if (fail_cnt == 0) begin
      $display("[PASS] tb_pe timing test done.");
    end else begin
      $display("[FAIL] tb_pe timing test done. fail_cnt=%0d", fail_cnt);
      $stop;
    end

    $finish;
  end

endmodule
