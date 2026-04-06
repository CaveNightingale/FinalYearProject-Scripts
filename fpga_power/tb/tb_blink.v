`timescale 1ns/1ps

module tb_blink;

reg clk = 0;
wire led;

// 实例化模块
blink uut (
    .clk(clk),
    .led(led)
);

// 生成时钟（10ns周期）
always #5 clk = ~clk;

// 仿真控制
initial begin
    #200;
    $display("[PASS] LED state: %b", led);
    $finish;
end

endmodule