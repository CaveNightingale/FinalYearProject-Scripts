`timescale 1ns / 1ps

module sram #(
    parameter DataWidth = 64,
    parameter DataCount = 65536
) (
    input  wire        clk,
    input  wire        en,
    input  wire        we,
    input  wire [$clog2(DataCount)-1:0] addr,
    input  wire [DataWidth-1:0] din,
    output reg  [DataWidth-1:0] dout
);

  reg [DataWidth-1:0] mem[0:DataCount-1];

  always @(posedge clk) begin
    if (en) begin
      if (we) begin
        mem[addr] <= din;
      end
      dout <= mem[addr];
    end
  end
endmodule
