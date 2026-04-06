module blink(
    input wire clk,
    output reg led
);

reg [3:0] counter = 0;
reg [11:0] p = 3322;

always @(posedge clk) begin
    counter <= counter != 11 ? counter + 1 : 0;
    led <= p[counter];
end

endmodule
