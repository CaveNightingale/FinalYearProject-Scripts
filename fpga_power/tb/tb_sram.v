`timescale 1ns / 1ps

module tb_sram;

    localparam integer DataWidth = 64;
    localparam integer DataCount = 65536;
    localparam integer AddrWidth = $clog2(DataCount);

    reg                  clk;
    reg                  en;
    reg                  we;
    reg  [AddrWidth-1:0] addr;
    reg  [DataWidth-1:0] din;
    wire [DataWidth-1:0] dout;

    integer i;
    integer error_count;
    reg [AddrWidth-1:0]  rand_addr;
    reg [DataWidth-1:0]  expected;

    sram #(
        .DataWidth(DataWidth),
        .DataCount(DataCount)
    ) dut (
        .clk(clk),
        .en(en),
        .we(we),
        .addr(addr),
        .din(din),
        .dout(dout)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task automatic write_word;
        input [AddrWidth-1:0] w_addr;
        input [DataWidth-1:0] w_data;
        begin
            @(negedge clk);
            en   = 1'b1;
            we   = 1'b1;
            addr = w_addr;
            din  = w_data;
            @(posedge clk);
        end
    endtask

    task automatic read_and_check;
        input [AddrWidth-1:0] r_addr;
        input [DataWidth-1:0] exp_data;
        begin
            @(negedge clk);
            en   = 1'b1;
            we   = 1'b0;
            addr = r_addr;
            din  = {DataWidth{1'b0}};
            @(posedge clk);
            #1;
            if (dout !== exp_data) begin
                error_count = error_count + 1;
                $display("[ERROR] t=%0t addr=%0d exp=%h got=%h", $time, r_addr, exp_data, dout);
            end
        end
    endtask

    initial begin
        en = 1'b0;
        we = 1'b0;
        addr = {AddrWidth{1'b0}};
        din  = {DataWidth{1'b0}};
        error_count = 0;

        repeat (2) @(posedge clk);

        write_word(16'd0,     64'h0123_4567_89ab_cdef);
        write_word(16'd1,     64'h1111_2222_3333_4444);
        write_word(16'd1024,  64'hdead_beef_cafe_f00d);
        write_word(16'd65535, 64'hffff_eeee_dddd_cccc);

        read_and_check(16'd0,     64'h0123_4567_89ab_cdef);
        read_and_check(16'd1,     64'h1111_2222_3333_4444);
        read_and_check(16'd1024,  64'hdead_beef_cafe_f00d);
        read_and_check(16'd65535, 64'hffff_eeee_dddd_cccc);

        for (i = 0; i < 16; i = i + 1) begin
            rand_addr = $urandom_range(DataCount-1, 0);
            expected  = {32'h5a5a_0000 + i[31:0], 16'h1357, rand_addr};
            write_word(rand_addr, expected);
            read_and_check(rand_addr, expected);
        end

        @(negedge clk);
        en = 1'b0;
        we = 1'b0;

        if (error_count == 0) begin
            $display("[PASS] SRAM TB passed.");
        end else begin
            $display("[FAIL] SRAM TB failed. error_count=%0d", error_count);
        end

        #20;
        $finish;
    end

endmodule
