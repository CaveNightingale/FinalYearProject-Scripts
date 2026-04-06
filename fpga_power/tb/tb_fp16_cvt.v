`timescale 1ns / 1ps

module tb_fp16_cvt;

    integer i;
    integer test_count;
    integer error_count;
    integer skip_count;
    integer sat_count;

    real ideal_val;
    real dut_val;
    real abs_err;
    real ulp;
    real max_fp16_abs;

    reg  [31:0] in_fp32_ftz;
    wire [15:0] out_fp16;

    fp16_cvt dut (
            .in_fp32_ftz(in_fp32_ftz),
            .out_fp16(out_fp16)
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

    function automatic integer is_valid_fp32_ftz;
        input [31:0] x;
        reg [7:0] exp_field;
        reg [22:0] mant_field;
        begin
            exp_field = x[30:23];
            mant_field = x[22:0];
            is_valid_fp32_ftz = (exp_field != 8'hff) && !((exp_field == 0) && (mant_field != 0));
        end
    endfunction

    function automatic [15:0] sat_fp16_bits;
        input sign;
        begin
            sat_fp16_bits = sign ? 16'hffff : 16'h7fff;
        end
    endfunction

    function automatic real decode_fp32_ftz;
        input [31:0] x;
        integer exp_unbiased;
        real frac;
        begin
            if (x[30:23] == 0) begin
                decode_fp32_ftz = 0.0;
            end else begin
                exp_unbiased = x[30:23] - 127;
                frac = 1.0 + (x[22:0] * 1.0) / 8388608.0;
                decode_fp32_ftz = (x[31] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
            end
        end
    endfunction

    function automatic real decode_fp16_fnuz;
        input [15:0] x;
        integer exp_unbiased;
        real frac;
        begin
            if (x[14:10] == 0) begin
                frac = (x[9:0] * 1.0) / 1024.0;
                decode_fp16_fnuz = (x[15] ? -1.0 : 1.0) * pow2i(1 - 15) * frac;
            end else begin
                exp_unbiased = x[14:10] - 15;
                frac = 1.0 + (x[9:0] * 1.0) / 1024.0;
                decode_fp16_fnuz = (x[15] ? -1.0 : 1.0) * pow2i(exp_unbiased) * frac;
            end
        end
    endfunction

    function automatic real ulp_of_fp16;
        input [15:0] x;
        integer exp_unbiased;
        begin
            if (x[14:10] == 0) begin
                ulp_of_fp16 = pow2i(1 - 15 - 10);
            end else begin
                exp_unbiased = x[14:10] - 15;
                ulp_of_fp16 = pow2i(exp_unbiased - 10);
            end
        end
    endfunction

    task automatic check_case;
        input [31:0] x;
        begin
            in_fp32_ftz = x;
            #1;

            if (!is_valid_fp32_ftz(x)) begin
                skip_count = skip_count + 1;
            end else begin
                test_count = test_count + 1;
                ideal_val = decode_fp32_ftz(x);
                dut_val = decode_fp16_fnuz(out_fp16);
                ulp = ulp_of_fp16(out_fp16);
                abs_err = fabs(dut_val - ideal_val);

                if (fabs(ideal_val) > max_fp16_abs) begin
                    sat_count = sat_count + 1;
                    if (out_fp16 !== sat_fp16_bits(x[31])) begin
                        error_count = error_count + 1;
                        if (error_count <= 40) begin
                            $display(
                                "[FAIL][fp16_cvt_sat] in=0x%08h out=0x%04h expected_sat=0x%04h ideal=%e",
                                x, out_fp16, sat_fp16_bits(x[31]), ideal_val
                            );
                        end
                    end
                end else if (abs_err > (ulp + 1.0e-20)) begin
                    error_count = error_count + 1;
                    if (error_count <= 40) begin
                        $display(
                            "[FAIL][fp16_cvt] in=0x%08h out=0x%04h ideal=%e out_val=%e abs_err=%e ulp=%e",
                            x, out_fp16, ideal_val, dut_val, abs_err, ulp
                        );
                    end
                end
            end
        end
    endtask

    initial begin
        $dumpfile("tb_fp16_cvt.vcd");
        $dumpvars(0, tb_fp16_cvt);

        test_count = 0;
        error_count = 0;
        skip_count = 0;
        sat_count = 0;
        in_fp32_ftz = 32'd0;
        #1;
        max_fp16_abs = decode_fp16_fnuz(16'h7fff);

        check_case(32'h00000000);
        check_case(32'h80000000);
        check_case(32'h00800000);
        check_case(32'h33800000);
        check_case(32'h38800000);
        check_case(32'h3f800000);
        check_case(32'h40000000);
        check_case(32'h40400000);
        check_case(32'h477fe000);
        check_case(32'h477ff000);
        check_case(32'h477fffff);
        check_case(32'hbf800000);
        check_case(32'hc0000000);

        for (i = 0; i < 20000; i = i + 1) begin
            check_case({$random, $random});
        end

        if (error_count == 0) begin
            $display("[PASS] tb_fp16_cvt passed. tests=%0d skip=%0d sat=%0d", test_count, skip_count,
                     sat_count);
        end else begin
            $display("[FAIL] tb_fp16_cvt failed. tests=%0d errors=%0d skip=%0d sat=%0d", test_count,
                     error_count, skip_count, sat_count);
        end

        #10;
        $finish;
    end

endmodule
