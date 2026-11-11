`timescale 1ns/1ps
// Self-checking testbench for nau.sv
//   python3 model/nau_golden.py --out vec              (as published)
//   python3 model/nau_golden.py --improved --out vec   (compile with +define+IMPROVED)
module tb_nau;
`ifdef IMPROVED
  localparam bit OPT = 1'b1;
`else
  localparam bit OPT = 1'b0;
`endif
  localparam int W = 24, NV = 4000;

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic                rst_n = 1'b0;
  logic                in_valid = 1'b0, in_softplus = 1'b0;
  logic signed [W-1:0] in_x = '0;
  logic                out_valid;
  logic signed [W-1:0] out_y;

  nau #(.EXP_NEAR0(OPT), .SP_ORDER2(OPT)) dut (.*);

  logic [W-1:0] vmem [3*NV];          // per line: mode, x, expected y
  initial $readmemh("vec/nau.hex", vmem);

  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    for (int i = 0; i < NV; i++) begin
      @(posedge clk);
      in_valid    <= 1'b1;
      in_softplus <= vmem[3*i][0];
      in_x        <= vmem[3*i+1];
    end
    @(posedge clk);
    in_valid <= 1'b0;
  end

  int idx = 0, errors = 0;
  always @(posedge clk) begin
    if (out_valid) begin
      if (out_y !== $signed(vmem[3*idx+2])) begin
        errors++;
        if (errors <= 10)
          $display("MISMATCH %0d mode=%0d x=%0d got=%0d exp=%0d", idx,
                   vmem[3*idx][0], $signed(vmem[3*idx+1]), out_y, $signed(vmem[3*idx+2]));
      end
      idx++;
    end
  end

  initial begin
    wait (idx == NV);
    repeat (2) @(posedge clk);
    if (errors == 0) $display("PASS: %0d NAU outputs match the golden model", idx);
    else             $display("FAIL: %0d / %0d mismatches", errors, idx);
    $finish;
  end

  initial begin
    #1_000_000;
    $display("TIMEOUT (received %0d outputs)", idx);
    $finish;
  end
endmodule
