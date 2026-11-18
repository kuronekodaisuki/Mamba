`timescale 1ns/1ps
// Self-checking testbench for ssm_discretize.sv (vectors from model/nau_golden.py)
module tb_ssm_discretize;
`ifdef IMPROVED
  localparam bit OPT = 1'b1;
`else
  localparam bit OPT = 1'b0;
`endif
  localparam int NV = 2000;

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic               rst_n = 1'b0;
  logic               in_valid = 1'b0;
  logic [7:0]         in_tag = '0;
  logic signed [15:0] in_dt_raw = '0, in_dt_bias = '0, in_A = '0;
  logic               out_valid;
  logic [7:0]         out_tag;
  logic signed [23:0] out_delta;
  logic signed [17:0] out_dA;

  ssm_discretize #(.EXP_NEAR0(OPT), .SP_ORDER2(OPT)) dut (.*);

  logic [23:0] vmem [5*NV];           // per line: dt_raw, dt_bias, A, delta, dA
  initial $readmemh("vec/disc.hex", vmem);

  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    for (int i = 0; i < NV; i++) begin
      @(posedge clk);
      in_valid   <= 1'b1;
      in_tag     <= 8'(i);
      in_dt_raw  <= vmem[5*i][15:0];
      in_dt_bias <= vmem[5*i+1][15:0];
      in_A       <= vmem[5*i+2][15:0];
    end
    @(posedge clk);
    in_valid <= 1'b0;
  end

  int idx = 0, errors = 0;
  always @(posedge clk) begin
    if (out_valid) begin
      if (out_delta !== $signed(vmem[5*idx+3]) ||
          out_dA    !== $signed(vmem[5*idx+4][17:0]) ||
          out_tag   !== 8'(idx)) begin
        errors++;
        if (errors <= 10)
          $display("MISMATCH %0d delta got=%0d exp=%0d  dA got=%0d exp=%0d  tag=%0d",
                   idx, out_delta, $signed(vmem[5*idx+3]),
                   out_dA, $signed(vmem[5*idx+4][17:0]), out_tag);
      end
      idx++;
    end
  end

  initial begin
    wait (idx == NV);
    repeat (2) @(posedge clk);
    if (errors == 0) $display("PASS: %0d discretize outputs match the golden model", idx);
    else             $display("FAIL: %0d / %0d mismatches", errors, idx);
    $finish;
  end

  initial begin
    #1_000_000;
    $display("TIMEOUT (received %0d outputs)", idx);
    $finish;
  end
endmodule
