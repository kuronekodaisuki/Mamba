`timescale 1ns/1ps
// Self-checking testbench for lin_dot.sv
//   python3 model/lin_golden.py --out vec
// Inputs are streamed with random bubbles; outputs are checked in order.
module tb_lin_dot;
  localparam int K = 128, NV = 1000, W_TAG = 16;
  localparam int W_O = 12 + $clog2(K);

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic                  rst_n = 1'b0;
  logic                  in_valid = 1'b0;
  logic [W_TAG-1:0]      in_tag = '0;
  logic [K-1:0][3:0]     in_w = '0;
  logic [K-1:0][7:0]     in_a = '0;
  logic                  out_valid;
  logic [W_TAG-1:0]      out_tag;
  logic signed [W_O-1:0] out_sum;

  lin_dot #(.K(K), .W_TAG(W_TAG)) dut (.*);

  logic [K*4-1:0] w_mem [NV];
  logic [K*8-1:0] a_mem [NV];
  logic [W_O-1:0] y_mem [NV];

  initial begin
    $readmemh("vec/lin_w.hex", w_mem);
    $readmemh("vec/lin_a.hex", a_mem);
    $readmemh("vec/lin_y.hex", y_mem);
  end

  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    for (int i = 0; i < NV; ) begin
      @(posedge clk);
      if ($urandom_range(0, 3) == 0) begin      // ~25% bubbles
        in_valid <= 1'b0;
      end else begin
        in_valid <= 1'b1;
        in_tag   <= W_TAG'(i);
        in_w     <= w_mem[i];
        in_a     <= a_mem[i];
        i++;
      end
    end
    @(posedge clk);
    in_valid <= 1'b0;
  end

  int idx = 0, errors = 0;
  always @(posedge clk) begin
    if (out_valid) begin
      if (out_sum !== $signed(y_mem[idx]) || out_tag !== W_TAG'(idx)) begin
        errors++;
        if (errors <= 10)
          $display("MISMATCH %0d tag=%0d got=%0d exp=%0d", idx, out_tag, out_sum,
                   $signed(y_mem[idx]));
      end
      idx++;
    end
  end

  initial begin
    wait (idx == NV);
    repeat (2) @(posedge clk);
    if (errors == 0) $display("PASS: %0d dot products match the golden model", idx);
    else             $display("FAIL: %0d / %0d mismatches", errors, idx);
    $finish;
  end

  initial begin
    #1_000_000;
    $display("TIMEOUT (received %0d outputs)", idx);
    $finish;
  end
endmodule
