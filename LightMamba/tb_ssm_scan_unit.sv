`timescale 1ns/1ps
// Self-checking testbench: compares out_y with the Python golden model.
// Generate vectors first:  python3 model/ssm_golden.py --out vec
// (N, P, NHEADS, T below must match the generator arguments)
module tb_ssm_scan_unit;
  localparam int N = 8, P = 4, NHEADS = 2, T = 6;
  localparam int DEPTH = NHEADS * P;
  localparam int W_A = 18, W_X = 16, W_B = 16, W_D = 16, W_Y = 24;

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic                      rst_n     = 1'b0;
  logic                      tok_load  = 1'b0;
  logic                      first_tok = 1'b0;
  logic [N-1:0][W_B-1:0]     B_in      = '0;
  logic [N-1:0][W_B-1:0]     C_in      = '0;
  logic                      in_valid  = 1'b0;
  logic [$clog2(NHEADS)-1:0] in_head   = '0;
  logic [$clog2(P)-1:0]      in_p      = '0;
  logic signed [W_A-1:0]     in_dA     = '0;
  logic signed [W_X-1:0]     in_xdt    = '0;
  logic signed [W_X-1:0]     in_x      = '0;
  logic signed [W_D-1:0]     in_D      = '0;
  logic                      out_valid;
  logic [$clog2(NHEADS)-1:0] out_head;
  logic [$clog2(P)-1:0]      out_p;
  logic signed [W_Y-1:0]     out_y;

  ssm_scan_unit #(.N(N), .P(P), .NHEADS(NHEADS)) dut (.*);

  logic [W_B-1:0] B_mem   [T*N];
  logic [W_B-1:0] C_mem   [T*N];
  logic [W_A-1:0] dA_mem  [T*NHEADS];
  logic [W_D-1:0] D_mem   [NHEADS];
  logic [W_X-1:0] xdt_mem [T*DEPTH];
  logic [W_X-1:0] x_mem   [T*DEPTH];
  logic [W_Y-1:0] y_mem   [T*DEPTH];

  initial begin
    $readmemh("vec/B.hex",   B_mem);
    $readmemh("vec/C.hex",   C_mem);
    $readmemh("vec/dA.hex",  dA_mem);
    $readmemh("vec/D.hex",   D_mem);
    $readmemh("vec/xdt.hex", xdt_mem);
    $readmemh("vec/x.hex",   x_mem);
    $readmemh("vec/y.hex",   y_mem);
  end

  // ---- stimulus ----
  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    for (int t = 0; t < T; t++) begin
      @(posedge clk);
      in_valid  <= 1'b0;
      tok_load  <= 1'b1;
      first_tok <= (t == 0);
      for (int n = 0; n < N; n++) begin
        B_in[n] <= B_mem[t*N + n];
        C_in[n] <= C_mem[t*N + n];
      end
      for (int h = 0; h < NHEADS; h++) begin
        for (int p = 0; p < P; p++) begin
          @(posedge clk);
          tok_load <= 1'b0;
          in_valid <= 1'b1;
          in_head  <= h;
          in_p     <= p;
          in_dA    <= dA_mem[t*NHEADS + h];
          in_D     <= D_mem[h];
          in_xdt   <= xdt_mem[t*DEPTH + h*P + p];
          in_x     <= x_mem[t*DEPTH + h*P + p];
        end
      end
    end
    @(posedge clk);
    in_valid <= 1'b0;
  end

  // ---- checker ----
  int idx = 0, errors = 0;
  always @(posedge clk) begin
    if (out_valid) begin
      if (out_y !== $signed(y_mem[idx])) begin
        errors++;
        if (errors <= 10)
          $display("MISMATCH idx=%0d head=%0d p=%0d got=%0d exp=%0d",
                   idx, out_head, out_p, out_y, $signed(y_mem[idx]));
      end
      idx++;
    end
  end

  initial begin
    wait (idx == T*DEPTH);
    repeat (2) @(posedge clk);
    if (errors == 0) $display("PASS: %0d outputs match the golden model", idx);
    else             $display("FAIL: %0d / %0d mismatches", errors, idx);
    $finish;
  end

  initial begin
    #1_000_000;
    $display("TIMEOUT (received %0d outputs)", idx);
    $finish;
  end
endmodule
