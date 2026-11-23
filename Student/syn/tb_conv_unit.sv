`timescale 1ns/1ps
// Self-checking testbench for conv_unit.sv
//   python3 model/vec_golden.py --conv --out vec
// Stream order: token, layer, channel (with random bubbles); the first token
// has in_first = 1. Parameter memory model with read latency RL.
module tb_conv_unit;
  localparam int NCH = 16, NLAYER = 2, RL = 2;
  localparam int MAXN = 4096;

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic               rst_n = 1'b0;
  logic               in_valid = 1'b0, in_first = 1'b0;
  logic [0:0]         in_layer = '0;
  logic [3:0]         in_ch = '0;
  logic signed [23:0] in_x = '0;
  logic [15:0]        in_tag = '0;
  logic [4:0]         p_addr;
  logic [71:0]        p_data;
  logic               out_valid;
  logic signed [23:0] out_y;
  logic [15:0]        out_tag;

  conv_unit #(.NCH(NCH), .NLAYER(NLAYER), .RL(RL)) dut (.*);

  logic [15:0] cfg [3];
  logic [71:0] pmem [NCH * NLAYER];
  logic [23:0] xv [MAXN];
  logic [23:0] yv [MAXN];
  int T, N;

  initial begin
    $readmemh("vec/cv_cfg.hex", cfg);
    $readmemh("vec/cv_p.hex",   pmem);
    $readmemh("vec/cv_x.hex",   xv);
    $readmemh("vec/cv_y.hex",   yv);
    T = cfg[0];
    N = T * NLAYER * NCH;
    $display("config: %0d tokens x %0d layers x %0d channels", T, NLAYER, NCH);
  end

  logic [RL-1:0][71:0] ppipe;
  always_ff @(posedge clk) begin
    ppipe[0] <= pmem[p_addr];
    for (int i = 1; i < RL; i++) ppipe[i] <= ppipe[i-1];
  end
  assign p_data = ppipe[RL-1];

  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    for (int i = 0; i < N; ) begin
      @(posedge clk);
      if ($urandom_range(0, 3) == 0) in_valid <= 1'b0;
      else begin
        in_valid <= 1'b1;
        in_first <= (i < NLAYER * NCH);
        in_layer <= 1'((i / NCH) % NLAYER);
        in_ch    <= 4'(i % NCH);
        in_x     <= xv[i];
        in_tag   <= 16'(i);
        i++;
      end
    end
    @(posedge clk);
    in_valid <= 1'b0;
  end

  int oi = 0, errors = 0;
  always @(posedge clk) begin
    if (out_valid) begin
      if (out_y !== $signed(yv[oi]) || out_tag !== 16'(oi)) begin
        errors++;
        if (errors <= 10)
          $display("MISMATCH %0d (token %0d layer %0d ch %0d) got=%0d exp=%0d", oi,
                   oi / (NLAYER * NCH), (oi / NCH) % NLAYER, oi % NCH, out_y, $signed(yv[oi]));
      end
      oi++;
    end
  end

  initial begin
    wait (N > 0 && oi == N);
    repeat (2) @(posedge clk);
    if (errors == 0) $display("PASS: %0d conv + SiLU outputs match the golden model", oi);
    else             $display("FAIL: %0d / %0d mismatches", errors, oi);
    $finish;
  end

  initial begin
    #1_000_000;
    $display("TIMEOUT (%0d outputs)", oi);
    $finish;
  end
endmodule
