`timescale 1ns/1ps
// Self-checking testbench for norm_q8.sv
//   python3 model/vec_golden.py --norm --out vec            (n = 256)
//   python3 model/vec_golden.py --norm --n 512 --out vec    (n = 512)
// Each vector is streamed with random bubbles; every int8 output word and the
// exponent k are checked bit for bit. The norm weights sit at a non-zero base
// address of a memory model with read latency RL.
module tb_norm_q8;
  localparam int NMAX = 512, K = 128, WAW = 14, RL = 2;
  localparam int MAXT = 64;

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic                      rst_n = 1'b0;
  logic                      start = 1'b0;
  logic [3:0]                n_log2 = '0;
  logic [WAW-1:0]            w_base = '0;
  logic                      busy;
  logic                      in_valid = 1'b0;
  logic signed [31:0]        in_v = '0;
  logic [WAW-1:0]            w_addr;
  logic signed [17:0]        w_data;
  logic                      act_we;
  logic [1:0]                act_g;
  logic [K-1:0][7:0]         act_d;
  logic signed [7:0]         k_out;
  logic                      done;

  norm_q8 #(.NMAX(NMAX), .K(K), .WAW(WAW), .RL(RL)) dut (.*);

  // ---- vectors -------------------------------------------------------------------
  logic [15:0]    cfg  [3];
  logic [17:0]    wvec [NMAX];
  logic [31:0]    vvec [MAXT * NMAX];
  logic [K*8-1:0] qvec [MAXT * NMAX / K];
  logic [7:0]     kvec [MAXT];
  int T, n, base;

  initial begin
    $readmemh("vec/nq_cfg.hex", cfg);
    $readmemh("vec/nq_w.hex",   wvec);
    $readmemh("vec/nq_v.hex",   vvec);
    $readmemh("vec/nq_q.hex",   qvec);
    $readmemh("vec/nq_k.hex",   kvec);
    T    = cfg[0];
    n    = 1 << cfg[1];
    base = cfg[2];
    $display("config: %0d vectors of n = %0d, weights at %0d", T, n, base);
  end

  // ---- norm-weight memory model (latency RL) ---------------------------------------
  logic [17:0] wmem [1 << WAW];
  initial begin
    #1;
    for (int i = 0; i < n; i++) wmem[base + i] = wvec[i];
  end
  logic [RL-1:0][17:0] wpipe;
  always_ff @(posedge clk) begin
    wpipe[0] <= wmem[w_addr];
    for (int i = 1; i < RL; i++) wpipe[i] <= wpipe[i-1];
  end
  assign w_data = wpipe[RL-1];

  // ---- stimulus ----------------------------------------------------------------------
  int errors = 0, words = 0, cur = 0;

  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    for (cur = 0; cur < T; cur++) begin
      @(posedge clk);
      start  <= 1'b1;
      n_log2 <= cfg[1][3:0];
      w_base <= WAW'(base);
      @(posedge clk);
      start <= 1'b0;
      for (int i = 0; i < n; ) begin
        @(posedge clk);
        if ($urandom_range(0, 3) == 0) begin
          in_valid <= 1'b0;
        end else begin
          in_valid <= 1'b1;
          in_v     <= vvec[cur * n + i];
          i++;
        end
      end
      @(posedge clk);
      in_valid <= 1'b0;
      wait (done);
      @(posedge clk);
    end
    repeat (4) @(posedge clk);
    if (errors == 0 && words == T * n / K)
      $display("PASS: %0d vectors (%0d words) match the golden model", T, words);
    else
      $display("FAIL: %0d mismatches, %0d / %0d words", errors, words, T * n / K);
    $finish;
  end

  // ---- checker -------------------------------------------------------------------------
  always @(posedge clk) begin
    if (act_we) begin
      if (act_d !== qvec[words] || act_g !== 2'(words % (n / K)) || k_out !== kvec[cur]) begin
        errors++;
        if (errors <= 10)
          $display("MISMATCH vector %0d group %0d: k got %0d exp %0d, data %s",
                   cur, act_g, k_out, $signed(kvec[cur]),
                   (act_d === qvec[words]) ? "ok" : "differs");
      end
      words++;
    end
  end

  initial begin
    #5_000_000;
    $display("TIMEOUT (%0d words)", words);
    $finish;
  end
endmodule
