`timescale 1ns/1ps
// Self-checking testbench for lin_unit.sv
//   python3 model/lin_golden.py --unit --out vec                       (random weights)
//   python3 model/lin_golden.py --unit --npz ../export/weights.npz \
//           --layer 0 --proj in --out vec                              (trained weights)
// The weights are placed at a non-zero base address and the job is run twice
// back to back; every float32 row result is checked bit for bit.
module tb_lin_unit;
  localparam int K = 128, G_MAX = 4, AW = 16, RW = 12;
  localparam int MAXW = 8192, BASE = 37;
  localparam int RL = 3;            // memory read latency (sdp_ram with IN_REG + OUT_REG)

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic                  rst_n = 1'b0;
  logic                  act_we = 1'b0;
  logic [1:0]            act_g = '0;
  logic [K-1:0][7:0]     act_d = '0;
  logic                  start = 1'b0;
  logic [AW-1:0]         base = '0;
  logic [RW-1:0]         n_rows = '0;
  logic [2:0]            n_groups = '0;
  logic signed [7:0]     k = '0;
  logic                  busy;
  logic [AW-1:0]         w_addr;
  logic [K-1:0][3:0]     w_data;
  logic [15:0]           s_data;
  logic                  y_valid;
  logic [RW-1:0]         y_row;
  logic [31:0]           y_data;
  logic                  done;

  lin_unit #(.K(K), .G_MAX(G_MAX), .AW(AW), .RW(RW), .RL(RL)) dut (.*);

  // ---- weight / scale memory model (read latency RL, like sdp_ram) ----------------
  logic [K*4-1:0] wmem [MAXW];
  logic [15:0]    smem [MAXW];
  logic [RL-1:0][K*4-1:0] wpipe;
  logic [RL-1:0][15:0]    spipe;
  always_ff @(posedge clk) begin
    wpipe[0] <= wmem[w_addr];
    spipe[0] <= smem[w_addr];
    for (int i = 1; i < RL; i++) begin
      wpipe[i] <= wpipe[i-1];
      spipe[i] <= spipe[i-1];
    end
  end
  assign w_data = wpipe[RL-1];
  assign s_data = spipe[RL-1];

  // ---- vectors ------------------------------------------------------------------------
  logic [K*4-1:0] wfile [MAXW];
  logic [15:0]    sfile [MAXW];
  logic [K*8-1:0] afile [G_MAX];
  logic [15:0]    cfg   [3];
  logic [31:0]    yexp  [4096];
  int rows, groups;

  initial begin
    $readmemh("vec/lu_w.hex",   wfile);
    $readmemh("vec/lu_s.hex",   sfile);
    $readmemh("vec/lu_a.hex",   afile);
    $readmemh("vec/lu_cfg.hex", cfg);
    $readmemh("vec/lu_y.hex",   yexp);
    rows   = cfg[0];
    groups = cfg[1];
    for (int i = 0; i < rows * groups; i++) begin
      wmem[BASE + i] = wfile[i];
      smem[BASE + i] = sfile[i];
    end
    $display("config: %0d rows x %0d groups, k = %0d", rows, groups, $signed(cfg[2][7:0]));
  end

  // ---- stimulus ------------------------------------------------------------------------
  int errors = 0, got = 0;

  task automatic run_job(input int pass);
    @(posedge clk);
    start    <= 1'b1;
    base     <= AW'(BASE);
    n_rows   <= RW'(rows);
    n_groups <= 3'(groups);
    k        <= cfg[2][7:0];
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);
    wait (!busy);
    $display("pass %0d finished at %0t", pass, $time);
  endtask

  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    for (int g = 0; g < groups; g++) begin
      @(posedge clk);
      act_we <= 1'b1;
      act_g  <= 2'(g);
      act_d  <= afile[g];
    end
    @(posedge clk);
    act_we <= 1'b0;
    run_job(1);
    run_job(2);
    repeat (4) @(posedge clk);
    if (errors == 0 && got == 2 * rows)
      $display("PASS: %0d rows (2 passes) match the golden model", got);
    else
      $display("FAIL: %0d mismatches, %0d / %0d rows received", errors, got, 2 * rows);
    $finish;
  end

  // ---- checker -----------------------------------------------------------------------
  int expect_row = 0;
  always @(posedge clk) begin
    if (y_valid) begin
      if (y_row !== RW'(expect_row) || y_data !== yexp[expect_row]) begin
        errors++;
        if (errors <= 10)
          $display("MISMATCH row %0d (got row %0d) got=%h exp=%h",
                   expect_row, y_row, y_data, yexp[expect_row]);
      end
      got++;
      expect_row = (expect_row == rows - 1) ? 0 : expect_row + 1;
    end
  end

  initial begin
    #2_000_000;
    $display("TIMEOUT (received %0d rows)", got);
    $finish;
  end
endmodule
