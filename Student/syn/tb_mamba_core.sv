`timescale 1ns/1ps
// Self-checking testbench for mamba_core.sv against the hwref layer dumps.
//
// Small random model (default):
//   python3 hwref.py --make-test-model ../testmodel
//   python3 export.py --out ../testmodel --mem --dump 6
//   copy ../testmodel/mem and ../testmodel/dump to vec/mem and vec/dump
// Trained student: export.py --out ../export --mem --dump 6, compile with +define+FULL
//
// Every residual update (dbg_*) is checked against L??_h_out; in_proj rows,
// conv outputs, SSM outputs, gate outputs and both norm exponents are checked
// too, so a mismatch points at the stage that differs.
module tb_mamba_core;
`ifdef FULL
  localparam int D = 256, DI = 512, N = 64, P = 64, NL = 12;
`else
  localparam int D = 128, DI = 256, N = 32, P = 32, NL = 2;
`endif
  localparam int T = 6;                          // tokens in the dump
  localparam int K = 128;
  localparam int NH = DI / P, CONV_DIM = DI + 2 * N, IN_ROWS = 2 * DI + 2 * N + NH;
  localparam int LIN_WORDS = NL * (IN_ROWS * D / K + D * DI / K);
  localparam int NORM_WORDS = NL * (D + DI) + D;

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic          rst_n = 1'b0;
  logic          cfg_we = 1'b0;
  logic [2:0]    cfg_sel = '0;
  logic [15:0]   cfg_addr = '0;
  logic [511:0]  cfg_data = '0;
  logic          tok_go = 1'b0, tok_first = 1'b0;
  logic          busy, tok_done, dbg_valid;
  logic [7:0]    dbg_layer;
  logic [11:0]   dbg_idx;
  logic [31:0]   dbg_h;

  mamba_core #(.D(D), .DI(DI), .N(N), .P(P), .NL(NL)) dut (.*);

  // ---- memory images ----------------------------------------------------------------
  logic [511:0] m_linw [LIN_WORDS];
  logic [15:0]  m_lins [LIN_WORDS];
  logic [17:0]  m_normw [NORM_WORDS];
  logic [71:0]  m_convp [NL * CONV_DIM];
  logic [47:0]  m_ssmp [NL * NH];

  // ---- dumps (token-major) ----------------------------------------------------------------
  logic [31:0] e_hin  [T * D];
  // flat arrays, layer L starts at L * (per-layer size)
  logic [31:0] e_hout [NL * T * D];
  logic [23:0] e_zx   [NL * T * IN_ROWS];
  logic [23:0] e_xbc  [NL * T * CONV_DIM];
  logic [23:0] e_y    [NL * T * DI];
  logic [31:0] e_g    [NL * T * DI];
  logic [7:0]  e_kin  [NL * T];
  logic [7:0]  e_kg   [NL * T];

  initial begin
    $readmemh("vec/mem/linw.hex",  m_linw);
    $readmemh("vec/mem/lins.hex",  m_lins);
    $readmemh("vec/mem/normw.hex", m_normw);
    $readmemh("vec/mem/convp.hex", m_convp);
    $readmemh("vec/mem/ssmp.hex",  m_ssmp);
    $readmemh("vec/dump/L00_h_in.hex", e_hin);
    for (int L = 0; L < NL; L++) begin
      // zero padding written out by hand: some simulators pad "%02d" with spaces
      automatic string pf = $sformatf("vec/dump/L%0d%0d_", L / 10, L % 10);
      $readmemh({pf, "h_out.hex"}, e_hout, L * T * D);
      $readmemh({pf, "zx.hex"},    e_zx,   L * T * IN_ROWS);
      $readmemh({pf, "xbc.hex"},   e_xbc,  L * T * CONV_DIM);
      $readmemh({pf, "y.hex"},     e_y,    L * T * DI);
      $readmemh({pf, "g.hex"},     e_g,    L * T * DI);
      $readmemh({pf, "k_in.hex"},  e_kin,  L * T);
      $readmemh({pf, "k_g.hex"},   e_kg,   L * T);
    end
    // a file shorter than the array (e.g. images of the small model with +define+FULL)
    // leaves the tail at x; stop instead of reporting thousands of mismatches
    if ($isunknown(m_linw[LIN_WORDS - 1]) || $isunknown(m_lins[LIN_WORDS - 1]) ||
        $isunknown(m_normw[NORM_WORDS - 1]) || $isunknown(m_convp[NL * CONV_DIM - 1]) ||
        $isunknown(m_ssmp[NL * NH - 1]))
      $fatal(1, "vec/mem does not match this configuration (D=%0d NL=%0d, %0d weight words expected)",
             D, NL, LIN_WORDS);
    if ($isunknown(e_hin[T * D - 1]) || $isunknown(e_hout[NL * T * D - 1]) || $isunknown(e_kg[NL * T - 1]))
      $fatal(1, "vec/dump does not match this configuration (D=%0d NL=%0d T=%0d)", D, NL, T);
  end

  task automatic write_cfg(input logic [2:0] sel, input int addr, input logic [511:0] data);
    @(posedge clk);
    cfg_we   <= 1'b1;
    cfg_sel  <= sel;
    cfg_addr <= 16'(addr);
    cfg_data <= data;
  endtask

  // ---- stimulus ------------------------------------------------------------------------------
  int tok = 0;
  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    $display("loading memories (%0d weight words) ...", LIN_WORDS);
    for (int i = 0; i < LIN_WORDS; i++)       write_cfg(3'd0, i, m_linw[i]);
    for (int i = 0; i < LIN_WORDS; i++)       write_cfg(3'd1, i, 512'(m_lins[i]));
    for (int i = 0; i < NORM_WORDS; i++)      write_cfg(3'd2, i, 512'(m_normw[i]));
    for (int i = 0; i < NL * CONV_DIM; i++)   write_cfg(3'd3, i, 512'(m_convp[i]));
    for (int i = 0; i < NL * NH; i++)         write_cfg(3'd4, i, 512'(m_ssmp[i]));
    for (tok = 0; tok < T; tok++) begin
      for (int i = 0; i < D; i++) write_cfg(3'd5, i, 512'(e_hin[tok * D + i]));
      @(posedge clk);
      cfg_we    <= 1'b0;
      tok_go    <= 1'b1;
      tok_first <= (tok == 0);
      @(posedge clk);
      tok_go <= 1'b0;
      wait (tok_done);
      $display("token %0d done at %0t", tok, $time);
      @(posedge clk);
    end
    report();
    $finish;
  end

  // ---- checkers ---------------------------------------------------------------------------------
  int n_h = 0, e_h = 0, n_zx = 0, e_zx_c = 0, n_cv = 0, e_cv = 0;
  int n_y = 0, e_yc = 0, n_g = 0, e_gc = 0, e_k = 0, g_idx = 0;

  task automatic report();
    $display("checked: h %0d (%0d bad), in_proj %0d (%0d), conv %0d (%0d), ssm y %0d (%0d), gate %0d (%0d), k errors %0d",
             n_h, e_h, n_zx, e_zx_c, n_cv, e_cv, n_y, e_yc, n_g, e_gc, e_k);
    if (n_h == T * NL * D && e_h + e_zx_c + e_cv + e_yc + e_gc + e_k == 0)
      $display("PASS: %0d tokens x %0d layers match the fixed-point reference", T, NL);
    else
      $display("FAIL");
  endtask

  always @(posedge clk) begin
    automatic int L = dut.layer;
    // residual
    if (dbg_valid) begin
      if (dbg_h !== e_hout[(dbg_layer * T + tok) * D + dbg_idx]) begin
        e_h++;
        if (e_h <= 5) $display("H   MISMATCH tok %0d layer %0d idx %0d got %h exp %h",
                               tok, dbg_layer, dbg_idx, dbg_h, e_hout[(dbg_layer * T + tok) * D + dbg_idx]);
      end
      n_h++;
    end
    // in_proj rows (state S_IN = 2)
    if (dut.st == 2 && dut.ln_valid) begin
      if (dut.y24 !== $signed(e_zx[(L * T + tok) * IN_ROWS + dut.ln_row])) begin
        e_zx_c++;
        if (e_zx_c <= 5) $display("ZX  MISMATCH tok %0d layer %0d row %0d got %0d exp %0d",
                                  tok, L, dut.ln_row, dut.y24, $signed(e_zx[(L * T + tok) * IN_ROWS + dut.ln_row]));
      end
      n_zx++;
    end
    // conv + SiLU
    if (dut.cv_valid) begin
      if (dut.cv_y !== $signed(e_xbc[(L * T + tok) * CONV_DIM + dut.cv_tag])) begin
        e_cv++;
        if (e_cv <= 5) $display("CV  MISMATCH tok %0d layer %0d ch %0d got %0d exp %0d",
                                tok, L, dut.cv_tag, dut.cv_y, $signed(e_xbc[(L * T + tok) * CONV_DIM + dut.cv_tag]));
      end
      n_cv++;
    end
    // SSM output
    if (dut.sc_valid) begin
      automatic int r = (dut.sc_head - L * NH) * P + dut.sc_p;
      if (dut.sc_y !== $signed(e_y[(L * T + tok) * DI + r])) begin
        e_yc++;
        if (e_yc <= 5) $display("Y   MISMATCH tok %0d layer %0d row %0d got %0d exp %0d",
                                tok, L, r, dut.sc_y, $signed(e_y[(L * T + tok) * DI + r]));
      end
      n_y++;
    end
    // gate output (in row order)
    if (dut.gt_valid) begin
      if (dut.gt_g !== $signed(e_g[(L * T + tok) * DI + g_idx])) begin
        e_gc++;
        if (e_gc <= 5) $display("G   MISMATCH tok %0d layer %0d row %0d got %0d exp %0d",
                                tok, L, g_idx, dut.gt_g, $signed(e_g[(L * T + tok) * DI + g_idx]));
      end
      n_g++;
      g_idx = (g_idx == DI - 1) ? 0 : g_idx + 1;
    end
    // norm exponents
    if (dut.nq_done) begin
      if (dut.st == 1 && dut.nq_k !== $signed(e_kin[L * T + tok])) begin
        e_k++;
        $display("K   MISMATCH pre-norm tok %0d layer %0d got %0d exp %0d", tok, L, dut.nq_k, $signed(e_kin[L * T + tok]));
      end
      if (dut.st == 4 && dut.nq_k !== $signed(e_kg[L * T + tok])) begin
        e_k++;
        $display("K   MISMATCH gated norm tok %0d layer %0d got %0d exp %0d", tok, L, dut.nq_k, $signed(e_kg[L * T + tok]));
      end
    end
  end

  initial begin
    #200_000_000;
    $display("TIMEOUT (token %0d)", tok);
    report();
    $finish;
  end
endmodule
