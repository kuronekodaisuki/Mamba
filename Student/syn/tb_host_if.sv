`timescale 1ns/1ps
// host_if + mamba_core driven only through the Avalon-MM BAR (as the PCIe hard IP
// rxm_bar0 would), against the hwref dumps in vec/ (same files as tb_mamba_core).
//
//   linw words : one 4-beat burst (64 bytes) per word
//   other mems : one 64-bit write of the low bytes, then byte 63 alone (commit)
//   per token  : h written through sel 5, CTRL go, STATUS polled, OUT read back
//                with 4-beat burst reads and compared with L??_h_out
//   token 0 captures layer 0 (CAP_LAYER), the others the last layer
//
// Compile with +define+FULL for the trained student (vec/ from ../export).
module tb_host_if;
`ifdef FULL
  localparam int D = 256, DI = 512, N = 64, P = 64, NL = 12;
`else
  localparam int D = 128, DI = 256, N = 32, P = 32, NL = 2;
`endif
  localparam int T = 6;
  localparam int K = 128;
  localparam int NH = DI / P, CONV_DIM = DI + 2 * N, IN_ROWS = 2 * DI + 2 * N + NH;
  localparam int LIN_WORDS = NL * (IN_ROWS * D / K + D * DI / K);
  localparam int NORM_WORDS = NL * (D + DI) + D;
  localparam int AVW = 128, AVB = AVW / 8, AW = 16, BCW = 6;

  logic clk = 1'b0;
  always #2 clk = ~clk;
  logic rst_n = 1'b0;

  // ---- Avalon master signals -------------------------------------------------------------
  logic [AW-1:0]    address = '0;
  logic             read = 1'b0, write = 1'b0;
  logic [AVW-1:0]   writedata = '0;
  logic [AVB-1:0]   byteenable = '0;
  logic [BCW-1:0]   burstcount = BCW'(1);
  logic             waitrequest, readdatavalid;
  logic [AVW-1:0]   readdata;

  logic          cfg_we, tok_go, tok_first, busy, tok_done, dbg_valid;
  logic [2:0]    cfg_sel;
  logic [15:0]   cfg_addr;
  logic [511:0]  cfg_data;
  logic [7:0]    dbg_layer;
  logic [11:0]   dbg_idx;
  logic [31:0]   dbg_h;

  host_if #(.AVW(AVW), .AW(AW), .BCW(BCW), .D(D), .NL(NL)) u_if (
    .clk(clk), .rst_n(rst_n),
    .avs_address(address), .avs_read(read), .avs_write(write), .avs_writedata(writedata),
    .avs_byteenable(byteenable), .avs_burstcount(burstcount), .avs_waitrequest(waitrequest),
    .avs_readdata(readdata), .avs_readdatavalid(readdatavalid),
    .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
    .tok_go(tok_go), .tok_first(tok_first), .busy(busy), .tok_done(tok_done),
    .dbg_valid(dbg_valid), .dbg_layer(dbg_layer), .dbg_idx(dbg_idx), .dbg_h(dbg_h));

  mamba_core #(.D(D), .DI(DI), .N(N), .P(P), .NL(NL)) u_core (
    .clk(clk), .rst_n(rst_n),
    .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
    .tok_go(tok_go), .tok_first(tok_first), .busy(busy), .tok_done(tok_done),
    .dbg_valid(dbg_valid), .dbg_layer(dbg_layer), .dbg_idx(dbg_idx), .dbg_h(dbg_h));

  // ---- images / dumps --------------------------------------------------------------------
  logic [511:0] m_linw [LIN_WORDS];
  logic [15:0]  m_lins [LIN_WORDS];
  logic [17:0]  m_normw [NORM_WORDS];
  logic [71:0]  m_convp [NL * CONV_DIM];
  logic [47:0]  m_ssmp [NL * NH];
  logic [31:0]  e_hin  [T * D];
  logic [31:0]  e_hout [NL * T * D];

  initial begin
    $readmemh("vec/mem/linw.hex",  m_linw);
    $readmemh("vec/mem/lins.hex",  m_lins);
    $readmemh("vec/mem/normw.hex", m_normw);
    $readmemh("vec/mem/convp.hex", m_convp);
    $readmemh("vec/mem/ssmp.hex",  m_ssmp);
    $readmemh("vec/dump/L00_h_in.hex", e_hin);
    for (int L = 0; L < NL; L++)
      $readmemh($sformatf("vec/dump/L%0d%0d_h_out.hex", L / 10, L % 10), e_hout, L * T * D);
    if ($isunknown(m_linw[LIN_WORDS - 1]) || $isunknown(m_ssmp[NL * NH - 1]) ||
        $isunknown(e_hout[NL * T * D - 1]))
      $fatal(1, "vec/ does not match this configuration (D=%0d NL=%0d)", D, NL);
  end

  // ---- bus functional model ---------------------------------------------------------------
  // one write burst of n beats starting at byte address a (beat aligned)
  task automatic bus_write(input int a, input int n, input logic [3:0][AVW-1:0] d,
                           input logic [3:0][AVB-1:0] be);
    for (int i = 0; i < n; i++) begin
      address    <= AW'(a);
      write      <= 1'b1;
      burstcount <= BCW'(n);
      writedata  <= d[i];
      byteenable <= be[i];
      @(posedge clk);
      while (waitrequest) @(posedge clk);
    end
    write <= 1'b0;
  endtask

  // 64-bit write at a (8-byte aligned) into the right lane of a beat
  task automatic wr64(input int a, input logic [63:0] v);
    logic [3:0][AVW-1:0] d = '0;
    logic [3:0][AVB-1:0] be = '0;
    int l = (a % AVB) / 8;
    d[0][64*l +: 64] = v;
    be[0][8*l +: 8]  = 8'hFF;
    bus_write(a - a % AVB, 1, d, be);
  endtask

  // read burst of n beats from a, data returned in q
  logic [AVW-1:0] rq [$];
  int rd_expect = 0;                                   // beats requested since rq was cleared
  task automatic bus_read(input int a, input int n);
    address    <= AW'(a);
    read       <= 1'b1;
    burstcount <= BCW'(n);
    @(posedge clk);
    while (waitrequest) @(posedge clk);
    read <= 1'b0;
    rd_expect += n;
    while (rq.size() < rd_expect) @(posedge clk);
  endtask

  always @(posedge clk) if (readdatavalid) rq.push_back(readdata);

  task automatic rd64(input int a, output logic [63:0] v);
    rq.delete(); rd_expect = 0;
    bus_read(a - a % AVB, 1);
    v = rq[0][64 * ((a % AVB) / 8) +: 64];
  endtask

  // ---- cfg words through the window --------------------------------------------------------
  task automatic set_ptr(input int sel, input int addr);
    wr64('h18, 64'((sel << 16) | addr));
  endtask

  task automatic word_burst(input logic [511:0] w);      // 4 beats of 16 bytes
    logic [3:0][AVW-1:0] d;
    logic [3:0][AVB-1:0] be;
    for (int i = 0; i < 4; i++) begin
      d[i]  = w[128*i +: 128];
      be[i] = '1;
    end
    bus_write('h40, 4, d, be);
  endtask

  task automatic word_narrow(input logic [63:0] lo);   // low 8 bytes, then byte 63
    logic [3:0][AVW-1:0] d = '0;
    logic [3:0][AVB-1:0] be = '0;
    wr64('h40, lo);
    be[0][15] = 1'b1;                                   // byte 0x7F = window byte 63
    bus_write('h70, 1, d, be);
  endtask

  // ---- test ---------------------------------------------------------------------------------
  int errors = 0, checked = 0;
  logic [63:0] v;

  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    repeat (4) @(posedge clk);

    rd64('h0, v);
    if (v !== {32'h4D414D42, 16'd1, 16'(D)}) begin
      errors++; $display("ID mismatch %h", v);
    end

    $display("loading memories through the BAR (%0d weight words) ...", LIN_WORDS);
    set_ptr(0, 0);
    for (int i = 0; i < LIN_WORDS; i++)  word_burst(m_linw[i]);
    set_ptr(1, 0);
    for (int i = 0; i < LIN_WORDS; i++)  word_narrow(64'(m_lins[i]));
    set_ptr(2, 0);
    for (int i = 0; i < NORM_WORDS; i++) word_narrow(64'(m_normw[i]));
    set_ptr(3, 0);
    for (int i = 0; i < NL * CONV_DIM; i++) begin  // 72 bits: two 64-bit writes + byte 63
      wr64('h48, 64'(m_convp[i] >> 64));
      word_narrow(m_convp[i][63:0]);
    end
    set_ptr(4, 0);
    for (int i = 0; i < NL * NH; i++)    word_narrow(64'(m_ssmp[i]));
    rd64('h18, v);
    if (v[18:0] !== {3'd4, 16'(NL * NH)}) begin
      errors++; $display("CFG_PTR after load: %h", v);
    end

    for (int tok = 0; tok < T; tok++) begin
      automatic int cap = (tok == 0) ? 0 : NL - 1;
      wr64('h20, 64'(cap));
      set_ptr(5, 0);
      for (int i = 0; i < D; i++) word_narrow(64'(e_hin[tok * D + i]));
      wr64('h10, (tok == 0) ? 64'd3 : 64'd1);           // go (+ first)
      do rd64('h8, v); while (!v[1]);                   // poll done
      if (v[47:16] !== 32'(tok + 1)) begin
        errors++; $display("STATUS token count %0d, expected %0d", v[47:16], tok + 1);
      end
      rd64('h28, v);
      $display("token %0d: %0d cycles (layer %0d captured)", tok, v[31:0], cap);
      // OUT: D*4 bytes, 4-beat bursts
      rq.delete(); rd_expect = 0;
      for (int a = 0; a < D * 4; a += 4 * AVB) bus_read('h1000 + a, 4);
      for (int i = 0; i < D; i++) begin
        automatic logic [31:0] got = rq[i / (AVB / 4)][32 * (i % (AVB / 4)) +: 32];
        automatic logic [31:0] exp = e_hout[(cap * T + tok) * D + i];
        checked++;
        if (got !== exp) begin
          errors++;
          if (errors <= 10) $display("OUT MISMATCH tok %0d idx %0d got %h exp %h", tok, i, got, exp);
        end
      end
    end

    if (errors == 0 && checked == T * D)
      $display("PASS: %0d tokens through the BAR, %0d outputs match the reference", T, checked);
    else
      $display("FAIL: %0d errors, %0d checked", errors, checked);
    $finish;
  end

  initial begin
    #400_000_000;
    $display("TIMEOUT");
    $finish;
  end
endmodule
