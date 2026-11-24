// -----------------------------------------------------------------------------
// Large simple dual-port RAM split into banks (depth) x slices (width), each
// piece with its own copy of the address / write registers.
//
// A 40k x 512 memory spans ~1000 M20K blocks. With one address register (as in
// sdp_ram with IN_REG = 1) every address bit fans out to all of them across the
// chip, and Quartus Pro ignores maxfan on these registers. Here the fan-out is
// built explicitly as a two-level register tree:
//   level 0 : one copy per bank                 (fan-out NS)
//   level 1 : one copy per (bank, slice)        (fan-out ~DB/512 blocks)
// and the bank outputs are selected after the M20K output register.
//
// Read latency raddr -> rdata = 5 cycles:
//   level 0, level 1, M20K read, M20K output register, bank mux
// Writes go through the same two register levels (2 cycles later than issued).
// Read-during-write to the same address returns don't-care.
// -----------------------------------------------------------------------------
module sdp_ram_wide #(
  parameter int W  = 512,
  parameter int D  = 40128,
  parameter int WS = 40,                 // slice width (M20K 512 x 40 mode)
  parameter int DB = 8192,               // bank depth (power of two)
  parameter int AW = $clog2(D)           // derived, do not override
)(
  input  logic          clk,
  input  logic          we,
  input  logic [AW-1:0] waddr,
  input  logic [W-1:0]  wdata,
  input  logic [AW-1:0] raddr,
  output logic [W-1:0]  rdata
);
  localparam int NS  = (W + WS - 1) / WS;
  localparam int WP  = NS * WS;
  localparam int NB  = (D + DB - 1) / DB;
  localparam int BW  = (NB > 1) ? $clog2(DB) : AW;         // address bits inside a bank
  localparam int NBW = (NB > 1) ? $clog2(NB) : 1;

  function automatic logic [NBW-1:0] bank_of(input logic [AW-1:0] a);
    bank_of = (NB > 1) ? NBW'(a >> BW) : '0;
  endfunction

  // ---- level 0: one register set per bank ---------------------------------------
  (* dont_merge *) logic [BW-1:0]  ra0 [NB];
  (* dont_merge *) logic [BW-1:0]  wa0 [NB];
  (* dont_merge *) logic           we0 [NB];
  (* dont_merge *) logic [WP-1:0]  wd0 [NB];
  logic [NBW-1:0] bs0;

  always_ff @(posedge clk) begin
    for (int b = 0; b < NB; b++) begin
      ra0[b] <= BW'(raddr);
      wa0[b] <= BW'(waddr);
      we0[b] <= we && (bank_of(waddr) == NBW'(b));
      wd0[b] <= WP'(wdata);
    end
    bs0 <= bank_of(raddr);
  end

  // ---- bank select follows the read through level 1, M20K read, output register --
  // (the last stage has one copy per slice: one bs3 driving the whole W-bit mux
  //  missed 4 ns)
  logic [NBW-1:0] bs1, bs2;
  (* dont_merge *) logic [NS-1:0][NBW-1:0] bs3;
  always_ff @(posedge clk) begin
    bs1 <= bs0;
    bs2 <= bs1;
    for (int s = 0; s < NS; s++) bs3[s] <= bs2;
  end

  // ---- level 1 + memories -----------------------------------------------------------
  logic [NB-1:0][WP-1:0] bank_q;

  for (genvar b = 0; b < NB; b++) begin : g_bank
    localparam int DBB = (b == NB - 1) ? (D - b * DB) : DB;   // last bank may be shorter
    localparam int BAW = $clog2(DBB);
    for (genvar s = 0; s < NS; s++) begin : g_slice
      (* dont_merge *) logic [BW-1:0] ra1, wa1;
      (* dont_merge *) logic          we1;
      (* dont_merge *) logic [WS-1:0] wd1;
      always_ff @(posedge clk) begin
        ra1 <= ra0[b];
        wa1 <= wa0[b];
        we1 <= we0[b];
        wd1 <= wd0[b][s*WS +: WS];
      end
      sdp_ram #(.W(WS), .D(DBB), .IN_REG(1'b0), .OUT_REG(1'b1)) u_mem (
        .clk(clk), .we(we1), .waddr(BAW'(wa1)), .wdata(wd1),
        .raddr(BAW'(ra1)), .rdata(bank_q[b][s*WS +: WS]));
    end
  end

  // ---- bank mux (per slice) -----------------------------------------------------------------
  logic [WP-1:0] rd_mux;
  always_ff @(posedge clk)
    for (int s = 0; s < NS; s++) rd_mux[s*WS +: WS] <= bank_q[bs3[s]][s*WS +: WS];
  assign rdata = W'(rd_mux);
endmodule
