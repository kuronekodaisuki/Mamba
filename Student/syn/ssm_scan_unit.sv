// -----------------------------------------------------------------------------
// Mamba2 selective-scan (SSM) state-update unit
//
// For each token, for each (head, p) row, N state lanes in parallel:
//   h[n] <- dA * h[n] + (dt*x) * B[n]        (dA = exp(dt*A), per head)
//   y    <- sum_n h[n] * C[n] + D * x
//
// Usage per token:
//   1) pulse tok_load for one cycle with B_in, C_in, first_tok
//      (B, C shared across heads: Mamba2 with ngroups = 1)
//   2) stream NHEADS*P rows, one per cycle (in_valid), in any fixed order
//   tok_load and in_valid must not be asserted in the same cycle.
//   first_tok = 1 treats the previous state as zero (sequence start),
//   so the state memory never needs to be cleared.
//
// Pipeline (every multiplier has registered operands and a registered
// product, so it maps onto DSP input / output registers; the M20K read has
// address and output registers):
//   s1  M20K address register        (read in flight)
//   s2  M20K output register
//   s2b relay register (M20K and DSP columns are far apart)
//   s3  operands: h_prev (or 0), B[n] of this token, dA, dt*x, D, x
//   s4  products dA*h, xdt*B, D*x
//   s5  requantize + add
//   s6  saturate -> new state, written back;  C[n] selected
//   s7  products h*C
//   s8.. adder tree (log2 N), then 2 output stages
// Latency in_valid -> out_valid : 10 + log2(N) cycles
// State lives in on-chip RAM (NHEADS*P words x N*W_H bits).
// -----------------------------------------------------------------------------
module ssm_scan_unit #(
  parameter int N      = 128,   // d_state (power of two)
  parameter int P      = 64,    // headdim
  parameter int NHEADS = 24,
  // fixed-point formats: W = total bits, F = fractional bits
  parameter int W_H = 20, parameter int F_H = 16,   // state h (20 bits: 2 lanes per 40-bit M20K word)
  parameter int W_A = 18, parameter int F_A = 16,   // dA in [0,1]
  parameter int W_X = 16, parameter int F_X  = 10,  // x and dt*x (width)
  parameter int F_XD = 14,                          // dt*x fraction (dt can be ~1e-3)
  parameter int W_B = 16, parameter int F_B = 12,   // B and C
  parameter int W_D = 16, parameter int F_D = 12,   // D
  parameter int W_Y = 24, parameter int F_Y = 16    // y
)(
  input  logic                      clk,
  input  logic                      rst_n,
  // token header
  input  logic                      tok_load,
  input  logic                      first_tok,
  input  logic [N-1:0][W_B-1:0]     B_in,
  input  logic [N-1:0][W_B-1:0]     C_in,
  // row stream
  input  logic                      in_valid,
  input  logic [$clog2(NHEADS)-1:0] in_head,
  input  logic [$clog2(P)-1:0]      in_p,
  input  logic signed [W_A-1:0]     in_dA,
  input  logic signed [W_X-1:0]     in_xdt,
  input  logic signed [W_X-1:0]     in_x,
  input  logic signed [W_D-1:0]     in_D,
  // output stream (same order as input)
  output logic                      out_valid,
  output logic [$clog2(NHEADS)-1:0] out_head,
  output logic [$clog2(P)-1:0]      out_p,
  output logic signed [W_Y-1:0]     out_y
);
  localparam int HW    = $clog2(NHEADS);
  localparam int PW    = $clog2(P);
  localparam int DEPTH = NHEADS * P;
  localparam int AW    = $clog2(DEPTH);
  localparam int LVL   = $clog2(N);
  localparam int WB_LAT = 7;                 // read address -> write-back

  // product widths and requantization shifts
  localparam int W_PA = W_A + W_H,  SH_A = F_A;                // dA*h   -> F_H
  localparam int W_PB = W_X + W_B,  SH_B = F_XD + F_B - F_H;   // xdt*B  -> F_H
  localparam int W_PC = W_H + W_B,  SH_C = F_H + F_B - F_Y;    // sum h*C -> F_Y
  localparam int W_PD = W_D + W_X,  SH_D = F_D + F_X - F_Y;    // D*x    -> F_Y
  localparam int W_SUM = W_PC + LVL;

  localparam int WRA = W_PA  - SH_A + 1;
  localparam int WRB = W_PB  - SH_B + 1;
  localparam int WS1 = ((WRA > WRB) ? WRA : WRB) + 1;
  localparam int WRC = W_SUM - SH_C + 1;
  localparam int WRD = W_PD  - SH_D + 1;
  localparam int WS2 = ((WRC > WRD) ? WRC : WRD) + 1;

  // synthesis translate_off
  initial begin
    if (SH_A < 1 || SH_B < 1 || SH_C < 1 || SH_D < 1)
      $fatal(1, "ssm_scan_unit: every requantization shift must be >= 1");
    if ((1 << LVL) != N)
      $fatal(1, "ssm_scan_unit: N must be a power of two");
    // the next read of a row comes DEPTH + 1 cycles later (tok_load in between)
    if (DEPTH < WB_LAT)
      $fatal(1, "ssm_scan_unit: NHEADS*P must not be below the write-back latency");
  end
  // synthesis translate_on

  // ---------------------------------------------------------------------------
  // Token header, ping-pong banked so rows of the previous token that are
  // still in the pipeline keep using their own B / C.
  // ---------------------------------------------------------------------------
  logic [1:0][N-1:0][W_B-1:0] B_q, C_q;
  logic [1:0]                 first_q;
  logic                       bank;     // bank of the most recently loaded token

  always_ff @(posedge clk) begin
    if (!rst_n)        bank <= 1'b0;
    else if (tok_load) bank <= ~bank;
  end

  always_ff @(posedge clk) begin
    if (tok_load) begin
      B_q[~bank]     <= B_in;
      C_q[~bank]     <= C_in;
      first_q[~bank] <= first_tok;
    end
  end

  // ---------------------------------------------------------------------------
  // Row control / operands travelling with the data
  // ---------------------------------------------------------------------------
  typedef struct packed {
    logic                  v;
    logic                  bank;
    logic [HW-1:0]         h;
    logic [PW-1:0]         p;
    logic [AW-1:0]         a;
    logic signed [W_A-1:0] dA;
    logic signed [W_X-1:0] xdt;
    logic signed [W_X-1:0] x;
    logic signed [W_D-1:0] D;
  } row_t;

  row_t r0, r0d, r1, r2, r3, r4, r5, r6;
  logic [AW-1:0] raddr;

  assign raddr = AW'(in_head) * AW'(P) + AW'(in_p);
  assign r0    = '{v: in_valid, bank: bank, h: in_head, p: in_p, a: raddr,
                   dA: in_dA, xdt: in_xdt, x: in_x, D: in_D};

  always_ff @(posedge clk) begin
    r0d <= r0; r1 <= r0d; r2 <= r1;  r3 <= r2;  r4 <= r3;  r5 <= r4;  r6 <= r5;
    r0d.v <= rst_n & r0.v;
    r1.v <= rst_n & r0d.v;
    r2.v <= rst_n & r1.v;
    r3.v <= rst_n & r2.v;
    r4.v <= rst_n & r3.v;
    r5.v <= rst_n & r4.v;
    r6.v <= rst_n & r5.v;
  end

  // ---------------------------------------------------------------------------
  // State memory: one word = N lanes of one (head, p) row.
  // A row is read at s0 and written back WB_LAT cycles later; the same row is not read
  // again until the next token (DEPTH cycles later), so there is no
  // read-after-write hazard.
  //
  // The word is split into NMEM memories of MW bits so that each one matches
  // the 40-bit M20K width (W_H = 20: 2 lanes per memory, 32 memories,
  // 12 blocks each = 384 M20K). One 1280-bit memory was mapped by Quartus at
  // only 60% efficiency.
  // ---------------------------------------------------------------------------
  localparam int M20K_W = 40;
  localparam int LPM  = (M20K_W % W_H == 0 && N % (M20K_W / W_H) == 0) ? M20K_W / W_H : 1;
  localparam int NMEM = N / LPM;               // memories
  localparam int MW   = LPM * W_H;             // bits per memory

  wire  [N*W_H-1:0]      rd_s1;                // read data (address registered)
  logic [N*W_H-1:0]      rd_s2;                // output register
  logic [N*W_H-1:0]      rd_s2b;               // relay register
  logic                  wr_en;
  logic [AW-1:0]         waddr;
  logic [N-1:0][W_H-1:0] wr_data;
  logic [N*W_H-1:0]      wr_flat;

  assign wr_flat = wr_data;

  for (genvar j = 0; j < NMEM; j++) begin : g_mem
    sdp_ram #(.W(MW), .D(DEPTH)) u_ram (
      .clk(clk), .we(wr_en), .waddr(waddr), .wdata(wr_flat[j*MW +: MW]),
      .raddr(raddr), .rdata(rd_s1[j*MW +: MW]));
  end

  always_ff @(posedge clk) begin
    rd_s2  <= rd_s1;
    rd_s2b <= rd_s2;
  end

  // ---------------------------------------------------------------------------
  // Per-lane copies of the bank select and of the "first token" clear, one
  // stage early, so no single register fans out to all N lanes (dont_merge
  // keeps Quartus from merging the copies back into one register).
  // ---------------------------------------------------------------------------
  (* dont_merge *) logic [N-1:0] sel_s2;    // bank of the row now in s2
  (* dont_merge *) logic [N-1:0] clr_s2;    // first token of that bank
  (* dont_merge *) logic [N-1:0] sel_s5;    // bank of the row now in s5

  always_ff @(posedge clk)
    for (int n = 0; n < N; n++) begin
      sel_s2[n] <= r1.bank;
      clr_s2[n] <= first_q[r1.bank];
      sel_s5[n] <= r4.bank;
    end

  // ---------------------------------------------------------------------------
  // s3: multiplier operands
  // ---------------------------------------------------------------------------
  logic [N-1:0][W_H-1:0] op_h;
  logic [N-1:0][W_B-1:0] op_B;
  logic [N-1:0][W_H-1:0] rd_lanes;

  assign rd_lanes = rd_s2b;

  always_ff @(posedge clk)
    for (int n = 0; n < N; n++) begin
      op_h[n] <= clr_s2[n] ? '0 : rd_lanes[n];
      op_B[n] <= B_q[sel_s2[n]][n];
    end

  // ---------------------------------------------------------------------------
  // s4: products  dA*h_prev, xdt*B, D*x
  // ---------------------------------------------------------------------------
  logic [N-1:0][W_PA-1:0] s4_pa;
  logic [N-1:0][W_PB-1:0] s4_pb;
  logic signed [W_PD-1:0] s4_dx;

  always_ff @(posedge clk) begin
    s4_dx <= r3.D * r3.x;
    for (int n = 0; n < N; n++) begin
      s4_pa[n] <= r3.dA  * $signed(op_h[n]);
      s4_pb[n] <= r3.xdt * $signed(op_B[n]);
    end
  end

  // ---------------------------------------------------------------------------
  // s5: requantize and add (unsaturated sum registered)
  // ---------------------------------------------------------------------------
  logic [N-1:0][WS1-1:0] sum_c, s5_sum;

  for (genvar n = 0; n < N; n++) begin : g_upd
    logic signed [WRA-1:0] ta;
    logic signed [WRB-1:0] tb;
    rshift_round #(.WI(W_PA), .SHIFT(SH_A)) u_ra (.din(s4_pa[n]), .dout(ta));
    rshift_round #(.WI(W_PB), .SHIFT(SH_B)) u_rb (.din(s4_pb[n]), .dout(tb));
    assign sum_c[n] = WS1'(ta) + WS1'(tb);
  end

  logic signed [W_PD-1:0] s5_dx;

  always_ff @(posedge clk) begin
    s5_sum <= sum_c;
    s5_dx  <= s4_dx;
  end

  // ---------------------------------------------------------------------------
  // s6: saturate -> new state (written back); select C
  // ---------------------------------------------------------------------------
  logic [N-1:0][W_H-1:0] hnew_c;

  for (genvar n = 0; n < N; n++) begin : g_sat
    saturate #(.WI(WS1), .WO(W_H)) u_sat (.din(s5_sum[n]), .dout(hnew_c[n]));
  end

  logic [N-1:0][W_H-1:0]  s6_hs;
  logic [N-1:0][W_B-1:0]  s6_C;
  logic signed [W_PD-1:0] s6_dx;

  always_ff @(posedge clk) begin
    s6_hs <= hnew_c;
    s6_dx <= s5_dx;
    for (int n = 0; n < N; n++)
      s6_C[n] <= C_q[sel_s5[n]][n];
  end

  assign wr_en   = r6.v;
  assign waddr   = r6.a;
  assign wr_data = s6_hs;

  // ---------------------------------------------------------------------------
  // s7: h*C
  // ---------------------------------------------------------------------------
  logic [N-1:0][W_PC-1:0] s7_pc;
  logic signed [W_PD-1:0] s7_dx;
  logic                   s7_v;
  logic [HW-1:0]          s7_h;
  logic [PW-1:0]          s7_p;

  always_ff @(posedge clk) begin
    s7_v  <= rst_n & r6.v;
    s7_h  <= r6.h;
    s7_p  <= r6.p;
    s7_dx <= s6_dx;
    for (int n = 0; n < N; n++)
      s7_pc[n] <= $signed(s6_hs[n]) * $signed(s6_C[n]);
  end

  // ---------------------------------------------------------------------------
  // Reduction over N lanes, control delayed alongside
  // ---------------------------------------------------------------------------
  logic signed [W_SUM-1:0] tree_sum;
  adder_tree #(.N(N), .WI(W_PC)) u_tree (.clk(clk), .din(s7_pc), .dout(tree_sum));

  logic [LVL-1:0]           d_v;
  logic [LVL-1:0][HW-1:0]   d_h;
  logic [LVL-1:0][PW-1:0]   d_p;
  logic [LVL-1:0][W_PD-1:0] d_dx;

  always_ff @(posedge clk) begin
    d_v[0]  <= rst_n & s7_v;
    d_h[0]  <= s7_h;
    d_p[0]  <= s7_p;
    d_dx[0] <= s7_dx;
    for (int k = 1; k < LVL; k++) begin
      d_v[k]  <= rst_n & d_v[k-1];
      d_h[k]  <= d_h[k-1];
      d_p[k]  <= d_p[k-1];
      d_dx[k] <= d_dx[k-1];
    end
  end

  // ---------------------------------------------------------------------------
  // Output: y = round(sum h*C) + round(D*x), saturated (two stages)
  // ---------------------------------------------------------------------------
  logic signed [WRC-1:0] tc, o1_c;
  logic signed [WRD-1:0] td, o1_d;
  logic                  o1_v;
  logic [HW-1:0]         o1_h;
  logic [PW-1:0]         o1_p;

  rshift_round #(.WI(W_SUM), .SHIFT(SH_C)) u_rc (.din(tree_sum),    .dout(tc));
  rshift_round #(.WI(W_PD),  .SHIFT(SH_D)) u_rd (.din(d_dx[LVL-1]), .dout(td));

  always_ff @(posedge clk) begin
    o1_v <= rst_n & d_v[LVL-1];
    o1_h <= d_h[LVL-1];
    o1_p <= d_p[LVL-1];
    o1_c <= tc;
    o1_d <= td;
  end

  logic signed [WS2-1:0] ys;
  logic signed [W_Y-1:0] y_c;
  assign ys = WS2'(o1_c) + WS2'(o1_d);
  saturate #(.WI(WS2), .WO(W_Y)) u_ys (.din(ys), .dout(y_c));

  always_ff @(posedge clk) begin
    out_valid <= rst_n & o1_v;
    out_head  <= o1_h;
    out_p     <= o1_p;
    out_y     <= y_c;
  end
endmodule
