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
// Latency in_valid -> out_valid : 5 + log2(N) cycles
// State lives in on-chip RAM (NHEADS*P words x N*W_H bits).
// -----------------------------------------------------------------------------
module ssm_scan_unit #(
  parameter int N      = 128,   // d_state (power of two)
  parameter int P      = 64,    // headdim
  parameter int NHEADS = 24,
  // fixed-point formats: W = total bits, F = fractional bits
  parameter int W_H = 24, parameter int F_H = 16,   // state h
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
    if (DEPTH <= 4)
      $fatal(1, "ssm_scan_unit: NHEADS*P must exceed the write-back latency");
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
  // State memory: one word = N lanes of one (head, p) row.
  // A row is read at stage 0 and written back at stage 3; the same row is
  // not read again until the next token (>= DEPTH cycles later), so there is
  // no read-after-write hazard.
  // ---------------------------------------------------------------------------
  // max_depth = 2048 makes Quartus use the 2048 x 10 M20K mode. Without it the
  // 24-bit lanes are mapped one by one into 512 x 40 blocks (60% efficiency).
  (* ramstyle = "M20K, no_rw_check" *)
  logic [N*W_H-1:0]      state_mem [DEPTH];
  logic [N*W_H-1:0]      rd_data;
  logic [AW-1:0]         raddr, waddr;
  logic                  wr_en;
  logic [N-1:0][W_H-1:0] wr_data;

  assign raddr = AW'(in_head) * AW'(P) + AW'(in_p);

  always_ff @(posedge clk) begin
    if (wr_en) state_mem[waddr] <= wr_data;
    rd_data <= state_mem[raddr];
  end

  // ---------------------------------------------------------------------------
  // Stage 1: memory read in flight, register row inputs
  // ---------------------------------------------------------------------------
  logic                  s1_v, s1_bank;
  logic [HW-1:0]         s1_h;
  logic [PW-1:0]         s1_p;
  logic [AW-1:0]         s1_a;
  logic signed [W_A-1:0] s1_dA;
  logic signed [W_X-1:0] s1_xdt, s1_x;
  logic signed [W_D-1:0] s1_D;

  always_ff @(posedge clk) begin
    s1_v    <= rst_n & in_valid;
    s1_bank <= bank;
    s1_h    <= in_head;
    s1_p    <= in_p;
    s1_a    <= raddr;
    s1_dA   <= in_dA;
    s1_xdt  <= in_xdt;
    s1_x    <= in_x;
    s1_D    <= in_D;
  end

  // ---------------------------------------------------------------------------
  // Stage 2: multiplies  dA*h_prev, xdt*B, D*x
  // ---------------------------------------------------------------------------
  logic [N*W_H-1:0] hprev;
  assign hprev = first_q[s1_bank] ? '0 : rd_data;

  logic                   s2_v, s2_bank;
  logic [HW-1:0]          s2_h;
  logic [PW-1:0]          s2_p;
  logic [AW-1:0]          s2_a;
  logic [N-1:0][W_PA-1:0] s2_pa;
  logic [N-1:0][W_PB-1:0] s2_pb;
  logic signed [W_PD-1:0] s2_dx;

  always_ff @(posedge clk) begin
    s2_v    <= rst_n & s1_v;
    s2_bank <= s1_bank;
    s2_h    <= s1_h;
    s2_p    <= s1_p;
    s2_a    <= s1_a;
    s2_dx   <= s1_D * s1_x;
    for (int n = 0; n < N; n++) begin
      s2_pa[n] <= s1_dA  * $signed(hprev[n*W_H +: W_H]);
      s2_pb[n] <= s1_xdt * $signed(B_q[s1_bank][n]);
    end
  end

  // ---------------------------------------------------------------------------
  // Stage 3: requantize, add, saturate -> new state (written back)
  // ---------------------------------------------------------------------------
  logic [N-1:0][W_H-1:0] s2_hnew;

  for (genvar n = 0; n < N; n++) begin : g_upd
    logic signed [WRA-1:0] ta;
    logic signed [WRB-1:0] tb;
    logic signed [WS1-1:0] ts;
    rshift_round #(.WI(W_PA), .SHIFT(SH_A)) u_ra (.din(s2_pa[n]), .dout(ta));
    rshift_round #(.WI(W_PB), .SHIFT(SH_B)) u_rb (.din(s2_pb[n]), .dout(tb));
    assign ts = WS1'(ta) + WS1'(tb);
    saturate #(.WI(WS1), .WO(W_H)) u_sat (.din(ts), .dout(s2_hnew[n]));
  end

  logic                   s3_v, s3_bank;
  logic [HW-1:0]          s3_h;
  logic [PW-1:0]          s3_p;
  logic [AW-1:0]          s3_a;
  logic [N-1:0][W_H-1:0]  s3_hnew;
  logic signed [W_PD-1:0] s3_dx;

  always_ff @(posedge clk) begin
    s3_v    <= rst_n & s2_v;
    s3_bank <= s2_bank;
    s3_h    <= s2_h;
    s3_p    <= s2_p;
    s3_a    <= s2_a;
    s3_hnew <= s2_hnew;
    s3_dx   <= s2_dx;
  end

  assign wr_en   = s3_v;
  assign waddr   = s3_a;
  assign wr_data = s3_hnew;

  // ---------------------------------------------------------------------------
  // Stage 4: h*C
  // ---------------------------------------------------------------------------
  logic                   s4_v;
  logic [HW-1:0]          s4_h;
  logic [PW-1:0]          s4_p;
  logic [N-1:0][W_PC-1:0] s4_pc;
  logic signed [W_PD-1:0] s4_dx;

  always_ff @(posedge clk) begin
    s4_v  <= rst_n & s3_v;
    s4_h  <= s3_h;
    s4_p  <= s3_p;
    s4_dx <= s3_dx;
    for (int n = 0; n < N; n++)
      s4_pc[n] <= $signed(s3_hnew[n]) * $signed(C_q[s3_bank][n]);
  end

  // ---------------------------------------------------------------------------
  // Stage 5..: reduction over N lanes, control delayed alongside
  // ---------------------------------------------------------------------------
  logic signed [W_SUM-1:0] tree_sum;
  adder_tree #(.N(N), .WI(W_PC)) u_tree (.clk(clk), .din(s4_pc), .dout(tree_sum));

  logic [LVL-1:0]           d_v;
  logic [LVL-1:0][HW-1:0]   d_h;
  logic [LVL-1:0][PW-1:0]   d_p;
  logic [LVL-1:0][W_PD-1:0] d_dx;

  always_ff @(posedge clk) begin
    d_v[0]  <= rst_n & s4_v;
    d_h[0]  <= s4_h;
    d_p[0]  <= s4_p;
    d_dx[0] <= s4_dx;
    for (int k = 1; k < LVL; k++) begin
      d_v[k]  <= rst_n & d_v[k-1];
      d_h[k]  <= d_h[k-1];
      d_p[k]  <= d_p[k-1];
      d_dx[k] <= d_dx[k-1];
    end
  end

  // ---------------------------------------------------------------------------
  // Output: y = round(sum h*C) + round(D*x), saturated
  // ---------------------------------------------------------------------------
  logic signed [WRC-1:0] tc;
  logic signed [WRD-1:0] td;
  logic signed [WS2-1:0] ys;
  logic signed [W_Y-1:0] y_c;

  rshift_round #(.WI(W_SUM), .SHIFT(SH_C)) u_rc (.din(tree_sum),     .dout(tc));
  rshift_round #(.WI(W_PD),  .SHIFT(SH_D)) u_rd (.din(d_dx[LVL-1]),  .dout(td));
  assign ys = WS2'(tc) + WS2'(td);
  saturate #(.WI(WS2), .WO(W_Y)) u_ys (.din(ys), .dout(y_c));

  always_ff @(posedge clk) begin
    out_valid <= rst_n & d_v[LVL-1];
    out_head  <= d_h[LVL-1];
    out_p     <= d_p[LVL-1];
    out_y     <= y_c;
  end
endmodule
