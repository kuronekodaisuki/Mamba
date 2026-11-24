// -----------------------------------------------------------------------------
// Depthwise causal conv1d (4 taps) + bias + SiLU for the Mamba2 xBC channels
// (model/vec_golden.py: silu_fx(conv_fx(...)), bit exact)
//
//   acc = w0*x[t-3] + w1*x[t-2] + w2*x[t-1] + w3*x[t]       x: Q.12 (24 bits)
//   y   = sat24(round(acc * s) + b)                         s: fp16, b: Q.12
//   out = SiLU(y)
//
// One channel per beat (in_layer, in_ch, in_x), any order, bubbles allowed.
// The last three inputs of every (layer, channel) are kept in an internal
// state memory; in_first marks the first token of a sequence (history = 0).
// A (layer, channel) must not come back within ~10 cycles (it comes back
// one token later in a real stream).
//
// Parameters come from an external memory, one 72-bit word per (layer, ch)
// at address layer * NCH + ch, valid RL cycles after p_addr:
//   p_data = {b[71:48] (Q.12), s[47:32] (fp16), w3, w2, w1, w0 [31:0] (int8)}
//   (w3 multiplies the current token, as in student/export.py hex/L??_conv_w.hex)
//
// Latency in -> out: RL + 10 + 7 (SiLU) cycles.
// -----------------------------------------------------------------------------
module conv_unit #(
  parameter int NCH    = 640,
  parameter int NLAYER = 12,
  parameter int RL     = 1,          // >= 1
  parameter int W_TAG  = 16,
  parameter int AW     = $clog2(NCH * NLAYER)   // derived, do not override
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    in_valid,
  input  logic                    in_first,
  input  logic [$clog2(NLAYER)-1:0] in_layer,
  input  logic [$clog2(NCH)-1:0]  in_ch,
  input  logic signed [23:0]      in_x,
  input  logic [W_TAG-1:0]        in_tag,
  // parameter memory
  output logic [AW-1:0]           p_addr,
  input  logic [71:0]             p_data,
  // result
  output logic                    out_valid,
  output logic signed [23:0]      out_y,
  output logic [W_TAG-1:0]        out_tag
);
  // ---- address, state read -------------------------------------------------------
  logic [AW-1:0] addr0;
  assign addr0  = AW'(in_layer) * AW'(NCH) + AW'(in_ch);
  assign p_addr = addr0;

  logic [71:0] st_rd;                 // {x[t-1], x[t-2], x[t-3]}, 24 bits each
  logic        st_we;
  logic [AW-1:0] st_wa;
  logic [71:0] st_wd;

  sdp_ram #(.W(72), .D(NCH * NLAYER)) u_state (
    .clk(clk), .we(st_we), .waddr(st_wa), .wdata(st_wd), .raddr(addr0), .rdata(st_rd));

  // input fields delayed by RL
  logic [RL-1:0]              a_v, a_f;
  logic [RL-1:0][AW-1:0]      a_addr;
  logic [RL-1:0][23:0]        a_x;
  logic [RL-1:0][W_TAG-1:0]   a_tag;

  always_ff @(posedge clk) begin
    a_v[0]    <= rst_n & in_valid;
    a_f[0]    <= in_first;
    a_addr[0] <= addr0;
    a_x[0]    <= in_x;
    a_tag[0]  <= in_tag;
    for (int i = 1; i < RL; i++) begin
      a_v[i]    <= rst_n & a_v[i-1];
      a_f[i]    <= a_f[i-1];
      a_addr[i] <= a_addr[i-1];
      a_x[i]    <= a_x[i-1];
      a_tag[i]  <= a_tag[i-1];
    end
  end

  // align the state (latency 1) with the parameters (latency RL).
  // With RL > 1 the "first token" clear is applied while registering the M20K
  // output, so the multipliers get a fabric register (no M20K -> LUT -> DSP path).
  logic [71:0] st_al;
  logic [71:0] hist;
  if (RL > 1) begin : g_al
    logic [RL-2:0][71:0] d;
    always_ff @(posedge clk) begin
      d[0] <= a_f[0] ? 72'd0 : st_rd;
      for (int i = 1; i < RL - 1; i++) d[i] <= d[i-1];
    end
    assign st_al = d[RL-2];
    assign hist  = st_al;
  end else begin : g_no_al
    assign st_al = st_rd;
    assign hist  = a_f[RL-1] ? 72'd0 : st_al;
  end

  // ---- c1: taps, products, state update -----------------------------------------------
  logic signed [23:0] h0, h1, h2, xc;
  assign h0   = hist[23:0];
  assign h1   = hist[47:24];
  assign h2   = hist[71:48];
  assign xc   = a_x[RL-1];

  logic [4:0]  ex;
  logic [10:0] mm;
  assign ex = p_data[46:42];
  assign mm = (ex == 5'd0) ? {1'b0, p_data[41:32]} : {1'b1, p_data[41:32]};

  logic                v1;
  logic signed [31:0]  m0, m1, m2, m3;
  logic signed [11:0]  sm1;
  logic [4:0]          sh1;
  logic signed [23:0]  b1;
  logic [W_TAG-1:0]    t1;

  always_ff @(posedge clk) begin
    v1  <= rst_n & a_v[RL-1];
    m0  <= $signed(p_data[7:0])   * h0;
    m1  <= $signed(p_data[15:8])  * h1;
    m2  <= $signed(p_data[23:16]) * h2;
    m3  <= $signed(p_data[31:24]) * xc;
    sm1 <= p_data[47] ? -$signed({1'b0, mm}) : $signed({1'b0, mm});
    sh1 <= (ex == 5'd0) ? 5'd0 : ex - 5'd1;
    b1  <= $signed(p_data[71:48]);
    t1  <= a_tag[RL-1];
    st_we <= rst_n & a_v[RL-1];
    st_wa <= a_addr[RL-1];
    st_wd <= {xc, h2, h1};
  end

  // ---- c2: sum of the four taps, two adder stages ---------------------------------------------
  // (Quartus packs m0 + m1 into the DSP; the rest of a one-cycle 4-input sum and
  // the next multiplier did not fit 4 ns; preserve keeps acc2 out of the DSP)
  logic                v2a;
  logic signed [32:0]  s01, s23;
  logic signed [11:0]  sm2a;
  logic [4:0]          sh2a;
  logic signed [23:0]  b2a;
  logic [W_TAG-1:0]    t2a;

  always_ff @(posedge clk) begin
    v2a  <= rst_n & v1;
    s01  <= 33'(m0) + 33'(m1);
    s23  <= 33'(m2) + 33'(m3);
    sm2a <= sm1;
    sh2a <= sh1;
    b2a  <= b1;
    t2a  <= t1;
  end

  logic                v2;
  (* preserve *) logic signed [33:0] acc2;
  logic signed [11:0]  sm2;
  logic [4:0]          sh2;
  logic signed [23:0]  b2;
  logic [W_TAG-1:0]    t2;

  always_ff @(posedge clk) begin
    v2   <= rst_n & v2a;
    acc2 <= 34'(s01) + 34'(s23);
    sm2  <= sm2a;
    sh2  <= sh2a;
    b2   <= b2a;
    t2   <= t2a;
  end

  // ---- c2b: multiplier operands (Quartus packed acc2 into the DSP input register,
  //      leaving DSP adder -> 34-bit add -> DSP input in one cycle) ------------------------------
  logic                v2b;
  logic signed [17:0]  lo2b;
  logic signed [16:0]  hi2b;
  logic signed [11:0]  sm2b;
  logic [4:0]          sh2b;
  logic signed [23:0]  b2b;
  logic [W_TAG-1:0]    t2b;

  always_ff @(posedge clk) begin
    v2b  <= rst_n & v2;
    lo2b <= $signed({1'b0, acc2[16:0]});
    hi2b <= $signed(acc2[33:17]);
    sm2b <= sm2;
    sh2b <= sh2;
    b2b  <= b2;
    t2b  <= t2;
  end

  // ---- c3: acc * m as two partial products (every DSP input <= 18 bits) ---------------------
  logic                v3;
  logic signed [29:0]  plo3;        // 18 x 12
  logic signed [28:0]  phi3;        // 17 x 12
  logic [4:0]          sh3;
  logic signed [23:0]  b3;
  logic [W_TAG-1:0]    t3;

  always_ff @(posedge clk) begin
    v3   <= rst_n & v2b;
    plo3 <= lo2b * sm2b;
    phi3 <= hi2b * sm2b;
    sh3  <= sh2b;
    b3   <= b2b;
    t3   <= t2b;
  end

  logic                v4;
  logic signed [45:0]  am4;
  logic [4:0]          sh4;
  logic signed [23:0]  b4;
  logic [W_TAG-1:0]    t4;

  always_ff @(posedge clk) begin
    v4  <= rst_n & v3;
    am4 <= (46'(phi3) <<< 17) + 46'(plo3);
    sh4 <= sh3;
    b4  <= b3;
    t4  <= t3;
  end

  // ---- c5: shift by sh - 24 (right: add 2^(r-1) first) ---------------------------------------
  logic                v5, left5;
  logic signed [51:0]  a5;
  logic [4:0]          r5;           // right shift amount (1..24) or left (0..5)
  logic signed [23:0]  b5;
  logic [W_TAG-1:0]    t5;

  always_ff @(posedge clk) begin
    v5    <= rst_n & v4;
    left5 <= (sh4 >= 5'd24);
    if (sh4 >= 5'd24) begin
      r5 <= sh4 - 5'd24;
      a5 <= 52'(am4);
    end else begin
      r5 <= 5'd24 - sh4;
      a5 <= 52'(am4) + (52'sd1 <<< (5'd23 - sh4));
    end
    b5 <= b4;
    t5 <= t4;
  end

  logic                v6;
  logic signed [51:0]  s6;
  logic signed [23:0]  b6;
  logic [W_TAG-1:0]    t6;

  always_ff @(posedge clk) begin
    v6 <= rst_n & v5;
    s6 <= left5 ? (a5 <<< r5) : (a5 >>> r5);
    b6 <= b5;
    t6 <= t5;
  end

  // ---- c7: + bias, saturate --------------------------------------------------------------------
  logic signed [52:0]  y7w;
  logic signed [23:0]  y7c;
  assign y7w = 53'(s6) + 53'(b6);
  saturate #(.WI(53), .WO(24)) u_s (.din(y7w), .dout(y7c));

  logic                v7;
  logic signed [23:0]  y7;
  logic [W_TAG-1:0]    t7;

  always_ff @(posedge clk) begin
    v7 <= rst_n & v6;
    y7 <= y7c;
    t7 <= t6;
  end

  // ---- SiLU -----------------------------------------------------------------------------------
  silu_unit #(.W(24), .W_TAG(W_TAG)) u_silu (
    .clk(clk), .rst_n(rst_n), .in_valid(v7), .in_x(y7), .in_tag(t7),
    .out_valid(out_valid), .out_y(out_y), .out_tag(out_tag));
endmodule
