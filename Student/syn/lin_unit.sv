// -----------------------------------------------------------------------------
// One int8 x int4 linear layer for one token:  y[r] = 2^k * sum_g dot(a_g, W[r,g]) * s[r,g]
//
//   1) load the quantized activation vector, one 128-element group per
//      act_we beat (act_g = group index)
//   2) pulse start with the word address of row 0 / group 0 in the weight
//      memory, the number of rows and groups, and the activation exponent k
//   3) rows come out in order on y_valid / y_row / y_data (float32);
//      done pulses with the last row. A new start is accepted when !busy.
//
// Weight memory layout (as written by student/export.py hex/L??_*_w.hex):
//   word base + r*G + g = K int4 weights of row r, group g; the fp16 scale of
//   the same (r, g) sits at the same address of the scale memory.
//   Both memories are outside this unit; w_data / s_data must be valid RL
//   cycles after w_addr (RL = 1 for rtl/sdp_ram.sv).
//
// One weight word per cycle: in_proj (1160 rows x 2 groups) takes 2320
// cycles + ~20 cycles of pipeline latency.
// -----------------------------------------------------------------------------
module lin_unit #(
  parameter int K     = 128,
  parameter int G_MAX = 4,
  parameter int AW    = 16,       // weight memory address
  parameter int RW    = 12,       // row index
  parameter int RL    = 1,        // weight / scale memory read latency
  parameter int W_K   = 8,
  parameter bit USE_DSP = 1'b0
)(
  input  logic                         clk,
  input  logic                         rst_n,
  // activation load
  input  logic                         act_we,
  input  logic [$clog2(G_MAX)-1:0]     act_g,
  input  logic [K-1:0][7:0]            act_d,
  // job
  input  logic                         start,
  input  logic [AW-1:0]                base,
  input  logic [RW-1:0]                n_rows,
  input  logic [$clog2(G_MAX):0]       n_groups,
  input  logic signed [W_K-1:0]        k,
  output logic                         busy,
  // weight / scale memory read port
  output logic [AW-1:0]                w_addr,
  input  logic [K-1:0][3:0]            w_data,
  input  logic [15:0]                  s_data,
  // results
  output logic                         y_valid,
  output logic [RW-1:0]                y_row,
  output logic [31:0]                  y_data,
  output logic                         done
);
  localparam int GW      = $clog2(G_MAX);
  localparam int DOT_LAT = 2 + $clog2(K);
  localparam int W_ACC   = 12 + $clog2(K);

  // ---- activation buffer ------------------------------------------------------
  logic [G_MAX-1:0][K-1:0][7:0] act_mem;

  always_ff @(posedge clk)
    if (act_we) act_mem[act_g] <= act_d;

  // ---- job registers -----------------------------------------------------------
  logic [RW-1:0]          rows_q;
  logic [GW:0]            groups_q;
  logic signed [W_K-1:0]  k_q;

  // ---- issue: one weight word per cycle ------------------------------------------
  logic          issuing;
  logic [AW-1:0] addr;
  logic [RW-1:0] r_cnt;
  logic [GW:0]   g_cnt;
  logic          last_g, last_r;

  assign last_g = (g_cnt == groups_q - 1'b1);
  assign last_r = (r_cnt == rows_q - 1'b1);
  assign w_addr = addr;

  logic done_c;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      busy    <= 1'b0;
      issuing <= 1'b0;
    end else if (start && !busy) begin
      busy     <= 1'b1;
      issuing  <= 1'b1;
      addr     <= base;
      r_cnt    <= '0;
      g_cnt    <= '0;
      rows_q   <= n_rows;
      groups_q <= n_groups;
      k_q      <= k;
    end else begin
      if (issuing) begin
        addr <= addr + 1'b1;
        if (last_g) begin
          g_cnt <= '0;
          r_cnt <= r_cnt + 1'b1;
          if (last_r) issuing <= 1'b0;
        end else begin
          g_cnt <= g_cnt + 1'b1;
        end
      end
      if (done_c) busy <= 1'b0;
    end
  end

  // ---- align activation group and row info with the memory read latency ----------
  typedef struct packed {
    logic          v;
    logic          first;
    logic          last;
    logic [RW-1:0] row;
  } meta_t;

  meta_t                         m_d [RL];
  logic [RL-1:0][K-1:0][7:0]     a_d;

  always_ff @(posedge clk) begin
    m_d[0] <= '{v: rst_n & issuing, first: (g_cnt == '0), last: last_g, row: r_cnt};
    a_d[0] <= act_mem[g_cnt[GW-1:0]];
    for (int i = 1; i < RL; i++) begin
      m_d[i] <= m_d[i-1];
      a_d[i] <= a_d[i-1];
    end
  end

  // ---- dot product ------------------------------------------------------------------
  logic                    d_valid;
  logic [RW+1:0]           d_tag;
  logic signed [W_ACC-1:0] d_sum;

  lin_dot #(.K(K), .W_TAG(RW + 2), .USE_DSP(USE_DSP)) u_dot (
    .clk(clk), .rst_n(rst_n),
    .in_valid(m_d[RL-1].v),
    .in_tag({m_d[RL-1].first, m_d[RL-1].last, m_d[RL-1].row}),
    .in_w(w_data), .in_a(a_d[RL-1]),
    .out_valid(d_valid), .out_tag(d_tag), .out_sum(d_sum));

  // scale words arrive with the weights; delay them by the dot-product latency
  logic [DOT_LAT-1:0][15:0] s_d;

  always_ff @(posedge clk) begin
    s_d[0] <= s_data;
    for (int i = 1; i < DOT_LAT; i++) s_d[i] <= s_d[i-1];
  end

  // ---- rescale to float32 -------------------------------------------------------------
  logic [RW-1:0] y_row_c;

  lin_scale #(.W_ACC(W_ACC), .W_K(W_K), .W_TAG(RW)) u_scale (
    .clk(clk), .rst_n(rst_n),
    .in_valid(d_valid), .in_first(d_tag[RW+1]), .in_last(d_tag[RW]),
    .in_acc(d_sum), .in_scale(s_d[DOT_LAT-1]), .in_k(k_q), .in_tag(d_tag[RW-1:0]),
    .out_valid(y_valid), .out_tag(y_row_c), .out_y(y_data));

  assign y_row  = y_row_c;
  assign done_c = y_valid && (y_row_c == rows_q - 1'b1);
  assign done   = done_c;
endmodule
