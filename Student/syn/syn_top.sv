// -----------------------------------------------------------------------------
// Synthesis wrapper for fmax / resource measurement on Arria 10 (not for the board)
//
// ssm_discretize -> ssm_scan_unit with the student configuration:
//   d_state N = 64, headdim P = 64, 8 heads x 12 layers = 96 state "heads"
//   (the state memory holds all 12 layers: 96*64 rows x 64*20 bits = 7.9 Mb)
// plus one lin_unit with the weight / scale memories of all 12 layers:
//   12 x (in_proj 1160 x 2 + out_proj 256 x 4) = 40128 words x (512 + 16) bits
//
// All wide inputs come from a 1-bit serial shift chain and all outputs are
// XOR-reduced to one pin, so the design fits the package and nothing is
// optimized away. I/O paths are cut in top.sdc.
// -----------------------------------------------------------------------------
module syn_top (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       sin,
  input  logic [7:0] ctl,
  output logic       sout
);
  localparam int N = 64, P = 64, NHEADS = 96;
  localparam int HW = $clog2(NHEADS), PW = $clog2(P);
  localparam int KL = 128;                    // lin_dot lanes
  localparam int WD = 12 * (1160 * 2 + 256 * 4);   // weight words, all layers
  localparam int WA = $clog2(WD);
  localparam int W_IN = 2*N*16 + HW + PW + 3*16 + 8 + 3*16
                      + KL*8 + 2 + WA + 12 + 3 + 8 + KL*4 + 16 + WA;

  // ---- reset / control / serial input --------------------------------------
  logic             rst_q, rst_qq;
  logic [7:0]       ctl_q;
  logic [W_IN-1:0]  sh;

  always_ff @(posedge clk) begin
    rst_q  <= rst_n;
    rst_qq <= rst_q;
    ctl_q  <= ctl;
    sh     <= {sh[W_IN-2:0], sin};
  end

  // ---- slice the chain into module inputs ----------------------------------
  logic [N-1:0][15:0]  B_in, C_in;
  logic [HW-1:0]       in_head;
  logic [PW-1:0]       in_p;
  logic signed [15:0]  in_xdt, in_x, in_D;
  logic [7:0]          d_tag;
  logic signed [15:0]  d_raw, d_bias, d_A;
  logic [KL-1:0][7:0]  a_d;              // lin_unit activation group
  logic [1:0]          a_g;
  logic [WA-1:0]       j_base;
  logic [11:0]         j_rows;
  logic [2:0]          j_groups;
  logic signed [7:0]   j_k;
  logic [KL-1:0][3:0]  m_wd;             // weight memory write port
  logic [15:0]         m_sd;
  logic [WA-1:0]       m_wa;

  assign {B_in, C_in, in_head, in_p, in_xdt, in_x, in_D, d_tag, d_raw, d_bias, d_A,
          a_d, a_g, j_base, j_rows, j_groups, j_k, m_wd, m_sd, m_wa} = sh;

  // ---- discretization (FastMamba NAU, improved options) ---------------------
  logic                d_valid;
  logic [7:0]          d_tag_o;
  logic signed [23:0]  d_delta;
  logic signed [17:0]  d_dA;

  ssm_discretize #(.EXP_NEAR0(1'b1), .SP_ORDER2(1'b1)) u_disc (
    .clk(clk), .rst_n(rst_qq),
    .in_valid(ctl_q[0]), .in_tag(d_tag),
    .in_dt_raw(d_raw), .in_dt_bias(d_bias), .in_A(d_A),
    .out_valid(d_valid), .out_tag(d_tag_o), .out_delta(d_delta), .out_dA(d_dA));

  // ---- selective scan --------------------------------------------------------
  logic                s_valid;
  logic [HW-1:0]       s_head;
  logic [PW-1:0]       s_p;
  logic signed [23:0]  s_y;

  ssm_scan_unit #(.N(N), .P(P), .NHEADS(NHEADS)) u_scan (
    .clk(clk), .rst_n(rst_qq),
    .tok_load(ctl_q[1]), .first_tok(ctl_q[2]), .B_in(B_in), .C_in(C_in),
    .in_valid(ctl_q[3]), .in_head(in_head), .in_p(in_p),
    .in_dA(d_dA), .in_xdt(in_xdt), .in_x(in_x), .in_D(in_D),
    .out_valid(s_valid), .out_head(s_head), .out_p(s_p), .out_y(s_y));

  // ---- weight / scale memories of all layers ---------------------------------
  logic [WA-1:0]       w_addr;
  logic [KL-1:0][3:0]  w_data;
  logic [15:0]         s_data;

  sdp_ram #(.W(KL*4), .D(WD), .IN_REG(1'b1), .OUT_REG(1'b1)) u_wmem (
    .clk(clk), .we(ctl_q[6]), .waddr(m_wa), .wdata(m_wd), .raddr(w_addr), .rdata(w_data));
  sdp_ram #(.W(16), .D(WD), .IN_REG(1'b1), .OUT_REG(1'b1)) u_smem (
    .clk(clk), .we(ctl_q[6]), .waddr(m_wa), .wdata(m_sd), .raddr(w_addr), .rdata(s_data));

  // ---- linear layer unit -------------------------------------------------------
  logic                q_valid, q_done, q_busy;
  logic [11:0]         q_row;
  logic [31:0]         q_y;

  lin_unit #(.K(KL), .G_MAX(4), .AW(WA), .RW(12), .RL(3)) u_lin (
    .clk(clk), .rst_n(rst_qq),
    .act_we(ctl_q[4]), .act_g(a_g), .act_d(a_d),
    .start(ctl_q[5]), .base(j_base), .n_rows(j_rows), .n_groups(j_groups), .k(j_k),
    .busy(q_busy),
    .w_addr(w_addr), .w_data(w_data), .s_data(s_data),
    .y_valid(q_valid), .y_row(q_row), .y_data(q_y), .done(q_done));

  // ---- keep every output alive ----------------------------------------------
  always_ff @(posedge clk)
    sout <= ^{d_valid, d_tag_o, d_delta, s_valid, s_head, s_p, s_y,
              q_valid, q_done, q_busy, q_row, q_y};
endmodule
