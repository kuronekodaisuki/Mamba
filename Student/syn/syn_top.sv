// -----------------------------------------------------------------------------
// Synthesis wrapper for fmax / resource measurement on Arria 10 (not for the board)
//
// ssm_discretize -> ssm_scan_unit with the student configuration, plus one
// lin_dot (128 lanes, ALM multipliers) -> lin_scale:
//   d_state N = 64, headdim P = 64, 8 heads x 12 layers = 96 state "heads"
//   (the state memory holds all 12 layers: 96*64 rows x 64*24 bits = 9.4 Mb)
//
// All wide inputs come from a 1-bit serial shift chain and all outputs are
// XOR-reduced to one pin, so the design fits the package and nothing is
// optimized away. I/O paths are cut in top.sdc.
// -----------------------------------------------------------------------------
module syn_top (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       sin,
  input  logic [3:0] ctl,
  output logic       sout
);
  localparam int N = 64, P = 64, NHEADS = 96;
  localparam int HW = $clog2(NHEADS), PW = $clog2(P);
  localparam int KL = 128;                    // lin_dot lanes
  localparam int W_IN = 2*N*16 + HW + PW + 3*16 + 8 + 3*16 + KL*4 + KL*8 + 16 + 8;

  // ---- reset / control / serial input --------------------------------------
  logic             rst_q, rst_qq;
  logic [3:0]       ctl_q;
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
  logic [KL-1:0][3:0]  l_w;
  logic [KL-1:0][7:0]  l_a;
  logic [15:0]         l_s;
  logic signed [7:0]   l_k;

  assign {B_in, C_in, in_head, in_p, in_xdt, in_x, in_D, d_tag, d_raw, d_bias, d_A, l_w, l_a, l_s, l_k} = sh;

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

  // ---- int8 x int4 group dot product (ALM multipliers) ----------------------
  logic                l_valid;
  logic [15:0]         l_tag;
  logic signed [18:0]  l_sum;

  lin_dot #(.K(KL), .USE_DSP(1'b0)) u_lin (
    .clk(clk), .rst_n(rst_qq), .in_valid(ctl_q[0]), .in_tag(16'(d_tag)),
    .in_w(l_w), .in_a(l_a),
    .out_valid(l_valid), .out_tag(l_tag), .out_sum(l_sum));

  // ---- row rescale to float32 ------------------------------------------------
  logic                q_valid;
  logic [15:0]         q_tag;
  logic [31:0]         q_y;

  lin_scale u_lsc (
    .clk(clk), .rst_n(rst_qq), .in_valid(l_valid), .in_first(ctl_q[1]), .in_last(ctl_q[2]),
    .in_acc(l_sum), .in_scale(l_s), .in_k(l_k), .in_tag(l_tag),
    .out_valid(q_valid), .out_tag(q_tag), .out_y(q_y));

  // ---- keep every output alive ----------------------------------------------
  always_ff @(posedge clk)
    sout <= ^{d_valid, d_tag_o, d_delta, s_valid, s_head, s_p, s_y, q_valid, q_tag, q_y};
endmodule
