// -----------------------------------------------------------------------------
// Mamba2 output gate:  g = y * SiLU(z)   (model/vec_golden.py:gate_fx, bit exact)
//
//   y : SSM output, Q.16 (W_Y bits, from ssm_scan_unit)
//   z : in_proj output, Q.12 (24 bits)
//   g : Q.16, saturated to 32 bits (input format of norm_q8)
//
// One element per cycle, latency 8.
// -----------------------------------------------------------------------------
module gate_unit #(
  parameter int W_Y   = 24,
  parameter int W_TAG = 16
)(
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  in_valid,
  input  logic signed [W_Y-1:0] in_y,
  input  logic signed [23:0]    in_z,
  input  logic [W_TAG-1:0]      in_tag,
  output logic                  out_valid,
  output logic signed [31:0]    out_g,
  output logic [W_TAG-1:0]      out_tag
);
  logic                  sv;
  logic signed [23:0]    sz;
  logic [W_TAG-1:0]      st;

  silu_unit #(.W(24), .W_TAG(W_TAG)) u_silu (
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_x(in_z), .in_tag(in_tag),
    .out_valid(sv), .out_y(sz), .out_tag(st));

  // y waits for SiLU(z) (latency 6)
  logic [5:0][W_Y-1:0] yd;
  always_ff @(posedge clk) begin
    yd[0] <= in_y;
    for (int i = 1; i < 6; i++) yd[i] <= yd[i-1];
  end

  logic                      v1;
  logic signed [W_Y+23:0]    p1;
  logic [W_TAG-1:0]          t1;

  always_ff @(posedge clk) begin
    v1 <= rst_n & sv;
    p1 <= $signed(yd[5]) * sz;
    t1 <= st;
  end

  logic signed [W_Y+12:0] r1;
  logic signed [31:0]     g1;
  rshift_round #(.WI(W_Y + 24), .SHIFT(12)) u_r (.din(p1), .dout(r1));
  saturate #(.WI(W_Y + 13), .WO(32)) u_s (.din(r1), .dout(g1));

  always_ff @(posedge clk) begin
    out_valid <= rst_n & v1;
    out_g     <= g1;
    out_tag   <= t1;
  end
endmodule
