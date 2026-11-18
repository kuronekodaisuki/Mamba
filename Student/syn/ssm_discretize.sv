// -----------------------------------------------------------------------------
// SSM discretization front end (FastMamba step 1 + step 2), one head per cycle
//
//   delta = softplus(dt_raw + dt_bias)          NAU in softplus mode
//   dA    = exp(delta * A)                      NAU in exp mode
//
// delta goes on to form dt*x, dA feeds ssm_scan_unit.in_dA (W_DA / F_DA).
// For decode, Mamba2-130M needs only 24 heads per token per layer, so one
// head per cycle is plenty; FastMamba uses 24 lanes because it targets prefill.
// Latency: 20 cycles.
// -----------------------------------------------------------------------------
module ssm_discretize #(
  parameter int W_RAW = 16, parameter int F_RAW = 10,   // dt_raw, dt_bias
  parameter int W_AA  = 16, parameter int F_AA  = 10,   // A (negative)
  parameter int W_I   = 24, parameter int F_I   = 18,   // NAU format, delta
  parameter int W_DA  = 18, parameter int F_DA  = 16,   // dA output
  parameter int W_TAG = 8,                              // e.g. head index
  parameter bit EXP_NEAR0 = 1'b0,
  parameter bit SP_ORDER2 = 1'b0
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    in_valid,
  input  logic [W_TAG-1:0]        in_tag,
  input  logic signed [W_RAW-1:0] in_dt_raw,
  input  logic signed [W_RAW-1:0] in_dt_bias,
  input  logic signed [W_AA-1:0]  in_A,
  output logic                    out_valid,
  output logic [W_TAG-1:0]        out_tag,
  output logic signed [W_I-1:0]   out_delta,
  output logic signed [W_DA-1:0]  out_dA
);
  localparam int NAU_LAT = 8;
  localparam int LAT     = 1 + NAU_LAT + 2 + NAU_LAT + 1;

  // ---- z = dt_raw + dt_bias, aligned to the NAU format ----------------------
  logic signed [W_I+1:0]   zsh;
  logic signed [W_I-1:0]   zsat, z1;
  logic                    v1;
  logic signed [W_AA-1:0]  A1;

  assign zsh  = (W_I+2)'($signed({in_dt_raw[W_RAW-1], in_dt_raw}) +
                         $signed({in_dt_bias[W_RAW-1], in_dt_bias})) <<< (F_I - F_RAW);
  saturate #(.WI(W_I+2), .WO(W_I)) u_satz (.din(zsh), .dout(zsat));

  always_ff @(posedge clk) begin
    v1 <= rst_n & in_valid;
    z1 <= zsat;
    A1 <= in_A;
  end

  // ---- delta = softplus(z) ---------------------------------------------------
  logic                  vd;
  logic signed [W_I-1:0] delta;

  nau #(.W_I(W_I), .F_I(F_I), .EXP_NEAR0(EXP_NEAR0), .SP_ORDER2(SP_ORDER2)) u_softplus (
    .clk(clk), .rst_n(rst_n), .in_valid(v1), .in_softplus(1'b1), .in_x(z1),
    .out_valid(vd), .out_y(delta));

  logic signed [W_AA-1:0] A_d;
  pipe_delay #(.W(W_AA), .D(NAU_LAT)) u_dA (.clk(clk), .rst_n(1'b1), .d(A1), .q(A_d));

  // ---- delta * A, requantize to the NAU format -------------------------------
  localparam int SH = F_I + F_AA - F_I;   // = F_AA
  logic signed [W_I+W_AA-1:0] p2;
  logic signed [W_I-1:0]      delta2, delta3, x3;
  logic                       v2, v3;
  logic signed [W_I+W_AA-SH:0] pr;
  logic signed [W_I-1:0]      psat;

  always_ff @(posedge clk) begin
    v2     <= rst_n & vd;
    p2     <= delta * A_d;
    delta2 <= delta;
  end

  rshift_round #(.WI(W_I+W_AA), .SHIFT(SH)) u_rp (.din(p2), .dout(pr));
  saturate #(.WI(W_I+W_AA-SH+1), .WO(W_I)) u_satp (.din(pr), .dout(psat));

  always_ff @(posedge clk) begin
    v3     <= rst_n & v2;
    x3     <= psat;
    delta3 <= delta2;
  end

  // ---- dA = exp(delta * A) ---------------------------------------------------
  logic                  ve;
  logic signed [W_I-1:0] e;

  nau #(.W_I(W_I), .F_I(F_I), .EXP_NEAR0(EXP_NEAR0), .SP_ORDER2(SP_ORDER2)) u_exp (
    .clk(clk), .rst_n(rst_n), .in_valid(v3), .in_softplus(1'b0), .in_x(x3),
    .out_valid(ve), .out_y(e));

  logic signed [W_I-1:0] delta_d;
  pipe_delay #(.W(W_I), .D(NAU_LAT)) u_dd (.clk(clk), .rst_n(1'b1), .d(delta3), .q(delta_d));

  // ---- output: dA in the ssm_scan_unit format --------------------------------
  logic signed [W_I-(F_I-F_DA):0] er;
  logic signed [W_DA-1:0]         esat;

  rshift_round #(.WI(W_I), .SHIFT(F_I - F_DA)) u_re (.din(e), .dout(er));
  saturate #(.WI(W_I-(F_I-F_DA)+1), .WO(W_DA)) u_sate (.din(er), .dout(esat));

  always_ff @(posedge clk) begin
    out_valid <= rst_n & ve;
    out_delta <= delta_d;
    out_dA    <= esat;
  end

  pipe_delay #(.W(W_TAG), .D(LAT)) u_tag (.clk(clk), .rst_n(1'b1), .d(in_tag), .q(out_tag));
endmodule


// Simple shift-register delay (D >= 1)
module pipe_delay #(
  parameter int W = 1,
  parameter int D = 1
)(
  input  logic         clk,
  input  logic         rst_n,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  logic [D-1:0][W-1:0] r;
  always_ff @(posedge clk) begin
    if (!rst_n) r <= '0;
    else begin
      r[0] <= d;
      for (int i = 1; i < D; i++) r[i] <= r[i-1];
    end
  end
  assign q = r[D-1];
endmodule
