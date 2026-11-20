// -----------------------------------------------------------------------------
// One-group dot product for the int8 x int4 linear layers
//
//   out_sum = sum_{k<K} a[k] * w[k]        (exact, signed)
//
// One weight word per cycle: K int4 weights (one 128-element group of one
// output row, the format written by student/export.py: lane k in bits
// [4k+3:4k]) and the matching K int8 activations. The caller walks over the
// rows and groups; scaling by the group scale and the activation scale is
// done downstream.
//
// Pipeline: operand registers -> product registers -> adder tree (log2 K)
// Latency in_valid -> out_valid : 2 + log2(K) cycles
//
// USE_DSP = 0 builds the K small multipliers in ALMs (keeps DSPs free),
// USE_DSP = 1 lets Quartus put them in DSP blocks.
// -----------------------------------------------------------------------------
module lin_dot #(
  parameter int K       = 128,
  parameter int W_A     = 8,
  parameter int W_W     = 4,
  parameter int W_TAG   = 16,
  parameter bit USE_DSP = 1'b0,
  parameter int W_P     = W_A + W_W,          // derived, do not override
  parameter int W_O     = W_P + $clog2(K)     // derived, do not override
)(
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 in_valid,
  input  logic [W_TAG-1:0]     in_tag,
  input  logic [K-1:0][W_W-1:0] in_w,
  input  logic [K-1:0][W_A-1:0] in_a,
  output logic                 out_valid,
  output logic [W_TAG-1:0]     out_tag,
  output logic signed [W_O-1:0] out_sum
);
  localparam int LVL = $clog2(K);

  // synthesis translate_off
  initial if ((1 << LVL) != K) $fatal(1, "lin_dot: K must be a power of two");
  // synthesis translate_on

  // ---- stage 1: operand registers --------------------------------------------
  logic [K-1:0][W_W-1:0] w1;
  logic [K-1:0][W_A-1:0] a1;

  always_ff @(posedge clk) begin
    w1 <= in_w;
    a1 <= in_a;
  end

  // ---- stage 2: products ------------------------------------------------------
  logic [K-1:0][W_P-1:0] p2;

  for (genvar k = 0; k < K; k++) begin : g_mul
    if (USE_DSP) begin : g_dsp
      (* multstyle = "dsp" *)   logic signed [W_P-1:0] p;
      assign p = $signed(a1[k]) * $signed(w1[k]);
      always_ff @(posedge clk) p2[k] <= p;
    end else begin : g_alm
      (* multstyle = "logic" *) logic signed [W_P-1:0] p;
      assign p = $signed(a1[k]) * $signed(w1[k]);
      always_ff @(posedge clk) p2[k] <= p;
    end
  end

  // ---- adder tree --------------------------------------------------------------
  adder_tree #(.N(K), .WI(W_P)) u_tree (.clk(clk), .din(p2), .dout(out_sum));

  // ---- valid / tag alongside ---------------------------------------------------
  localparam int LAT = 2 + LVL;
  logic [LAT-1:0]            v_d;
  logic [LAT-1:0][W_TAG-1:0] t_d;

  always_ff @(posedge clk) begin
    v_d[0] <= rst_n & in_valid;
    t_d[0] <= in_tag;
    for (int i = 1; i < LAT; i++) begin
      v_d[i] <= rst_n & v_d[i-1];
      t_d[i] <= t_d[i-1];
    end
  end

  assign out_valid = v_d[LAT-1];
  assign out_tag   = t_d[LAT-1];
endmodule
