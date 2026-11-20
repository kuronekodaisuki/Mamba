// -----------------------------------------------------------------------------
// Row rescale for the int8 x int4 linear layers
//
//   y = 2^k * sum_g acc_g * s_g        (rounded once to float32)
//
// acc_g : exact group dot product from lin_dot (signed W_ACC bits)
// s_g   : fp16 group scale. s = m * 2^(sh - 24) with
//           normal    m = 1024 + frac, sh = E - 1
//           subnormal m = frac,        sh = 0
//         so every term m * acc << sh is an integer in units of 2^-24 and the
//         row sum is exact (W_S bits).
// k     : activation scale exponent (activations were quantized with 2^k)
//
// The exact sum is converted to float32 with round-to-nearest-even; a result
// below the normal range flushes to +0, above it becomes +-inf.
// Bit-exact with model/lin_golden.py:lin_rescale.
//
// Input: one beat per group, in_first on the first group of a row, in_last on
// the last (both on the same beat for a one-group row). in_k and in_tag are
// taken from the last beat. Bubbles are allowed between beats.
// Latency from the last beat to out_valid: 8 cycles.
// -----------------------------------------------------------------------------
module lin_scale #(
  parameter int W_ACC = 19,
  parameter int W_K   = 8,
  parameter int W_TAG = 16
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    in_valid,
  input  logic                    in_first,
  input  logic                    in_last,
  input  logic signed [W_ACC-1:0] in_acc,
  input  logic [15:0]             in_scale,
  input  logic signed [W_K-1:0]   in_k,
  input  logic [W_TAG-1:0]        in_tag,
  output logic                    out_valid,
  output logic [W_TAG-1:0]        out_tag,
  output logic [31:0]             out_y
);
  localparam int W_M  = 12;                 // signed mantissa (11 bits + sign)
  localparam int W_PR = W_ACC + W_M;        // acc * m
  localparam int W_T  = W_PR + 29;          // << sh (sh <= 29)
  localparam int W_S  = W_T + 3;            // sum of up to 8 groups
  localparam int W_LZ = $clog2(W_S + 1);
  localparam int W_E  = 12;                 // signed exponent arithmetic

  // ---- stage 1: decode the fp16 scale -----------------------------------------
  logic                    v1, f1, l1;
  logic signed [W_ACC-1:0] acc1;
  logic signed [W_M-1:0]   m1;
  logic [4:0]              sh1;
  logic signed [W_K-1:0]   k1;
  logic [W_TAG-1:0]        tag1;

  logic [4:0]  ex;
  logic [10:0] mag_m;
  assign ex    = in_scale[14:10];
  assign mag_m = (ex == 5'd0) ? {1'b0, in_scale[9:0]} : {1'b1, in_scale[9:0]};

  always_ff @(posedge clk) begin
    v1   <= rst_n & in_valid;
    f1   <= in_first;
    l1   <= in_last;
    acc1 <= in_acc;
    m1   <= in_scale[15] ? -$signed({1'b0, mag_m}) : $signed({1'b0, mag_m});
    sh1  <= (ex == 5'd0) ? 5'd0 : ex - 5'd1;
    k1   <= in_k;
    tag1 <= in_tag;
  end

  // ---- stage 2: acc * m (DSP, registered operands and product) ------------------
  logic                    v2, f2, l2;
  logic signed [W_PR-1:0]  pr2;
  logic [4:0]              sh2;
  logic signed [W_K-1:0]   k2;
  logic [W_TAG-1:0]        tag2;

  always_ff @(posedge clk) begin
    v2   <= rst_n & v1;
    f2   <= f1;
    l2   <= l1;
    pr2  <= acc1 * m1;
    sh2  <= sh1;
    k2   <= k1;
    tag2 <= tag1;
  end

  // ---- stage 3: align to units of 2^-24 -------------------------------------------
  logic                    v3, f3, l3;
  logic signed [W_T-1:0]   t3;
  logic signed [W_K-1:0]   k3;
  logic [W_TAG-1:0]        tag3;

  always_ff @(posedge clk) begin
    v3   <= rst_n & v2;
    f3   <= f2;
    l3   <= l2;
    t3   <= W_T'(pr2) <<< sh2;
    k3   <= k2;
    tag3 <= tag2;
  end

  // ---- stage 4: exact row accumulation ---------------------------------------------
  logic signed [W_S-1:0] sum_q, nxt;
  logic                  v4;
  logic signed [W_S-1:0] fin4;
  logic signed [W_K-1:0] k4;
  logic [W_TAG-1:0]      tag4;

  assign nxt = (f3 ? W_S'(0) : sum_q) + W_S'(t3);

  always_ff @(posedge clk) begin
    if (v3) sum_q <= nxt;
    v4   <= rst_n & v3 & l3;
    fin4 <= nxt;
    k4   <= k3;
    tag4 <= tag3;
  end

  // ---- stage 5: sign / magnitude --------------------------------------------------
  logic                  v5, neg5;
  logic [W_S-1:0]        mag5;
  logic signed [W_K-1:0] k5;
  logic [W_TAG-1:0]      tag5;

  always_ff @(posedge clk) begin
    v5   <= rst_n & v4;
    neg5 <= fin4[W_S-1];
    mag5 <= fin4[W_S-1] ? W_S'(-fin4) : W_S'(fin4);
    k5   <= k4;
    tag5 <= tag4;
  end

  // ---- stage 6: leading-zero count ---------------------------------------------------
  function automatic logic [W_LZ-1:0] lzc(input logic [W_S-1:0] v);
    lzc = W_LZ'(W_S);
    for (int i = 0; i < W_S; i++)
      if (v[i]) lzc = W_LZ'(W_S - 1 - i);     // the highest set bit wins
  endfunction

  logic                  v6, neg6, zero6;
  logic [W_S-1:0]        mag6;
  logic [W_LZ-1:0]       lz6;
  logic signed [W_K-1:0] k6;
  logic [W_TAG-1:0]      tag6;

  always_ff @(posedge clk) begin
    v6    <= rst_n & v5;
    neg6  <= neg5;
    zero6 <= (mag5 == '0);
    mag6  <= mag5;
    lz6   <= lzc(mag5);
    k6    <= k5;
    tag6  <= tag5;
  end

  // ---- stage 7: normalize, biased exponent -------------------------------------------
  logic                  v7, neg7, zero7;
  logic [W_S-1:0]        norm7;
  logic signed [W_E-1:0] e7;
  logic [W_TAG-1:0]      tag7;

  always_ff @(posedge clk) begin
    v7    <= rst_n & v6;
    neg7  <= neg6;
    zero7 <= zero6;
    norm7 <= mag6 << lz6;
    e7    <= W_E'(W_S - 1 - 24 + 127) - W_E'(lz6) + W_E'(k6);
    tag7  <= tag6;
  end

  // ---- stage 8: round to nearest even, pack ----------------------------------------
  logic [23:0]           mant;
  logic                  guard, sticky, inc;
  logic [24:0]           mr;
  logic signed [W_E-1:0] e_r;
  logic [31:0]           y_c;

  assign mant   = norm7[W_S-1 -: 24];
  assign guard  = norm7[W_S-25];
  assign sticky = |norm7[W_S-26:0];
  assign inc    = guard & (sticky | mant[0]);
  assign mr     = {1'b0, mant} + 25'(inc);
  assign e_r    = mr[24] ? e7 + W_E'(1) : e7;

  always_comb begin
    if (zero7 || e7 <= 0)        y_c = 32'h0000_0000;
    else if (e_r >= W_E'(255))   y_c = {neg7, 8'hFF, 23'd0};
    else if (mr[24])             y_c = {neg7, e_r[7:0], mr[23:1]};
    else                         y_c = {neg7, e_r[7:0], mr[22:0]};
  end

  always_ff @(posedge clk) begin
    out_valid <= rst_n & v7;
    out_tag   <= tag7;
    out_y     <= y_c;
  end
endmodule
