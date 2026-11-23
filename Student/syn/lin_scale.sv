// -----------------------------------------------------------------------------
// Row rescale for the int8 x int4 linear layers
//
//   y = 2^k * sum_g acc_g * s_g        (rounded once to fixed point Q.F_OUT)
//
// acc_g : exact group dot product from lin_dot (signed W_ACC bits)
// s_g   : fp16 group scale. s = m * 2^(sh - 24) with
//           normal    m = 1024 + frac, sh = E - 1
//           subnormal m = frac,        sh = 0
//         so every term m * acc << sh is an integer in units of 2^-24 and the
//         row sum is exact (W_S bits).
// k     : activation scale exponent (activations were quantized with 2^k)
//
// The exact sum (units of 2^-24) is shifted by k - 24 + F_OUT: a right shift
// rounds half-up, a left shift is exact; the result saturates to W_OUT bits.
// Bit-exact with model/vec_golden.py:lin_fixed.
// (in_proj: F_OUT = 12, W_OUT = 24;  out_proj: F_OUT = 16, W_OUT = 32)
//
// Input: one beat per group, in_first on the first group of a row, in_last on
// the last (both on the same beat for a one-group row). in_k and in_tag are
// taken from the last beat. Bubbles are allowed between beats.
// Latency from the last beat to out_valid: 9 cycles.
// -----------------------------------------------------------------------------
module lin_scale #(
  parameter int W_ACC = 19,
  parameter int W_K   = 8,
  parameter int W_TAG = 16,
  parameter int F_OUT = 12,
  parameter int W_OUT = 24
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
  output logic signed [W_OUT-1:0] out_y
);
  localparam int W_M  = 12;                 // signed mantissa (11 bits + sign)
  localparam int W_PR = W_ACC + W_M;        // acc * m
  localparam int W_T  = W_PR + 29;          // << sh (sh <= 29)
  localparam int W_S  = W_T + 3;            // sum of up to 8 groups
  localparam int W_SH = 10;                 // signed shift amount
  localparam int MAXR = W_S;                // right shifts beyond this give 0 / -1
  localparam int MAXL = W_OUT + 1;          // left shifts beyond this always saturate
  localparam int W_X  = W_S + MAXL + 1;     // width after a left shift

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

  // ---- stage 2a: acc * m as two partial products --------------------------------
  // acc = hi * 2^10 + lo (lo unsigned). Every multiplier input is <= 18 bits:
  // a single 19 x 12 product was mapped by Quartus onto the 18-bit AY port of
  // the 18x19 DSP mode and rejected (Error 15649).
  localparam int W_LO = 10;
  localparam int W_HI = W_ACC - W_LO;

  logic                          v2a, f2a, l2a;
  logic signed [W_LO+W_M:0]      plo2a;      // (W_LO+1) x W_M
  logic signed [W_HI+W_M-1:0]    phi2a;      //  W_HI    x W_M
  logic [4:0]                    sh2a;
  logic signed [W_K-1:0]         k2a;
  logic [W_TAG-1:0]              tag2a;

  always_ff @(posedge clk) begin
    v2a   <= rst_n & v1;
    f2a   <= f1;
    l2a   <= l1;
    plo2a <= $signed({1'b0, acc1[W_LO-1:0]}) * m1;
    phi2a <= $signed(acc1[W_ACC-1:W_LO]) * m1;
    sh2a  <= sh1;
    k2a   <= k1;
    tag2a <= tag1;
  end

  // ---- stage 2b: recombine -----------------------------------------------------------
  logic                    v2, f2, l2;
  logic signed [W_PR-1:0]  pr2;
  logic [4:0]              sh2;
  logic signed [W_K-1:0]   k2;
  logic [W_TAG-1:0]        tag2;

  always_ff @(posedge clk) begin
    v2   <= rst_n & v2a;
    f2   <= f2a;
    l2   <= l2a;
    pr2  <= (W_PR'(phi2a) <<< W_LO) + W_PR'(plo2a);
    sh2  <= sh2a;
    k2   <= k2a;
    tag2 <= tag2a;
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

  // ---- stage 5: shift amount -------------------------------------------------------
  logic                    v5, left5;
  logic signed [W_S-1:0]   fin5;
  logic [6:0]              rs5, ls5;
  logic [W_TAG-1:0]        tag5;
  logic signed [W_SH-1:0]  sh;

  assign sh = W_SH'(k4) - W_SH'(24) + W_SH'(F_OUT);

  always_ff @(posedge clk) begin
    v5    <= rst_n & v4;
    fin5  <= fin4;
    left5 <= (sh >= 0);
    rs5   <= (sh >= 0) ? 7'd0 : ((-sh > MAXR) ? 7'(MAXR) : 7'(-sh));
    ls5   <= (sh <  0) ? 7'd0 : (( sh > MAXL) ? 7'(MAXL) : 7'(sh));
    tag5  <= tag4;
  end

  // ---- stage 6: add the rounding constant 2^(r-1) ----------------------------------
  logic                    v6, left6;
  logic signed [W_S:0]     a6;
  logic [6:0]              rs6, ls6;
  logic [W_TAG-1:0]        tag6;

  always_ff @(posedge clk) begin
    v6    <= rst_n & v5;
    left6 <= left5;
    rs6   <= rs5;
    ls6   <= ls5;
    a6    <= (W_S+1)'(fin5) + ((rs5 == 0) ? '0 : ((W_S+1)'(1) <<< (rs5 - 7'd1)));
    tag6  <= tag5;
  end

  // ---- stage 7: shift ----------------------------------------------------------------
  logic                    v7;
  logic signed [W_X-1:0]   x7;
  logic [W_TAG-1:0]        tag7;

  always_ff @(posedge clk) begin
    v7   <= rst_n & v6;
    x7   <= left6 ? (W_X'(a6) <<< ls6) : W_X'(a6 >>> rs6);
    tag7 <= tag6;
  end

  // ---- stage 8: saturate ---------------------------------------------------------------
  logic signed [W_OUT-1:0] y_c;
  saturate #(.WI(W_X), .WO(W_OUT)) u_sat (.din(x7), .dout(y_c));

  always_ff @(posedge clk) begin
    out_valid <= rst_n & v7;
    out_tag   <= tag7;
    out_y     <= y_c;
  end
endmodule
