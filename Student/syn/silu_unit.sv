// -----------------------------------------------------------------------------
// SiLU(x) = x * sigmoid(x) on Q.12 (model/vec_golden.py:silu_fx, bit exact)
//
//   a = |x|; sigmoid(a) = c0[i] + round(c1[i] * f / 2^9)   (128 segments of
//   width 1/8 on [0, 16), Q.16); a >= 16 -> 1; x < 0 -> 1 - sigmoid(|x|)
//   y = sat24(round(x * sigmoid / 2^16))
//
// One input per cycle, latency 6. in_tag travels with the data.
// -----------------------------------------------------------------------------
module silu_unit #(
  parameter int W     = 24,      // Q.12
  parameter int W_TAG = 16
)(
  input  logic                clk,
  input  logic                rst_n,
  input  logic                in_valid,
  input  logic signed [W-1:0] in_x,
  input  logic [W_TAG-1:0]    in_tag,
  output logic                out_valid,
  output logic signed [W-1:0] out_y,
  output logic [W_TAG-1:0]    out_tag
);
  localparam int FR = 9;                       // fraction bits inside a segment
  localparam logic [W-1:0] BIG = W'(16) << 12; // 16.0 in Q.12

  // ---- stage 1: |x|, segment, fraction ----------------------------------------
  logic                v1, n1, big1;
  logic signed [W-1:0] x1;
  logic [6:0]          i1;
  logic [FR-1:0]       f1;
  logic [W-1:0]        a;
  logic [W-1:0]        seg;

  assign a   = in_x[W-1] ? W'(-in_x) : W'(in_x);
  assign seg = a >> FR;

  always_ff @(posedge clk) begin
    v1   <= rst_n & in_valid;
    x1   <= in_x;
    n1   <= in_x[W-1];
    big1 <= (a >= BIG);
    i1   <= (seg > 127) ? 7'd127 : seg[6:0];
    f1   <= a[FR-1:0];
  end

  // ---- stage 2: table -------------------------------------------------------------
  logic [16:0] sc0;
  logic [11:0] sc1;
  always_comb begin
    case (i1)
      7'd0: begin sc0 = 17'd32769; sc1 = 12'd2045; end
      7'd1: begin sc0 = 17'd34815; sc1 = 12'd2029; end
      7'd2: begin sc0 = 17'd36845; sc1 = 12'd1998; end
      7'd3: begin sc0 = 17'd38844; sc1 = 12'd1952; end
      7'd4: begin sc0 = 17'd40798; sc1 = 12'd1894; end
      7'd5: begin sc0 = 17'd42692; sc1 = 12'd1823; end
      7'd6: begin sc0 = 17'd44516; sc1 = 12'd1744; end
      7'd7: begin sc0 = 17'd46260; sc1 = 12'd1656; end
      7'd8: begin sc0 = 17'd47917; sc1 = 12'd1563; end
      7'd9: begin sc0 = 17'd49480; sc1 = 12'd1467; end
      7'd10: begin sc0 = 17'd50947; sc1 = 12'd1369; end
      7'd11: begin sc0 = 17'd52316; sc1 = 12'd1271; end
      7'd12: begin sc0 = 17'd53587; sc1 = 12'd1174; end
      7'd13: begin sc0 = 17'd54760; sc1 = 12'd1079; end
      7'd14: begin sc0 = 17'd55839; sc1 = 12'd988; end
      7'd15: begin sc0 = 17'd56827; sc1 = 12'd902; end
      7'd16: begin sc0 = 17'd57729; sc1 = 12'd820; end
      7'd17: begin sc0 = 17'd58549; sc1 = 12'd743; end
      7'd18: begin sc0 = 17'd59291; sc1 = 12'd672; end
      7'd19: begin sc0 = 17'd59963; sc1 = 12'd606; end
      7'd20: begin sc0 = 17'd60568; sc1 = 12'd545; end
      7'd21: begin sc0 = 17'd61113; sc1 = 12'd489; end
      7'd22: begin sc0 = 17'd61601; sc1 = 12'd438; end
      7'd23: begin sc0 = 17'd62039; sc1 = 12'd392; end
      7'd24: begin sc0 = 17'd62430; sc1 = 12'd350; end
      7'd25: begin sc0 = 17'd62780; sc1 = 12'd312; end
      7'd26: begin sc0 = 17'd63092; sc1 = 12'd278; end
      7'd27: begin sc0 = 17'd63369; sc1 = 12'd247; end
      7'd28: begin sc0 = 17'd63617; sc1 = 12'd220; end
      7'd29: begin sc0 = 17'd63836; sc1 = 12'd195; end
      7'd30: begin sc0 = 17'd64031; sc1 = 12'd173; end
      7'd31: begin sc0 = 17'd64205; sc1 = 12'd154; end
      7'd32: begin sc0 = 17'd64358; sc1 = 12'd136; end
      7'd33: begin sc0 = 17'd64494; sc1 = 12'd121; end
      7'd34: begin sc0 = 17'd64615; sc1 = 12'd107; end
      7'd35: begin sc0 = 17'd64722; sc1 = 12'd95; end
      7'd36: begin sc0 = 17'd64817; sc1 = 12'd84; end
      7'd37: begin sc0 = 17'd64900; sc1 = 12'd74; end
      7'd38: begin sc0 = 17'd64974; sc1 = 12'd66; end
      7'd39: begin sc0 = 17'd65040; sc1 = 12'd58; end
      7'd40: begin sc0 = 17'd65098; sc1 = 12'd51; end
      7'd41: begin sc0 = 17'd65149; sc1 = 12'd45; end
      7'd42: begin sc0 = 17'd65194; sc1 = 12'd40; end
      7'd43: begin sc0 = 17'd65234; sc1 = 12'd35; end
      7'd44: begin sc0 = 17'd65270; sc1 = 12'd31; end
      7'd45: begin sc0 = 17'd65301; sc1 = 12'd28; end
      7'd46: begin sc0 = 17'd65328; sc1 = 12'd24; end
      7'd47: begin sc0 = 17'd65353; sc1 = 12'd22; end
      7'd48: begin sc0 = 17'd65374; sc1 = 12'd19; end
      7'd49: begin sc0 = 17'd65393; sc1 = 12'd17; end
      7'd50: begin sc0 = 17'd65410; sc1 = 12'd15; end
      7'd51: begin sc0 = 17'd65425; sc1 = 12'd13; end
      7'd52: begin sc0 = 17'd65438; sc1 = 12'd12; end
      7'd53: begin sc0 = 17'd65449; sc1 = 12'd10; end
      7'd54: begin sc0 = 17'd65459; sc1 = 12'd9; end
      7'd55: begin sc0 = 17'd65468; sc1 = 12'd8; end
      7'd56: begin sc0 = 17'd65476; sc1 = 12'd7; end
      7'd57: begin sc0 = 17'd65483; sc1 = 12'd6; end
      7'd58: begin sc0 = 17'd65490; sc1 = 12'd5; end
      7'd59: begin sc0 = 17'd65495; sc1 = 12'd5; end
      7'd60: begin sc0 = 17'd65500; sc1 = 12'd4; end
      7'd61: begin sc0 = 17'd65504; sc1 = 12'd4; end
      7'd62: begin sc0 = 17'd65508; sc1 = 12'd3; end
      7'd63: begin sc0 = 17'd65511; sc1 = 12'd3; end
      7'd64: begin sc0 = 17'd65514; sc1 = 12'd3; end
      7'd65: begin sc0 = 17'd65517; sc1 = 12'd2; end
      7'd66: begin sc0 = 17'd65519; sc1 = 12'd2; end
      7'd67: begin sc0 = 17'd65521; sc1 = 12'd2; end
      7'd68: begin sc0 = 17'd65523; sc1 = 12'd2; end
      7'd69: begin sc0 = 17'd65524; sc1 = 12'd1; end
      7'd70: begin sc0 = 17'd65526; sc1 = 12'd1; end
      7'd71: begin sc0 = 17'd65527; sc1 = 12'd1; end
      7'd72: begin sc0 = 17'd65528; sc1 = 12'd1; end
      7'd73: begin sc0 = 17'd65529; sc1 = 12'd1; end
      7'd74: begin sc0 = 17'd65530; sc1 = 12'd1; end
      7'd75: begin sc0 = 17'd65530; sc1 = 12'd1; end
      7'd76: begin sc0 = 17'd65531; sc1 = 12'd1; end
      7'd77: begin sc0 = 17'd65532; sc1 = 12'd1; end
      7'd78: begin sc0 = 17'd65532; sc1 = 12'd0; end
      7'd79: begin sc0 = 17'd65533; sc1 = 12'd0; end
      7'd80: begin sc0 = 17'd65533; sc1 = 12'd0; end
      7'd81: begin sc0 = 17'd65533; sc1 = 12'd0; end
      7'd82: begin sc0 = 17'd65534; sc1 = 12'd0; end
      7'd83: begin sc0 = 17'd65534; sc1 = 12'd0; end
      7'd84: begin sc0 = 17'd65534; sc1 = 12'd0; end
      7'd85: begin sc0 = 17'd65534; sc1 = 12'd0; end
      7'd86: begin sc0 = 17'd65535; sc1 = 12'd0; end
      7'd87: begin sc0 = 17'd65535; sc1 = 12'd0; end
      7'd88: begin sc0 = 17'd65535; sc1 = 12'd0; end
      7'd89: begin sc0 = 17'd65535; sc1 = 12'd0; end
      7'd90: begin sc0 = 17'd65535; sc1 = 12'd0; end
      7'd91: begin sc0 = 17'd65535; sc1 = 12'd0; end
      7'd92: begin sc0 = 17'd65535; sc1 = 12'd0; end
      7'd93: begin sc0 = 17'd65535; sc1 = 12'd0; end
      7'd94: begin sc0 = 17'd65535; sc1 = 12'd0; end
      7'd95: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd96: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd97: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd98: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd99: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd100: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd101: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd102: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd103: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd104: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd105: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd106: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd107: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd108: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd109: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd110: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd111: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd112: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd113: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd114: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd115: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd116: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd117: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd118: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd119: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd120: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd121: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd122: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd123: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd124: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd125: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd126: begin sc0 = 17'd65536; sc1 = 12'd0; end
      7'd127: begin sc0 = 17'd65536; sc1 = 12'd0; end
      default: begin sc0 = 17'd65536; sc1 = 12'd0; end
    endcase
  end

  logic                v2, n2, big2;
  logic signed [W-1:0] x2;
  logic [16:0]         c0_2;
  logic [11:0]         c1_2;
  logic [FR-1:0]       f2;

  always_ff @(posedge clk) begin
    v2   <= rst_n & v1;
    x2   <= x1;
    n2   <= n1;
    big2 <= big1;
    c0_2 <= sc0;
    c1_2 <= sc1;
    f2   <= f1;
  end

  // ---- stage 3: c1 * f ------------------------------------------------------------
  logic                v3, n3, big3;
  logic signed [W-1:0] x3;
  logic [16:0]         c0_3;
  logic [20:0]         p3;

  always_ff @(posedge clk) begin
    v3   <= rst_n & v2;
    x3   <= x2;
    n3   <= n2;
    big3 <= big2;
    c0_3 <= c0_2;
    p3   <= c1_2 * f2;
  end

  // ---- stage 4: sigmoid, Q.16 -------------------------------------------------------
  logic [16:0] s_pos, s_abs;
  assign s_pos = c0_3 + 17'((p3 + 21'd256) >> FR);
  assign s_abs = big3 ? 17'd65536 : s_pos;

  logic                v4;
  logic signed [W-1:0] x4;
  logic [16:0]         s4;
  logic [W_TAG-1:0]    t1, t2, t3, t4, t5;

  always_ff @(posedge clk) begin
    v4 <= rst_n & v3;
    x4 <= x3;
    s4 <= n3 ? 17'd65536 - s_abs : s_abs;
  end

  // ---- stage 5: x * sigmoid -----------------------------------------------------------
  logic                    v5;
  logic signed [W+17:0]    p5;

  always_ff @(posedge clk) begin
    v5 <= rst_n & v4;
    p5 <= x4 * $signed({1'b0, s4});
  end

  // ---- stage 6: round, saturate ---------------------------------------------------------
  logic signed [W+2:0] r6;
  logic signed [W-1:0] y6;
  rshift_round #(.WI(W + 18), .SHIFT(16)) u_r (.din(p5), .dout(r6));
  saturate #(.WI(W + 3), .WO(W)) u_s (.din(r6), .dout(y6));

  always_ff @(posedge clk) begin
    out_valid <= rst_n & v5;
    out_y     <= y6;
    t1 <= in_tag;  t2 <= t1;  t3 <= t2;  t4 <= t3;  t5 <= t4;  out_tag <= t5;
  end
endmodule
