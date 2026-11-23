// -----------------------------------------------------------------------------
// RMSNorm + per-token int8 quantization (model/vec_golden.py:norm_q8, bit exact)
//
//   ms = mean(v^2) + eps         exact, Q.32           v: Q.16 (32 bits)
//      = m * 4^j, m in [1,4)     m as Q2.14 (rounded half-up)
//   R  = rsqrt(m)                96-segment piecewise-linear table, Q.16
//   u  = round(v * w)            Q.16, saturated       w: norm weight Q.12 (18 bits)
//   q  = sat8(round(u * R / 2^t)), t smallest with max|u|*R <= 127 * 2^t
//   k  = t - 16 - j              (activation value = q * 2^k)
//
// Flow for one vector:
//   1) pulse start with n_log2 (n = 2^n_log2 <= NMAX) and the weight base address
//   2) stream the n values on in_valid / in_v (bubbles allowed). u is stored
//      in an internal buffer while sum(v^2) and max|u| are accumulated.
//      The norm weights are read from an external memory: w_data must be
//      valid RL cycles after w_addr.
//   3) ~10 cycles of scalar work (mean, normalize, rsqrt, scale exponent)
//   4) the int8 vector comes out as K-lane words on act_we / act_g / act_d,
//      the same interface as lin_unit's activation buffer; k_out is valid with
//      the first act_we and stays until the next start. done pulses with the
//      last word.
// -----------------------------------------------------------------------------
module norm_q8 #(
  parameter int NMAX = 512,
  parameter int K    = 128,
  parameter int WAW  = 14,        // norm-weight memory address
  parameter int RL   = 1,         // norm-weight memory read latency
  parameter int W_K  = 8
)(
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         start,
  input  logic [3:0]                   n_log2,
  input  logic [WAW-1:0]               w_base,
  output logic                         busy,
  // input vector
  input  logic                         in_valid,
  input  logic signed [31:0]           in_v,
  // norm-weight memory read port
  output logic [WAW-1:0]               w_addr,
  input  logic signed [17:0]           w_data,
  // int8 output, K lanes per word
  output logic                         act_we,
  output logic [$clog2(NMAX/K)-1:0]    act_g,
  output logic [K-1:0][7:0]            act_d,
  output logic signed [W_K-1:0]        k_out,
  output logic                         done
);
  localparam int IW    = $clog2(NMAX);
  localparam int GW    = $clog2(NMAX / K);
  localparam int LW    = $clog2(K);
  localparam int W_SS  = 64 + IW;                 // sum of squares
  localparam int W_BL  = $clog2(W_SS + 1);
  localparam int W_P   = 32 + 17;                 // u * R
  localparam int W_Q   = 60;                      // shifted / rounded
  localparam logic [W_SS-1:0] EPS_Q = W_SS'(42950);   // round(1e-5 * 2^32)

  typedef enum logic [3:0] {S_IDLE, S_IN, S_C1, S_C2, S_C3, S_C4, S_C5, S_C6,
                            S_C7, S_C8, S_C9, S_OUT} state_t;
  state_t st;

  logic [3:0]      nl_q;
  logic [IW:0]     n_q;
  logic [WAW-1:0]  base_q;
  logic [IW:0]     cnt_in, cnt_done, rd_cnt;

  // ---------------------------------------------------------------------------
  // pass 1: u = round(v*w) into the buffer, sum(v^2), max|u|
  // ---------------------------------------------------------------------------
  logic accept;
  assign accept = (st == S_IN) && in_valid && (cnt_in < n_q);
  assign w_addr = base_q + WAW'(cnt_in);

  logic [RL-1:0]               pv_ok;
  logic [RL-1:0][IW-1:0]       pv_idx;
  logic [RL-1:0][31:0]         pv_v;

  always_ff @(posedge clk) begin
    pv_ok[0]  <= rst_n & accept;
    pv_idx[0] <= cnt_in[IW-1:0];
    pv_v[0]   <= in_v;
    for (int i = 1; i < RL; i++) begin
      pv_ok[i]  <= rst_n & pv_ok[i-1];
      pv_idx[i] <= pv_idx[i-1];
      pv_v[i]   <= pv_v[i-1];
    end
  end

  // products (registered twice so Quartus can pipeline the wide multipliers)
  logic              b_ok, c_ok, d_ok;
  logic [IW-1:0]     b_idx, c_idx, d_idx;
  logic signed [49:0] uw_b, uw_c;
  logic [63:0]       sq_b, sq_c, sq_d;
  logic signed [31:0] u_d;

  always_ff @(posedge clk) begin
    b_ok  <= rst_n & pv_ok[RL-1];
    b_idx <= pv_idx[RL-1];
    uw_b  <= $signed(pv_v[RL-1]) * w_data;
    sq_b  <= $signed(pv_v[RL-1]) * $signed(pv_v[RL-1]);   // 64-bit context (no cast: a cast
                                                          // would multiply at 32 bits)
    c_ok  <= rst_n & b_ok;
    c_idx <= b_idx;
    uw_c  <= uw_b;
    sq_c  <= sq_b;
  end

  logic signed [38:0] uw_r;
  logic signed [31:0] u_sat;
  rshift_round #(.WI(50), .SHIFT(12)) u_rw (.din(uw_c), .dout(uw_r));
  saturate #(.WI(39), .WO(32)) u_su (.din(uw_r), .dout(u_sat));

  always_ff @(posedge clk) begin
    d_ok  <= rst_n & c_ok;
    d_idx <= c_idx;
    u_d   <= u_sat;
    sq_d  <= sq_c;
  end

  // buffer of u (one word per element)
  logic [31:0] buf_rd;
  logic [IW-1:0] rd_addr;
  sdp_ram #(.W(32), .D(NMAX)) u_buf (
    .clk(clk), .we(d_ok), .waddr(d_idx), .wdata(u_d), .raddr(rd_addr), .rdata(buf_rd));

  logic [W_SS-1:0] ss;
  logic [31:0]     umax;
  logic [31:0]     u_abs;
  assign u_abs = u_d[31] ? 32'(-u_d) : 32'(u_d);

  // ---------------------------------------------------------------------------
  // scalar stages
  // ---------------------------------------------------------------------------
  function automatic logic [W_BL-1:0] bitlen(input logic [W_SS-1:0] x);
    bitlen = '0;
    for (int i = 0; i < W_SS; i++)
      if (x[i]) bitlen = W_BL'(i + 1);
  endfunction

  logic [W_SS-1:0]       ms;
  logic [W_BL-1:0]       j2;          // 2*j
  logic [5:0]            jj;
  logic [16:0]           m16;
  logic [6:0]            ridx;
  logic [8:0]            rfr;
  logic [16:0]           rc0, rc0_q, R;
  logic signed [11:0]    rc1, rc1_q;
  logic [48:0]           amax;
  logic [6:0]            ab;           // bitlen(amax)
  logic signed [7:0]     t_q;
  logic signed [W_K-1:0] k_c;

  always_comb begin
    case (ridx)
      7'd0: begin rc0 = 17'd65533; rc1 = -12'sd1001; end
      7'd1: begin rc0 = 17'd64533; rc1 = -12'sd956; end
      7'd2: begin rc0 = 17'd63577; rc1 = -12'sd915; end
      7'd3: begin rc0 = 17'd62662; rc1 = -12'sd876; end
      7'd4: begin rc0 = 17'd61786; rc1 = -12'sd841; end
      7'd5: begin rc0 = 17'd60945; rc1 = -12'sd807; end
      7'd6: begin rc0 = 17'd60138; rc1 = -12'sd776; end
      7'd7: begin rc0 = 17'd59362; rc1 = -12'sd747; end
      7'd8: begin rc0 = 17'd58616; rc1 = -12'sd719; end
      7'd9: begin rc0 = 17'd57896; rc1 = -12'sd693; end
      7'd10: begin rc0 = 17'd57203; rc1 = -12'sd669; end
      7'd11: begin rc0 = 17'd56534; rc1 = -12'sd646; end
      7'd12: begin rc0 = 17'd55888; rc1 = -12'sd624; end
      7'd13: begin rc0 = 17'd55264; rc1 = -12'sd604; end
      7'd14: begin rc0 = 17'd54660; rc1 = -12'sd585; end
      7'd15: begin rc0 = 17'd54075; rc1 = -12'sd566; end
      7'd16: begin rc0 = 17'd53509; rc1 = -12'sd549; end
      7'd17: begin rc0 = 17'd52960; rc1 = -12'sd532; end
      7'd18: begin rc0 = 17'd52428; rc1 = -12'sd517; end
      7'd19: begin rc0 = 17'd51911; rc1 = -12'sd502; end
      7'd20: begin rc0 = 17'd51410; rc1 = -12'sd487; end
      7'd21: begin rc0 = 17'd50923; rc1 = -12'sd474; end
      7'd22: begin rc0 = 17'd50449; rc1 = -12'sd461; end
      7'd23: begin rc0 = 17'd49988; rc1 = -12'sd448; end
      7'd24: begin rc0 = 17'd49540; rc1 = -12'sd436; end
      7'd25: begin rc0 = 17'd49103; rc1 = -12'sd425; end
      7'd26: begin rc0 = 17'd48678; rc1 = -12'sd414; end
      7'd27: begin rc0 = 17'd48264; rc1 = -12'sd404; end
      7'd28: begin rc0 = 17'd47860; rc1 = -12'sd394; end
      7'd29: begin rc0 = 17'd47466; rc1 = -12'sd384; end
      7'd30: begin rc0 = 17'd47082; rc1 = -12'sd375; end
      7'd31: begin rc0 = 17'd46707; rc1 = -12'sd366; end
      7'd32: begin rc0 = 17'd46340; rc1 = -12'sd358; end
      7'd33: begin rc0 = 17'd45983; rc1 = -12'sd350; end
      7'd34: begin rc0 = 17'd45633; rc1 = -12'sd342; end
      7'd35: begin rc0 = 17'd45291; rc1 = -12'sd334; end
      7'd36: begin rc0 = 17'd44957; rc1 = -12'sd327; end
      7'd37: begin rc0 = 17'd44630; rc1 = -12'sd320; end
      7'd38: begin rc0 = 17'd44310; rc1 = -12'sd313; end
      7'd39: begin rc0 = 17'd43997; rc1 = -12'sd307; end
      7'd40: begin rc0 = 17'd43690; rc1 = -12'sd300; end
      7'd41: begin rc0 = 17'd43390; rc1 = -12'sd294; end
      7'd42: begin rc0 = 17'd43096; rc1 = -12'sd288; end
      7'd43: begin rc0 = 17'd42808; rc1 = -12'sd283; end
      7'd44: begin rc0 = 17'd42525; rc1 = -12'sd277; end
      7'd45: begin rc0 = 17'd42248; rc1 = -12'sd272; end
      7'd46: begin rc0 = 17'd41976; rc1 = -12'sd267; end
      7'd47: begin rc0 = 17'd41710; rc1 = -12'sd262; end
      7'd48: begin rc0 = 17'd41448; rc1 = -12'sd257; end
      7'd49: begin rc0 = 17'd41192; rc1 = -12'sd252; end
      7'd50: begin rc0 = 17'd40940; rc1 = -12'sd247; end
      7'd51: begin rc0 = 17'd40692; rc1 = -12'sd243; end
      7'd52: begin rc0 = 17'd40449; rc1 = -12'sd239; end
      7'd53: begin rc0 = 17'd40211; rc1 = -12'sd234; end
      7'd54: begin rc0 = 17'd39976; rc1 = -12'sd230; end
      7'd55: begin rc0 = 17'd39746; rc1 = -12'sd226; end
      7'd56: begin rc0 = 17'd39519; rc1 = -12'sd223; end
      7'd57: begin rc0 = 17'd39297; rc1 = -12'sd219; end
      7'd58: begin rc0 = 17'd39078; rc1 = -12'sd215; end
      7'd59: begin rc0 = 17'd38863; rc1 = -12'sd212; end
      7'd60: begin rc0 = 17'd38651; rc1 = -12'sd208; end
      7'd61: begin rc0 = 17'd38442; rc1 = -12'sd205; end
      7'd62: begin rc0 = 17'd38237; rc1 = -12'sd202; end
      7'd63: begin rc0 = 17'd38036; rc1 = -12'sd199; end
      7'd64: begin rc0 = 17'd37837; rc1 = -12'sd196; end
      7'd65: begin rc0 = 17'd37641; rc1 = -12'sd193; end
      7'd66: begin rc0 = 17'd37449; rc1 = -12'sd190; end
      7'd67: begin rc0 = 17'd37259; rc1 = -12'sd187; end
      7'd68: begin rc0 = 17'd37073; rc1 = -12'sd184; end
      7'd69: begin rc0 = 17'd36889; rc1 = -12'sd181; end
      7'd70: begin rc0 = 17'd36707; rc1 = -12'sd179; end
      7'd71: begin rc0 = 17'd36529; rc1 = -12'sd176; end
      7'd72: begin rc0 = 17'd36353; rc1 = -12'sd174; end
      7'd73: begin rc0 = 17'd36179; rc1 = -12'sd171; end
      7'd74: begin rc0 = 17'd36008; rc1 = -12'sd169; end
      7'd75: begin rc0 = 17'd35839; rc1 = -12'sd166; end
      7'd76: begin rc0 = 17'd35673; rc1 = -12'sd164; end
      7'd77: begin rc0 = 17'd35509; rc1 = -12'sd162; end
      7'd78: begin rc0 = 17'd35347; rc1 = -12'sd160; end
      7'd79: begin rc0 = 17'd35188; rc1 = -12'sd157; end
      7'd80: begin rc0 = 17'd35030; rc1 = -12'sd155; end
      7'd81: begin rc0 = 17'd34875; rc1 = -12'sd153; end
      7'd82: begin rc0 = 17'd34722; rc1 = -12'sd151; end
      7'd83: begin rc0 = 17'd34570; rc1 = -12'sd149; end
      7'd84: begin rc0 = 17'd34421; rc1 = -12'sd147; end
      7'd85: begin rc0 = 17'd34274; rc1 = -12'sd146; end
      7'd86: begin rc0 = 17'd34128; rc1 = -12'sd144; end
      7'd87: begin rc0 = 17'd33984; rc1 = -12'sd142; end
      7'd88: begin rc0 = 17'd33843; rc1 = -12'sd140; end
      7'd89: begin rc0 = 17'd33702; rc1 = -12'sd138; end
      7'd90: begin rc0 = 17'd33564; rc1 = -12'sd137; end
      7'd91: begin rc0 = 17'd33427; rc1 = -12'sd135; end
      7'd92: begin rc0 = 17'd33292; rc1 = -12'sd133; end
      7'd93: begin rc0 = 17'd33159; rc1 = -12'sd132; end
      7'd94: begin rc0 = 17'd33027; rc1 = -12'sd130; end
      7'd95: begin rc0 = 17'd32897; rc1 = -12'sd129; end
      default: begin rc0 = 17'd0; rc1 = 12'sd0; end
    endcase
  end

  logic [W_SS+1:0]  m_tmp;
  logic signed [20:0] rprod;
  logic [56:0]      lhs, rhs;
  logic signed [10:0] kk;

  // ---------------------------------------------------------------------------
  // pass 2: q = sat8(round(u * R / 2^t)), packed into K-lane words
  // ---------------------------------------------------------------------------
  logic              o1_ok, o2_ok, o3_ok, o4_ok;
  logic [IW-1:0]     o1_idx, o2_idx, o3_idx, o4_idx;
  logic signed [W_P-1:0] o2_p, o3_p;
  logic signed [W_Q-1:0] o4_s;
  logic [K-1:0][7:0] pack, pack_n;
  logic signed [7:0] q8;

  saturate #(.WI(W_Q), .WO(8)) u_sq (.din(o4_s), .dout(q8));

  assign rd_addr = rd_cnt[IW-1:0];

  always_comb begin
    pack_n = pack;
    pack_n[o4_idx[LW-1:0]] = q8;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st     <= S_IDLE;
      busy   <= 1'b0;
      act_we <= 1'b0;
      done   <= 1'b0;
    end else begin
      act_we <= 1'b0;
      done   <= 1'b0;
      case (st)
        S_IDLE: if (start) begin
          st       <= S_IN;
          busy     <= 1'b1;
          nl_q     <= n_log2;
          n_q      <= (IW+1)'(1) << n_log2;
          base_q   <= w_base;
          cnt_in   <= '0;
          cnt_done <= '0;
          ss       <= '0;
          umax     <= '0;
        end
        S_IN: begin
          if (accept) cnt_in <= cnt_in + 1'b1;
          if (d_ok) begin
            ss       <= ss + W_SS'(sq_d);
            umax     <= (u_abs > umax) ? u_abs : umax;
            cnt_done <= cnt_done + 1'b1;
          end
          if (cnt_done == n_q) st <= S_C1;
        end
        S_C1: begin                               // ms = sum/n + eps
          ms <= (ss >> nl_q) + EPS_Q;
          st <= S_C2;
        end
        S_C2: begin                               // 2j = 2*floor((bitlen-1)/2)
          j2 <= (bitlen(ms) - 1'b1) & ~W_BL'(1);
          st <= S_C3;
        end
        S_C3: begin                               // m = ms * 2^(14-2j), Q2.14
          if (j2 <= 14) m_tmp = (W_SS+2)'(ms) << (14 - j2);
          else          m_tmp = ((W_SS+2)'(ms) + ((W_SS+2)'(1) << (j2 - 15))) >> (j2 - 14);
          m16 <= (m_tmp >= 65536) ? 17'd65535 : 17'(m_tmp);
          jj  <= 6'(j2 >> 1);
          st  <= S_C4;
        end
        S_C4: begin                               // table index and fraction
          ridx <= 7'((m16 >> 9) - 17'd32);
          rfr  <= m16[8:0];
          st   <= S_C5;
        end
        S_C5: begin
          rc0_q <= rc0;
          rc1_q <= rc1;
          st    <= S_C6;
        end
        S_C6: begin                               // R = c0 + round(c1 * fr / 2^9)
          rprod = rc1_q * $signed({1'b0, rfr});
          R  <= 17'($signed({1'b0, rc0_q}) + ((rprod + 21'sd256) >>> 9));
          st <= S_C7;
        end
        S_C7: begin
          amax <= 49'(umax) * 49'(R);
          st   <= S_C8;
        end
        S_C8: begin                               // smallest t: amax <= 127 * 2^t
          ab = '0;
          for (int i = 0; i < 49; i++) if (amax[i]) ab = 7'(i + 1);
          lhs = 57'(amax) << 7;
          rhs = (57'(1) << (ab + 7)) - (57'(1) << ab);
          if (amax == '0)     t_q <= -8'sd64;
          else if (lhs <= rhs) t_q <= 8'(ab) - 8'sd7;
          else                 t_q <= 8'(ab) - 8'sd6;
          st <= S_C9;
        end
        S_C9: begin                               // k = t - 16 - j  (int8)
          kk = 11'(t_q) - 11'sd16 - 11'($signed({1'b0, jj}));
          k_out  <= (kk > 127) ? W_K'(127) : (kk < -128) ? W_K'(-128) : W_K'(kk);
          rd_cnt <= '0;
          st     <= S_OUT;
        end
        S_OUT: begin
          if (rd_cnt < n_q) rd_cnt <= rd_cnt + 1'b1;
          if (o4_ok) begin
            pack <= pack_n;
            if (o4_idx[LW-1:0] == LW'(K - 1)) begin
              act_we <= 1'b1;
              act_g  <= GW'(o4_idx >> LW);
              act_d  <= pack_n;
              if ((IW+1)'(o4_idx) == n_q - 1'b1) begin
                done <= 1'b1;
                busy <= 1'b0;
                st   <= S_IDLE;
              end
            end
          end
        end
        default: st <= S_IDLE;
      endcase
    end
  end

  // pass-2 pipeline: read (1) -> u*R (2) -> extra reg (3) -> shift / round (4)
  always_ff @(posedge clk) begin
    o1_ok  <= rst_n && (st == S_OUT) && (rd_cnt < n_q);
    o1_idx <= rd_cnt[IW-1:0];
    o2_ok  <= rst_n & o1_ok;
    o2_idx <= o1_idx;
    o2_p   <= $signed(buf_rd) * $signed({1'b0, R});
    o3_ok  <= rst_n & o2_ok;
    o3_idx <= o2_idx;
    o3_p   <= o2_p;
    o4_ok  <= rst_n & o3_ok;
    o4_idx <= o3_idx;
    if (t_q <= 0)
      o4_s <= W_Q'(o3_p) <<< ((-t_q > 9) ? 9 : -t_q);
    else
      o4_s <= (W_Q'(o3_p) + (W_Q'(1) <<< (t_q - 1))) >>> t_q;
  end
endmodule
