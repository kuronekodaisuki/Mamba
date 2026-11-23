// -----------------------------------------------------------------------------
// Mamba2 student core: all layers of one token, sequentially, fixed point
// (bit exact with student/hwref.py vec="fixed", i.e. model/vec_golden.py)
//
// Per token:  host writes h (D values, Q.16) with cfg_sel = 5, then pulses
// tok_go (tok_first = 1 for the first token of a sequence). For every layer:
//   1  norm_q8(h, pre-norm)            -> lin_unit activation buffer
//   2  in_proj (lin_unit)              rows: z | xBC | dt
//        z   -> z buffer,  xBC -> conv_unit (+SiLU) -> x buffer, B, C
//        dt  -> dt registers
//   3  ssm_discretize (NH heads)       -> delta, dA per head
//   4  ssm_scan_unit (NH*P rows)       y -> gate_unit (with z) -> norm_q8 (gated norm)
//   5  out_proj (lin_unit)             -> h += out (saturating)
// After every residual update the new h element is shown on dbg_* (the last
// layer's values are the token's output). tok_done pulses at the end.
//
// Host write port (cfg_*), word addresses as in student/export.py OUT/mem:
//   sel 0 linw  (512 bits)   1 lins (fp16)   2 normw (18 bits)
//       3 convp (72 bits)    4 ssmp ({D, A, dt_bias} 48 bits)   5 h (32 bits)
// -----------------------------------------------------------------------------
module mamba_core #(
  parameter int D   = 256,          // d_model
  parameter int DI  = 512,          // d_inner
  parameter int N   = 64,           // d_state
  parameter int P   = 64,           // headdim
  parameter int NL  = 12,           // layers
  parameter int K   = 128           // lanes per int8 group
)(
  input  logic          clk,
  input  logic          rst_n,
  // host write port
  input  logic          cfg_we,
  input  logic [2:0]    cfg_sel,
  input  logic [15:0]   cfg_addr,
  input  logic [511:0]  cfg_data,
  // token control
  input  logic          tok_go,
  input  logic          tok_first,
  output logic          busy,
  output logic          tok_done,
  // residual updates (per layer)
  output logic          dbg_valid,
  output logic [7:0]    dbg_layer,
  output logic [11:0]   dbg_idx,
  output logic [31:0]   dbg_h
);
  // ---- derived sizes ---------------------------------------------------------------
  localparam int NH        = DI / P;
  localparam int CONV_DIM  = DI + 2 * N;
  localparam int IN_ROWS   = 2 * DI + 2 * N + NH;
  localparam int IN_G      = D / K;
  localparam int OUT_G     = DI / K;
  localparam int IN_WORDS  = IN_ROWS * IN_G;
  localparam int STRIDE    = IN_WORDS + D * OUT_G;
  localparam int LIN_WORDS = NL * STRIDE;
  localparam int NORM_WORDS = NL * (D + DI) + D;
  localparam int LWA  = $clog2(LIN_WORDS);
  localparam int NWA  = $clog2(NORM_WORDS);
  localparam int LAW  = (NL > 1) ? $clog2(NL) : 1;
  localparam int HW   = $clog2(NH * NL);
  localparam int PW   = $clog2(P);
  localparam int RW   = 12;

  // ---- memories ------------------------------------------------------------------------
  logic [LWA-1:0]      lw_addr;
  logic [K-1:0][3:0]   lw_data;
  logic [15:0]         ls_data;
  sdp_ram #(.W(K*4), .D(LIN_WORDS), .IN_REG(1'b1), .OUT_REG(1'b1)) u_linw (
    .clk(clk), .we(cfg_we && cfg_sel == 3'd0), .waddr(LWA'(cfg_addr)), .wdata(cfg_data[K*4-1:0]),
    .raddr(lw_addr), .rdata(lw_data));
  sdp_ram #(.W(16), .D(LIN_WORDS), .IN_REG(1'b1), .OUT_REG(1'b1)) u_lins (
    .clk(clk), .we(cfg_we && cfg_sel == 3'd1), .waddr(LWA'(cfg_addr)), .wdata(cfg_data[15:0]),
    .raddr(lw_addr), .rdata(ls_data));

  logic [NWA-1:0]      nw_addr;
  logic signed [17:0]  nw_data;
  sdp_ram #(.W(18), .D(NORM_WORDS)) u_normw (
    .clk(clk), .we(cfg_we && cfg_sel == 3'd2), .waddr(NWA'(cfg_addr)), .wdata(cfg_data[17:0]),
    .raddr(nw_addr), .rdata(nw_data));

  localparam int CAW = $clog2(NL * CONV_DIM);
  logic [CAW-1:0]      cp_addr;
  logic [71:0]         cp_data;
  sdp_ram #(.W(72), .D(NL * CONV_DIM)) u_convp (
    .clk(clk), .we(cfg_we && cfg_sel == 3'd3), .waddr(CAW'(cfg_addr)), .wdata(cfg_data[71:0]),
    .raddr(cp_addr), .rdata(cp_data));

  localparam int SAW = $clog2(NL * NH);
  logic [SAW-1:0]      sp_addr;
  logic [47:0]         sp_data;
  sdp_ram #(.W(48), .D(NL * NH)) u_ssmp (
    .clk(clk), .we(cfg_we && cfg_sel == 3'd4), .waddr(SAW'(cfg_addr)), .wdata(cfg_data[47:0]),
    .raddr(sp_addr), .rdata(sp_data));

  // residual h
  localparam int DW = $clog2(D);
  logic            h_we;
  logic [DW-1:0]   h_wa, h_ra;
  logic [31:0]     h_wd, h_rd;
  sdp_ram #(.W(32), .D(D)) u_h (
    .clk(clk), .we(h_we || (cfg_we && cfg_sel == 3'd5)),
    .waddr(h_we ? h_wa : DW'(cfg_addr)), .wdata(h_we ? h_wd : cfg_data[31:0]),
    .raddr(h_ra), .rdata(h_rd));

  // z (Q.12) and x (Q.10) of the current layer
  localparam int IW = $clog2(DI);
  logic            z_we, x_we;
  logic [IW-1:0]   z_wa, z_ra, x_wa, x_ra;
  logic [23:0]     z_wd, z_rd;
  logic [15:0]     x_wd, x_rd;
  sdp_ram #(.W(24), .D(DI)) u_z (.clk(clk), .we(z_we), .waddr(z_wa), .wdata(z_wd), .raddr(z_ra), .rdata(z_rd));
  sdp_ram #(.W(16), .D(DI)) u_x (.clk(clk), .we(x_we), .waddr(x_wa), .wdata(x_wd), .raddr(x_ra), .rdata(x_rd));

  logic [NH-1:0][15:0]  dt_q;       // dt_raw, Q.10
  logic [N-1:0][15:0]   B_q, C_q;   // Q.12
  logic [NH-1:0][23:0]  delta_q;    // Q.18
  logic [NH-1:0][17:0]  dA_q;       // Q.16
  logic [NH-1:0][15:0]  D_q;        // Q.12

  // ---- FSM state ---------------------------------------------------------------------------
  typedef enum logic [3:0] {S_IDLE, S_N1, S_IN, S_DISC, S_SSM, S_OUT, S_NEXT} state_t;
  state_t st;
  logic [LAW-1:0] layer;
  logic           first_q;

  // ---- norm_q8 (shared by pre-norm and gated norm) ------------------------------------------
  logic              nq_start, nq_busy, nq_in_valid, nq_done;
  logic [3:0]        nq_nlog2;
  logic [NWA-1:0]    nq_base;
  logic signed [31:0] nq_in_v;
  logic              act_we;
  logic [1:0]        act_g;
  logic [K-1:0][7:0] act_d;
  logic signed [7:0] nq_k;

  norm_q8 #(.NMAX(512), .K(K), .WAW(NWA), .RL(1)) u_norm (
    .clk(clk), .rst_n(rst_n), .start(nq_start), .n_log2(nq_nlog2), .w_base(nq_base), .busy(nq_busy),
    .in_valid(nq_in_valid), .in_v(nq_in_v), .w_addr(nw_addr), .w_data(nw_data),
    .act_we(act_we), .act_g(act_g), .act_d(act_d), .k_out(nq_k), .done(nq_done));

  // ---- lin_unit (shared by in_proj and out_proj; Q.16 / 32-bit output) -----------------------
  logic              ln_start, ln_busy, ln_done, ln_valid;
  logic [LWA-1:0]    ln_base;
  logic [RW-1:0]     ln_rows, ln_row;
  logic [2:0]        ln_groups;
  logic signed [7:0] ln_k;
  logic signed [31:0] ln_y;

  lin_unit #(.K(K), .G_MAX(4), .AW(LWA), .RW(RW), .RL(3), .F_OUT(16), .W_OUT(32)) u_lin (
    .clk(clk), .rst_n(rst_n), .act_we(act_we), .act_g(act_g), .act_d(act_d),
    .start(ln_start), .base(ln_base), .n_rows(ln_rows), .n_groups(ln_groups), .k(ln_k), .busy(ln_busy),
    .w_addr(lw_addr), .w_data(lw_data), .s_data(ls_data),
    .y_valid(ln_valid), .y_row(ln_row), .y_data(ln_y), .done(ln_done));

  // ---- conv_unit -------------------------------------------------------------------------------
  logic              cv_in_valid, cv_valid;
  logic [$clog2(CONV_DIM)-1:0] cv_in_ch;
  logic signed [23:0] cv_in_x, cv_y;
  logic [15:0]       cv_tag;

  conv_unit #(.NCH(CONV_DIM), .NLAYER(NL), .RL(1)) u_conv (
    .clk(clk), .rst_n(rst_n), .in_valid(cv_in_valid), .in_first(first_q),
    .in_layer($clog2(NL)'(layer)), .in_ch(cv_in_ch), .in_x(cv_in_x), .in_tag(16'(cv_in_ch)),
    .p_addr(cp_addr), .p_data(cp_data), .out_valid(cv_valid), .out_y(cv_y), .out_tag(cv_tag));

  // ---- ssm_discretize ----------------------------------------------------------------------------
  logic              ds_in_valid, ds_valid;
  logic [7:0]        ds_in_tag, ds_tag;
  logic signed [23:0] ds_delta;
  logic signed [17:0] ds_dA;

  ssm_discretize #(.EXP_NEAR0(1'b1), .SP_ORDER2(1'b1)) u_disc (
    .clk(clk), .rst_n(rst_n), .in_valid(ds_in_valid), .in_tag(ds_in_tag),
    .in_dt_raw($signed(dt_q[ds_in_tag[$clog2(NH)-1:0]])), .in_dt_bias($signed(sp_data[15:0])),
    .in_A($signed(sp_data[31:16])),
    .out_valid(ds_valid), .out_tag(ds_tag), .out_delta(ds_delta), .out_dA(ds_dA));

  // ---- ssm_scan_unit ------------------------------------------------------------------------------
  logic              sc_load, sc_in_valid, sc_valid;
  logic [HW-1:0]     sc_in_head, sc_head;
  logic [PW-1:0]     sc_in_p, sc_p;
  logic signed [17:0] sc_in_dA;
  logic signed [15:0] sc_in_xdt, sc_in_x, sc_in_D;
  logic signed [23:0] sc_y;

  ssm_scan_unit #(.N(N), .P(P), .NHEADS(NH * NL)) u_scan (
    .clk(clk), .rst_n(rst_n), .tok_load(sc_load), .first_tok(first_q), .B_in(B_q), .C_in(C_q),
    .in_valid(sc_in_valid), .in_head(sc_in_head), .in_p(sc_in_p), .in_dA(sc_in_dA),
    .in_xdt(sc_in_xdt), .in_x(sc_in_x), .in_D(sc_in_D),
    .out_valid(sc_valid), .out_head(sc_head), .out_p(sc_p), .out_y(sc_y));

  // ---- gate_unit ----------------------------------------------------------------------------------------
  logic              gt_in_valid, gt_valid;
  logic signed [23:0] gt_in_y;
  logic signed [31:0] gt_g;

  gate_unit #(.W_Y(24)) u_gate (
    .clk(clk), .rst_n(rst_n), .in_valid(gt_in_valid), .in_y(gt_in_y), .in_z($signed(z_rd)),
    .in_tag(16'd0), .out_valid(gt_valid), .out_g(gt_g), .out_tag());

  // ===============================================================================================
  // datapath glue
  // ===============================================================================================
  function automatic logic signed [23:0] sat24(input logic signed [31:0] v);
    sat24 = (v > 32'sd8388607) ? 24'sd8388607 : (v < -32'sd8388608) ? -24'sd8388608 : v[23:0];
  endfunction
  function automatic logic signed [15:0] sat16(input logic signed [31:0] v);
    sat16 = (v > 32'sd32767) ? 16'sd32767 : (v < -32'sd32768) ? -16'sd32768 : v[15:0];
  endfunction

  // ---- in_proj routing: z | xBC | dt --------------------------------------------------------------
  logic signed [23:0] y24;
  assign y24 = sat24(ln_y);

  always_comb begin
    z_we        = (st == S_IN) && ln_valid && (ln_row < RW'(DI));
    z_wa        = IW'(ln_row);
    z_wd        = y24;
    cv_in_valid = (st == S_IN) && ln_valid && (ln_row >= RW'(DI)) && (ln_row < RW'(DI + CONV_DIM));
    cv_in_ch    = $clog2(CONV_DIM)'(ln_row - RW'(DI));
    cv_in_x     = y24;
  end

  // conv outputs: x (Q.10) | B | C (Q.12)
  always_comb begin
    x_we = cv_valid && (cv_tag < 16'(DI));
    x_wa = IW'(cv_tag);
    x_wd = sat16((32'(cv_y) + 32'sd2) >>> 2);
  end

  logic [$clog2(CONV_DIM+1)-1:0] cv_cnt;
  logic                          ln_fin;

  always_ff @(posedge clk) begin
    if ((st == S_IN) && ln_valid && (ln_row >= RW'(DI + CONV_DIM)))
      dt_q[ln_row - RW'(DI + CONV_DIM)] <= sat16((32'(y24) + 32'sd2) >>> 2);
    if (cv_valid && cv_tag >= 16'(DI) && cv_tag < 16'(DI + N))
      B_q[cv_tag - 16'(DI)] <= sat16(32'(cv_y));
    if (cv_valid && cv_tag >= 16'(DI + N))
      C_q[cv_tag - 16'(DI + N)] <= sat16(32'(cv_y));
  end

  // ---- discretization results ------------------------------------------------------------------------
  logic [$clog2(NH+1)-1:0] ds_cnt;
  always_ff @(posedge clk)
    if (ds_valid) begin
      delta_q[ds_tag[$clog2(NH)-1:0]] <= ds_delta;
      dA_q[ds_tag[$clog2(NH)-1:0]]    <= ds_dA;
    end

  // ---- SSM row stream: x read -> delta * x -> xdt ----------------------------------------------------
  localparam int RWS = $clog2(NH * P + 1);
  logic [RWS-1:0]    r_cnt;
  logic              q1_ok, q2_ok, q3_ok;
  logic [IW-1:0]     q1_r, q2_r, q3_r;
  logic signed [15:0] q2_x, q3_x;
  logic signed [39:0] q2_p;
  logic signed [15:0] q3_xdt;

  assign x_ra = IW'(r_cnt);

  always_ff @(posedge clk) begin
    q1_ok <= rst_n && (st == S_SSM) && (r_cnt < RWS'(NH * P)) && !sc_load;
    q1_r  <= IW'(r_cnt);
    q2_ok <= rst_n & q1_ok;
    q2_r  <= q1_r;
    q2_x  <= $signed(x_rd);
    q2_p  <= $signed(delta_q[q1_r / P]) * $signed(x_rd);
    q3_ok <= rst_n & q2_ok;
    q3_r  <= q2_r;
    q3_x  <= q2_x;
    q3_xdt <= sat16(32'((q2_p + 40'sd8192) >>> 14));
  end

  always_comb begin
    sc_in_valid = q3_ok;
    sc_in_head  = HW'(layer) * HW'(NH) + HW'(q3_r / P);
    sc_in_p     = PW'(q3_r % P);
    sc_in_dA    = $signed(dA_q[q3_r / P]);
    sc_in_xdt   = q3_xdt;
    sc_in_x     = q3_x;
    sc_in_D     = $signed(D_q[q3_r / P]);
  end

  // SSM output -> z read -> gate -> gated norm
  logic              y1_ok;
  logic signed [23:0] y1;
  assign z_ra = IW'((sc_head - HW'(layer) * HW'(NH)) * P + sc_p);

  always_ff @(posedge clk) begin
    y1_ok <= rst_n & sc_valid;
    y1    <= sc_y;
  end
  assign gt_in_valid = y1_ok;
  assign gt_in_y     = y1;

  // ---- out_proj: h += out ---------------------------------------------------------------------------
  logic              o1_ok, o2_ok;
  logic [DW-1:0]     o1_r, o2_r;
  logic signed [31:0] o1_y;
  logic signed [32:0] o2_sum;
  logic [$clog2(D+1)-1:0] h_cnt;

  always_ff @(posedge clk) begin
    o1_ok <= rst_n && (st == S_OUT) && ln_valid;
    o1_r  <= DW'(ln_row);
    o1_y  <= ln_y;
    o2_ok <= rst_n & o1_ok;
    o2_r  <= o1_r;
    o2_sum <= 33'($signed(h_rd)) + 33'(o1_y);
  end

  always_comb begin
    h_we = o2_ok;
    h_wa = o2_r;
    h_wd = (o2_sum > 33'sd2147483647) ? 32'h7FFFFFFF :
           (o2_sum < -33'sd2147483648) ? 32'h80000000 : o2_sum[31:0];
  end

  assign dbg_valid = o2_ok;
  assign dbg_layer = 8'(layer);
  assign dbg_idx   = 12'(o2_r);
  assign dbg_h     = h_wd;

  // ---- h read port: pre-norm stream or residual read-modify-write ---------------------------------------
  logic [$clog2(D+1)-1:0] n1_cnt;
  logic                   n1_ok;
  assign h_ra = (st == S_OUT) ? DW'(ln_row) : DW'(n1_cnt);

  // ===============================================================================================
  // control
  // ===============================================================================================
  logic [$clog2(NH+1)-1:0] dq_cnt;     // discretize issue counter
  logic                    dq_ok;
  logic [7:0]              dq_tag;

  // ssmp read: discretize issue
  assign sp_addr = SAW'(layer) * SAW'(NH) + SAW'(dq_cnt);

  always_ff @(posedge clk) begin
    dq_ok  <= rst_n && (st == S_DISC) && (dq_cnt < NH);
    dq_tag <= 8'(dq_cnt);
    if (dq_ok) D_q[dq_tag[$clog2(NH)-1:0]] <= sp_data[47:32];
  end
  assign ds_in_valid = dq_ok;
  assign ds_in_tag   = dq_tag;

  // pre-norm input stream (h read latency 1)
  always_ff @(posedge clk) n1_ok <= rst_n && (st == S_N1) && (n1_cnt < D) && !nq_start;

  always_comb begin
    nq_in_valid = (st == S_N1) ? n1_ok : gt_valid;
    nq_in_v     = (st == S_N1) ? $signed(h_rd) : gt_g;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st       <= S_IDLE;
      busy     <= 1'b0;
      tok_done <= 1'b0;
      nq_start <= 1'b0;
      ln_start <= 1'b0;
      sc_load  <= 1'b0;
    end else begin
      tok_done <= 1'b0;
      nq_start <= 1'b0;
      ln_start <= 1'b0;
      sc_load  <= 1'b0;
      if (ln_done) ln_fin <= 1'b1;
      if (cv_valid) cv_cnt <= cv_cnt + 1'b1;
      if (ds_valid) ds_cnt <= ds_cnt + 1'b1;
      if (o2_ok)    h_cnt  <= h_cnt + 1'b1;

      case (st)
        S_IDLE: if (tok_go) begin
          busy    <= 1'b1;
          first_q <= tok_first;
          layer   <= '0;
          st      <= S_NEXT;
        end

        S_NEXT: begin                          // start the pre-norm of this layer
          nq_start <= 1'b1;
          nq_nlog2 <= 4'($clog2(D));
          nq_base  <= NWA'(layer) * NWA'(D + DI);
          n1_cnt   <= '0;
          st       <= S_N1;
        end

        S_N1: begin
          if (!nq_start && n1_cnt < D) n1_cnt <= n1_cnt + 1'b1;
          if (nq_done) begin                   // in_proj, Q.12 out: k - 4
            ln_start  <= 1'b1;
            ln_base   <= LWA'(layer) * LWA'(STRIDE);
            ln_rows   <= RW'(IN_ROWS);
            ln_groups <= 3'(IN_G);
            ln_k      <= nq_k - 8'sd4;
            ln_fin    <= 1'b0;
            cv_cnt    <= '0;
            st        <= S_IN;
          end
        end

        S_IN: if (ln_fin && cv_cnt == CONV_DIM && !ln_start) begin
          dq_cnt <= '0;
          ds_cnt <= '0;
          st     <= S_DISC;
        end

        S_DISC: begin
          if (dq_cnt < NH) dq_cnt <= dq_cnt + 1'b1;
          if (ds_cnt == NH) begin
            sc_load  <= 1'b1;                  // B, C of this token
            nq_start <= 1'b1;                  // gated norm, input arrives from the gate
            nq_nlog2 <= 4'($clog2(DI));
            nq_base  <= NWA'(layer) * NWA'(D + DI) + NWA'(D);
            r_cnt    <= '0;
            st       <= S_SSM;
          end
        end

        S_SSM: begin
          if (!sc_load && r_cnt < NH * P) r_cnt <= r_cnt + 1'b1;
          if (nq_done) begin
            ln_start  <= 1'b1;
            ln_base   <= LWA'(layer) * LWA'(STRIDE) + LWA'(IN_WORDS);
            ln_rows   <= RW'(D);
            ln_groups <= 3'(OUT_G);
            ln_k      <= nq_k;
            ln_fin    <= 1'b0;
            h_cnt     <= '0;
            st        <= S_OUT;
          end
        end

        S_OUT: if (ln_fin && h_cnt == D && !ln_start) begin
          if (layer == LAW'(NL - 1)) begin
            busy     <= 1'b0;
            tok_done <= 1'b1;
            st       <= S_IDLE;
          end else begin
            layer <= layer + 1'b1;
            st    <= S_NEXT;
          end
        end

        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
