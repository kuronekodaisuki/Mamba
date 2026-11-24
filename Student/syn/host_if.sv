// -----------------------------------------------------------------------------
// Host interface of mamba_core: Avalon-MM slave for a PCIe BAR (Arria 10 hard IP
// rxm_bar0, AVW = 128 at 250 MHz in Gen2 x8 mode), same clock as the core.
//
// BAR0 map (byte offsets, little endian; registers are 64-bit slots, the fields
// live in the low 32 bits, so 32-bit or 64-bit accesses both work):
//   0x0000 ID        RO  {"MAMB" (0x4D414D42), version[15:0], D[15:0]}
//   0x0008 STATUS    RO  [0] busy  [1] done (sticky)  [47:16] tokens completed
//   0x0010 CTRL      WO  [0] go (start one token)  [1] first token of a sequence
//                        [2] clear done            (go also clears done)
//   0x0018 CFG_PTR   RW  [15:0] word address  [18:16] memory select (cfg_sel)
//   0x0020 CAP_LAYER RW  [7:0] layer whose residual output is captured into OUT
//                        (reset value NL-1 = token output)
//   0x0028 CYCLES    RO  [31:0] clock cycles of the last token (go -> done)
//   0x0040..0x007F CFG_WIN WO  one 512-bit host word (byte i = bits 8i+7:8i).
//                        The write that covers byte 63 commits the word to
//                        memory CFG_PTR.sel at CFG_PTR.addr, then addr += 1.
//                        Narrower memories take the low bits. Write the bytes
//                        of a word in order, ending with byte 63 (uncached
//                        stores do this); for narrow memories writing only the
//                        low bytes and then byte 63 is enough.
//   0x1000..        OUT  RO  D x 32-bit residual values of layer CAP_LAYER of
//                        the last token (filled while the token runs)
//
// Memory selects (mamba_core cfg port): 0 linw  1 lins  2 normw  3 convp
// 4 ssmp  5 h (token input, D x 32-bit Q.16 words).
//
// Avalon: byte addresses, bursts (burstcount) for reads and writes, reads
// pipelined with readdatavalid (latency 3). waitrequest is only asserted while
// the beats of a read burst are being issued.
// -----------------------------------------------------------------------------
module host_if #(
  parameter int AVW = 128,                 // Avalon data width (64, 128, 256)
  parameter int AW  = 16,                  // byte address bits decoded (BAR size)
  parameter int BCW = 6,                   // burstcount width
  parameter int D   = 256,                 // d_model (OUT entries)
  parameter int NL  = 12
)(
  input  logic                clk,
  input  logic                rst_n,
  // Avalon-MM slave
  input  logic [AW-1:0]       avs_address,
  input  logic                avs_read,
  input  logic                avs_write,
  input  logic [AVW-1:0]      avs_writedata,
  input  logic [AVW/8-1:0]    avs_byteenable,
  input  logic [BCW-1:0]      avs_burstcount,
  output logic                avs_waitrequest,
  output logic [AVW-1:0]      avs_readdata,
  output logic                avs_readdatavalid,
  // mamba_core
  output logic                cfg_we,
  output logic [2:0]          cfg_sel,
  output logic [15:0]         cfg_addr,
  output logic [511:0]        cfg_data,
  output logic                tok_go,
  output logic                tok_first,
  input  logic                busy,
  input  logic                tok_done,
  input  logic                dbg_valid,
  input  logic [7:0]          dbg_layer,
  input  logic [11:0]         dbg_idx,
  input  logic [31:0]         dbg_h
);
  localparam int AVB   = AVW / 8;                  // bytes per beat
  localparam int LB    = $clog2(AVB);
  localparam int NLN   = AVW / 64;                 // 64-bit lanes per beat
  localparam int NOB   = AVW / 32;                 // OUT entries per beat (banks)
  localparam int OBD   = D / NOB;                  // OUT bank depth
  localparam int OBA   = (OBD > 1) ? $clog2(OBD) : 1;
  localparam logic [15:0] VERSION = 16'd1;

  localparam logic [AW-1:0] A_ID     = AW'('h0000);
  localparam logic [AW-1:0] A_STATUS = AW'('h0008);
  localparam logic [AW-1:0] A_CTRL   = AW'('h0010);
  localparam logic [AW-1:0] A_PTR    = AW'('h0018);
  localparam logic [AW-1:0] A_CAP    = AW'('h0020);
  localparam logic [AW-1:0] A_CYC    = AW'('h0028);
  localparam logic [AW-1:0] A_WIN    = AW'('h0040);   // 64 bytes
  localparam logic [AW-1:0] A_OUT    = AW'('h1000);   // D * 4 bytes

  // synthesis translate_off
  initial begin
    if (AVW < 64 || AVW > 512 || (AVW & (AVW - 1)) != 0)
      $fatal(1, "host_if: AVW must be 64, 128, 256 or 512");
    if (D % NOB != 0 || D * 4 > 'h1000)
      $fatal(1, "host_if: D must be a multiple of AVW/32 and fit 4 KB");
  end
  // synthesis translate_on

  // ===========================================================================
  // write path: one beat per cycle, burst address generation
  // ===========================================================================
  logic [BCW-1:0] wr_left;          // beats still to come in the current write burst
  logic [AW-1:0]  wr_next;          // address of the next beat of that burst
  logic           rd_active;        // read burst beats being issued
  logic           w_ok;
  logic [AW-1:0]  w_addr;           // beat address (aligned to AVB)

  assign avs_waitrequest = rd_active;
  assign w_ok   = avs_write && !avs_waitrequest;
  assign w_addr = (wr_left != '0) ? wr_next : {avs_address[AW-1:LB], LB'(0)};

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wr_left <= '0;
    end else if (w_ok) begin
      if (wr_left != '0) wr_left <= wr_left - 1'b1;
      else               wr_left <= (avs_burstcount > 1) ? avs_burstcount - 1'b1 : '0;
      wr_next <= w_addr + AW'(AVB);
    end
  end

  // ---- registers ---------------------------------------------------------------------
  logic [2:0]   ptr_sel;
  logic [15:0]  ptr_addr;
  logic [7:0]   cap_layer;
  logic         done_q;
  logic [31:0]  tok_cnt, cyc_cnt, cyc_last;
  logic         run_q;
  logic         commit;
  logic [63:0][7:0] win;            // staged 512-bit word

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ptr_sel   <= '0;
      ptr_addr  <= '0;
      cap_layer <= 8'(NL - 1);
      done_q    <= 1'b0;
      tok_cnt   <= '0;
      cyc_last  <= '0;
      run_q     <= 1'b0;
      tok_go    <= 1'b0;
      tok_first <= 1'b0;
      commit    <= 1'b0;
    end else begin
      tok_go <= 1'b0;
      commit <= 1'b0;
      // the word staged by the previous beat is committed now (cfg_addr = current
      // pointer), then the pointer advances; a PTR write in this same cycle wins
      // (it comes from the host after that word, e.g. back-to-back on the bus)
      if (commit) ptr_addr <= ptr_addr + 1'b1;
      if (w_ok) begin
        for (int j = 0; j < NLN; j++) begin
          automatic logic [AW-1:0] ra = w_addr + AW'(8 * j);
          automatic logic [31:0]   wd = avs_writedata[64*j +: 32];
          if (avs_byteenable[8*j +: 4] != '0) begin
            case (ra)
              A_CTRL: begin
                if (wd[0] && !busy && !run_q) begin
                  tok_go    <= 1'b1;
                  tok_first <= wd[1];
                end
                if (wd[0] || wd[2]) done_q <= 1'b0;
              end
              A_PTR: begin
                ptr_addr <= wd[15:0];
                ptr_sel  <= wd[18:16];
              end
              A_CAP: cap_layer <= wd[7:0];
              default: ;
            endcase
          end
        end
        // staging window
        for (int b = 0; b < AVB; b++) begin
          automatic logic [AW-1:0] ba = w_addr + AW'(b);
          if (avs_byteenable[b] && ba >= A_WIN && ba < A_WIN + AW'(64)) begin
            win[ba[5:0]] <= avs_writedata[8*b +: 8];
            if (ba[5:0] == 6'd63) commit <= 1'b1;
          end
        end
      end
      // token bookkeeping
      if (tok_go) run_q <= 1'b1;
      if (tok_done) begin
        run_q    <= 1'b0;
        done_q   <= 1'b1;
        tok_cnt  <= tok_cnt + 1'b1;
        cyc_last <= cyc_cnt;
      end
    end
  end

  always_ff @(posedge clk)
    if (tok_go) cyc_cnt <= 32'd1;
    else if (run_q) cyc_cnt <= cyc_cnt + 1'b1;

  // the word is in `win` one cycle after its last byte
  assign cfg_we   = commit;
  assign cfg_sel  = ptr_sel;
  assign cfg_addr = ptr_addr;
  assign cfg_data = win;

  // ---- OUT buffer: NOB banks, entry i in bank i % NOB at row i / NOB --------------------
  logic             ob_we;
  logic [OBA-1:0]   ob_wa;
  logic [$clog2(NOB)-1:0] ob_wb;
  logic [31:0]      ob_wd;

  always_ff @(posedge clk) begin
    ob_we <= dbg_valid && (dbg_layer == cap_layer) && (dbg_idx < 12'(D));
    ob_wb <= dbg_idx[$clog2(NOB)-1:0];
    ob_wa <= OBA'(dbg_idx >> $clog2(NOB));
    ob_wd <= dbg_h;
  end

  // ===========================================================================
  // read path: beats issued one per cycle (bursts hold waitrequest),
  // r0 issue -> r1 (RAM data) -> r2 (registers sampled) -> readdata
  // ===========================================================================
  logic [BCW-1:0] rd_left;
  logic [AW-1:0]  rd_next;
  logic           r0_v;
  logic [AW-1:0]  r0_a;

  always_comb begin
    r0_v = 1'b0;
    r0_a = rd_next;
    if (rd_active) begin
      r0_v = 1'b1;
    end else if (avs_read) begin
      r0_v = 1'b1;
      r0_a = {avs_address[AW-1:LB], LB'(0)};
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_active <= 1'b0;
      rd_left   <= '0;
    end else if (rd_active) begin
      rd_next <= rd_next + AW'(AVB);
      rd_left <= rd_left - 1'b1;
      if (rd_left == BCW'(1)) rd_active <= 1'b0;
    end else if (avs_read) begin
      rd_next <= r0_a + AW'(AVB);
      if (avs_burstcount > 1) begin
        rd_active <= 1'b1;
        rd_left   <= avs_burstcount - 1'b1;
      end
    end
  end

  logic [AVW-1:0] ob_rd;
  for (genvar b = 0; b < NOB; b++) begin : g_ob
    sdp_ram #(.W(32), .D(OBD)) u_ob (
      .clk(clk), .we(ob_we && ob_wb == b), .waddr(ob_wa), .wdata(ob_wd),
      .raddr(OBA'((r0_a - A_OUT) >> LB)), .rdata(ob_rd[32*b +: 32]));
  end

  // r1: OUT RAM data ready; registers are sampled one cycle after the beat was
  // issued, so a read that follows a write on the bus sees its effect, including
  // the pointer increment of a CFG_WIN commit (PCIe ordering)
  logic           r1_v, r1_out, r2_v, r2_out;
  logic [AW-1:0]  r1_a;
  logic [AVW-1:0] r2_ob;
  logic [NLN-1:0][63:0] r2_lane;

  always_ff @(posedge clk) begin
    r1_v   <= rst_n & r0_v;
    r1_a   <= r0_a;
    r1_out <= (r0_a >= A_OUT) && (r0_a < A_OUT + AW'(D * 4));
    r2_v   <= rst_n & r1_v;
    r2_out <= r1_out;
    r2_ob  <= ob_rd;
    // per-lane register read (a 128-bit beat covers two 64-bit register slots)
    for (int j = 0; j < NLN; j++) begin
      automatic logic [AW-1:0] ra = r1_a + AW'(8 * j);
      case (ra)
        A_ID:     r2_lane[j] <= {32'h4D414D42, VERSION, 16'(D)};
        A_STATUS: r2_lane[j] <= {16'd0, tok_cnt, 14'd0, done_q, busy | run_q};
        A_PTR:    r2_lane[j] <= {45'd0, ptr_sel, ptr_addr};
        A_CAP:    r2_lane[j] <= {56'd0, cap_layer};
        A_CYC:    r2_lane[j] <= {32'd0, cyc_last};
        default:  r2_lane[j] <= '0;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    avs_readdatavalid <= rst_n & r2_v;
    avs_readdata      <= r2_out ? r2_ob : AVW'(r2_lane);
  end
endmodule
