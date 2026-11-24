// -----------------------------------------------------------------------------
// Board top for the Arria 10 GX FPGA Development Kit: PCIe hard IP (Gen2 x8,
// Avalon-MM, 128-bit at 250 MHz) -> host_if (BAR0) -> mamba_core.
// Everything runs on the hard IP application clock coreclkout_hip (250 MHz),
// so there is no clock crossing between the BAR and the core.
//
// Pins (from the Intel A10 dev kit PCIe example design, pcie_top.qsf):
//   pcie_refclk   AL37 (HCSL, 100 MHz from the edge connector)
//   pcie_perstn   BC30 (PERST#)     npor_n   T12 (board reset)
//   pcie_rx/tx    8 lanes of the edge connector
//   led_n[3:0]    K26 L28 K25 BA27, active low:
//                 [0] heartbeat  [1] link (app reset released)  [2] core busy
//                 [3] done flag of the last token
//
// Host side: see host_if.sv for the BAR0 map and host/mamba_host.py.
// -----------------------------------------------------------------------------
module pcie_top (
  input  logic       pcie_refclk,
  input  logic       pcie_perstn,
  input  logic       npor_n,
  input  logic [7:0] pcie_rx,
  output logic [7:0] pcie_tx,
  output logic [3:0] led_n
);
  localparam int D = 256, DI = 512, N = 64, P = 64, NL = 12;

  logic         clk;               // coreclkout_hip, 250 MHz
  logic         app_rst_n;
  logic [63:0]  bar_address;
  logic [15:0]  bar_byteenable;
  logic [127:0] bar_readdata, bar_writedata;
  logic         bar_read, bar_write, bar_readdatavalid, bar_waitrequest;
  logic [5:0]   bar_burstcount;

  pcie_hip u_hip (
    .coreclkout_hip(clk),
    .refclk(pcie_refclk),
    .npor(npor_n),
    .pin_perst(pcie_perstn),
    .app_nreset_status(app_rst_n),
    .test_in(32'd0),
    .simu_mode_pipe(1'b0),
    .sim_pipe_pclk_in(1'b0),
    .sim_pipe_rate(),
    .sim_ltssmstate(),
    .phystatus0(1'd0),
    .phystatus1(1'd0),
    .phystatus2(1'd0),
    .phystatus3(1'd0),
    .phystatus4(1'd0),
    .phystatus5(1'd0),
    .phystatus6(1'd0),
    .phystatus7(1'd0),
    .rxdata0(32'd0),
    .rxdata1(32'd0),
    .rxdata2(32'd0),
    .rxdata3(32'd0),
    .rxdata4(32'd0),
    .rxdata5(32'd0),
    .rxdata6(32'd0),
    .rxdata7(32'd0),
    .rxdatak0(4'd0),
    .rxdatak1(4'd0),
    .rxdatak2(4'd0),
    .rxdatak3(4'd0),
    .rxdatak4(4'd0),
    .rxdatak5(4'd0),
    .rxdatak6(4'd0),
    .rxdatak7(4'd0),
    .rxelecidle0(1'd0),
    .rxelecidle1(1'd0),
    .rxelecidle2(1'd0),
    .rxelecidle3(1'd0),
    .rxelecidle4(1'd0),
    .rxelecidle5(1'd0),
    .rxelecidle6(1'd0),
    .rxelecidle7(1'd0),
    .rxstatus0(3'd0),
    .rxstatus1(3'd0),
    .rxstatus2(3'd0),
    .rxstatus3(3'd0),
    .rxstatus4(3'd0),
    .rxstatus5(3'd0),
    .rxstatus6(3'd0),
    .rxstatus7(3'd0),
    .rxvalid0(1'd0),
    .rxvalid1(1'd0),
    .rxvalid2(1'd0),
    .rxvalid3(1'd0),
    .rxvalid4(1'd0),
    .rxvalid5(1'd0),
    .rxvalid6(1'd0),
    .rxvalid7(1'd0),
    .rxdataskip0(1'd0),
    .rxdataskip1(1'd0),
    .rxdataskip2(1'd0),
    .rxdataskip3(1'd0),
    .rxdataskip4(1'd0),
    .rxdataskip5(1'd0),
    .rxdataskip6(1'd0),
    .rxdataskip7(1'd0),
    .rxblkst0(1'd0),
    .rxblkst1(1'd0),
    .rxblkst2(1'd0),
    .rxblkst3(1'd0),
    .rxblkst4(1'd0),
    .rxblkst5(1'd0),
    .rxblkst6(1'd0),
    .rxblkst7(1'd0),
    .rxsynchd0(2'd0),
    .rxsynchd1(2'd0),
    .rxsynchd2(2'd0),
    .rxsynchd3(2'd0),
    .rxsynchd4(2'd0),
    .rxsynchd5(2'd0),
    .rxsynchd6(2'd0),
    .rxsynchd7(2'd0),
    .rx_in0(pcie_rx[0]), .rx_in1(pcie_rx[1]), .rx_in2(pcie_rx[2]), .rx_in3(pcie_rx[3]),
    .rx_in4(pcie_rx[4]), .rx_in5(pcie_rx[5]), .rx_in6(pcie_rx[6]), .rx_in7(pcie_rx[7]),
    .tx_out0(pcie_tx[0]), .tx_out1(pcie_tx[1]), .tx_out2(pcie_tx[2]), .tx_out3(pcie_tx[3]),
    .tx_out4(pcie_tx[4]), .tx_out5(pcie_tx[5]), .tx_out6(pcie_tx[6]), .tx_out7(pcie_tx[7]),
    // FPGA-initiated requests are not used
    .txs_address_i('0), .txs_chipselect_i(1'b0), .txs_byteenable_i('0), .txs_readdata_o(),
    .txs_writedata_i('0), .txs_read_i(1'b0), .txs_write_i(1'b0), .txs_burstcount_i(6'd1),
    .txs_readdatavalid_o(), .txs_waitrequest_o(),
    // BAR0
    .rxm_bar0_address_o(bar_address), .rxm_bar0_byteenable_o(bar_byteenable),
    .rxm_bar0_readdata_i(bar_readdata), .rxm_bar0_writedata_o(bar_writedata),
    .rxm_bar0_read_o(bar_read), .rxm_bar0_write_o(bar_write),
    .rxm_bar0_burstcount_o(bar_burstcount), .rxm_bar0_readdatavalid_i(bar_readdatavalid),
    .rxm_bar0_waitrequest_i(bar_waitrequest)
  );

  // core / interface reset: the hard IP application reset, synchronized
  logic [2:0] rst_sync;
  logic       rst_n;
  always_ff @(posedge clk or negedge app_rst_n)
    if (!app_rst_n) rst_sync <= '0;
    else            rst_sync <= {rst_sync[1:0], 1'b1};
  assign rst_n = rst_sync[2];

  logic          cfg_we, tok_go, tok_first, busy, tok_done, dbg_valid;
  logic [2:0]    cfg_sel;
  logic [15:0]   cfg_addr;
  logic [511:0]  cfg_data;
  logic [7:0]    dbg_layer;
  logic [11:0]   dbg_idx;
  logic [31:0]   dbg_h;

  host_if #(.AVW(128), .AW(16), .BCW(6), .D(D), .NL(NL)) u_if (
    .clk(clk), .rst_n(rst_n),
    .avs_address(bar_address[15:0]), .avs_read(bar_read), .avs_write(bar_write),
    .avs_writedata(bar_writedata), .avs_byteenable(bar_byteenable),
    .avs_burstcount(bar_burstcount), .avs_waitrequest(bar_waitrequest),
    .avs_readdata(bar_readdata), .avs_readdatavalid(bar_readdatavalid),
    .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
    .tok_go(tok_go), .tok_first(tok_first), .busy(busy), .tok_done(tok_done),
    .dbg_valid(dbg_valid), .dbg_layer(dbg_layer), .dbg_idx(dbg_idx), .dbg_h(dbg_h));

  mamba_core #(.D(D), .DI(DI), .N(N), .P(P), .NL(NL)) u_core (
    .clk(clk), .rst_n(rst_n),
    .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
    .tok_go(tok_go), .tok_first(tok_first), .busy(busy), .tok_done(tok_done),
    .dbg_valid(dbg_valid), .dbg_layer(dbg_layer), .dbg_idx(dbg_idx), .dbg_h(dbg_h));

  // ---- LEDs ----------------------------------------------------------------------------
  logic [27:0] hb;
  logic        done_led;
  always_ff @(posedge clk) begin
    hb <= hb + 1'b1;
    if (!rst_n)        done_led <= 1'b0;
    else if (tok_go)   done_led <= 1'b0;
    else if (tok_done) done_led <= 1'b1;
    led_n <= ~{done_led, busy, rst_n, hb[27]};
  end
endmodule
