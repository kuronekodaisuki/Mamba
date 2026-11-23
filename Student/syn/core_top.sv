// -----------------------------------------------------------------------------
// Synthesis wrapper of mamba_core (student configuration) for fmax / resources
// on the Arria 10 GX dev kit device. Not for the board: the host port will be
// PCIe later.
//
// The 512-bit host write port is fed from a 1-bit serial shift chain
// (cfg_data | cfg_addr | cfg_sel), latched on ctl[0]; ctl[1] / ctl[2] drive
// tok_go / tok_first. All outputs are XOR-reduced to one pin, so nothing is
// optimized away. I/O paths are cut in top.sdc.
// -----------------------------------------------------------------------------
module core_top (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       sin,
  input  logic [2:0] ctl,
  output logic       sout
);
  localparam int W_CH = 512 + 16 + 3;

  logic             rst_q, rst_qq;
  logic [2:0]       ctl_q;
  logic [W_CH-1:0]  sh;

  logic             cfg_we, tok_go, tok_first;
  logic [2:0]       cfg_sel;
  logic [15:0]      cfg_addr;
  logic [511:0]     cfg_data;

  always_ff @(posedge clk) begin
    rst_q  <= rst_n;
    rst_qq <= rst_q;
    ctl_q  <= ctl;
    sh     <= {sh[W_CH-2:0], sin};
    cfg_we    <= ctl_q[0];
    tok_go    <= ctl_q[1];
    tok_first <= ctl_q[2];
    if (ctl_q[0]) {cfg_data, cfg_addr, cfg_sel} <= sh;
  end

  logic          busy, tok_done, dbg_valid;
  logic [7:0]    dbg_layer;
  logic [11:0]   dbg_idx;
  logic [31:0]   dbg_h;

  mamba_core #(.D(256), .DI(512), .N(64), .P(64), .NL(12)) u_core (
    .clk(clk), .rst_n(rst_qq),
    .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
    .tok_go(tok_go), .tok_first(tok_first), .busy(busy), .tok_done(tok_done),
    .dbg_valid(dbg_valid), .dbg_layer(dbg_layer), .dbg_idx(dbg_idx), .dbg_h(dbg_h));

  always_ff @(posedge clk)
    sout <= ^{busy, tok_done, dbg_valid, dbg_layer, dbg_idx, dbg_h};
endmodule
