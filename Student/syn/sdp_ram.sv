// -----------------------------------------------------------------------------
// Simple dual-port RAM (one write port, one read port), registered read
// address: rdata is valid one cycle after raddr. Written in the standard
// template so Quartus infers M20K directly (a memory declared inside a
// generate loop was first elaborated as registers + mux, which is very slow).
// Read-during-write to the same address returns don't-care (no_rw_check).
// -----------------------------------------------------------------------------
module sdp_ram #(
  parameter int W  = 40,
  parameter int D  = 6144,
  parameter int AW = $clog2(D)          // derived, do not override
)(
  input  logic          clk,
  input  logic          we,
  input  logic [AW-1:0] waddr,
  input  logic [W-1:0]  wdata,
  input  logic [AW-1:0] raddr,
  output logic [W-1:0]  rdata
);
  (* ramstyle = "M20K, no_rw_check" *)
  logic [W-1:0] mem [0:D-1];

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end
endmodule
