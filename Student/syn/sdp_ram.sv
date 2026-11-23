// -----------------------------------------------------------------------------
// Simple dual-port RAM (one write port, one read port), registered read
// address. Written in the standard template so Quartus infers M20K directly
// (a memory declared inside a generate loop was first elaborated as
// registers + mux, which is very slow).
//
// Optional pipelining for large memories spread over many M20K blocks:
//   IN_REG  = 1 : extra register on the read address (and the write port),
//                 which Quartus can duplicate to spread the address fan-out
//   OUT_REG = 1 : extra register on the read data (M20K output register /
//                 after the depth mux)
// Read latency (raddr -> rdata) = 1 + IN_REG + OUT_REG cycles.
// Read-during-write to the same address returns don't-care (no_rw_check).
// -----------------------------------------------------------------------------
module sdp_ram #(
  parameter int W       = 40,
  parameter int D       = 6144,
  parameter bit IN_REG  = 1'b0,
  parameter bit OUT_REG = 1'b0,
  parameter int AW      = $clog2(D)          // derived, do not override
)(
  input  logic          clk,
  input  logic          we,
  input  logic [AW-1:0] waddr,
  input  logic [W-1:0]  wdata,
  input  logic [AW-1:0] raddr,
  output logic [W-1:0]  rdata
);
  // ---- optional input register ------------------------------------------------
  // maxfan: let Quartus duplicate these registers so that no copy drives more
  // than 32 M20K blocks (a 40k x 512 memory spans ~1300 blocks)
  (* maxfan = 32 *) logic          we_i;
  (* maxfan = 32 *) logic [AW-1:0] waddr_i, raddr_i;
  (* maxfan = 32 *) logic [W-1:0]  wdata_i;

  if (IN_REG) begin : g_in
    always_ff @(posedge clk) begin
      we_i    <= we;
      waddr_i <= waddr;
      wdata_i <= wdata;
      raddr_i <= raddr;
    end
  end else begin : g_no_in
    assign we_i    = we;
    assign waddr_i = waddr;
    assign wdata_i = wdata;
    assign raddr_i = raddr;
  end

  // ---- memory -------------------------------------------------------------------
  (* ramstyle = "M20K, no_rw_check" *)
  logic [W-1:0] mem [0:D-1];
  logic [W-1:0] rd;

  always_ff @(posedge clk) begin
    if (we_i) mem[waddr_i] <= wdata_i;
    rd <= mem[raddr_i];
  end

  // ---- optional output register -------------------------------------------------
  if (OUT_REG) begin : g_out
    always_ff @(posedge clk) rdata <= rd;
  end else begin : g_no_out
    assign rdata = rd;
  end
endmodule
