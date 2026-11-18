// -----------------------------------------------------------------------------
// Fixed-point helpers (combinational)
//   rshift_round : arithmetic right shift with round-half-up
//                  dout = floor((din + 2^(SHIFT-1)) / 2^SHIFT)
//   saturate     : signed saturation from WI bits to WO bits
// -----------------------------------------------------------------------------
module rshift_round #(
  parameter int WI    = 32,
  parameter int SHIFT = 8,               // must be >= 1
  parameter int WO    = WI - SHIFT + 1   // derived, do not override
)(
  input  logic signed [WI-1:0] din,
  output logic signed [WO-1:0] dout
);
  localparam logic signed [WI:0] HALF = (WI+1)'(1) <<< (SHIFT-1);

  logic signed [WI:0] r;
  assign r    = $signed({din[WI-1], din}) + HALF;   // widened: never overflows
  assign dout = r[WI:SHIFT];
endmodule


module saturate #(
  parameter int WI = 32,
  parameter int WO = 16
)(
  input  logic signed [WI-1:0] din,
  output logic signed [WO-1:0] dout
);
  if (WI <= WO) begin : g_ext
    assign dout = WO'(din);                        // sign extension
  end else begin : g_sat
    localparam logic signed [WI-1:0] MAXV = {{(WI-WO+1){1'b0}}, {(WO-1){1'b1}}};
    localparam logic signed [WI-1:0] MINV = {{(WI-WO+1){1'b1}}, {(WO-1){1'b0}}};
    always_comb begin
      if      (din > MAXV) dout = MAXV[WO-1:0];
      else if (din < MINV) dout = MINV[WO-1:0];
      else                 dout = din[WO-1:0];
    end
  end
endmodule
