// -----------------------------------------------------------------------------
// Fully pipelined binary adder tree (signed, lossless)
//   latency = LVL = log2(N) cycles, N must be a power of two
// -----------------------------------------------------------------------------
module adder_tree #(
  parameter int N   = 128,
  parameter int WI  = 40,
  parameter int LVL = $clog2(N),   // derived, do not override
  parameter int WO  = WI + LVL     // derived, do not override
)(
  input  logic                 clk,
  input  logic [N-1:0][WI-1:0] din,
  output logic signed [WO-1:0] dout
);
  for (genvar l = 0; l <= LVL; l++) begin : g_lvl
    logic [(N>>l)-1:0][WO-1:0] v;

    if (l == 0) begin : g_in
      for (genvar i = 0; i < N; i++) begin : g_ext
        assign v[i] = WO'($signed(din[i]));
      end
    end else begin : g_add
      always_ff @(posedge clk)
        for (int k = 0; k < (N >> l); k++)
          v[k] <= $signed(g_lvl[l-1].v[2*k]) + $signed(g_lvl[l-1].v[2*k+1]);
    end
  end

  assign dout = g_lvl[LVL].v[0];
endmodule
