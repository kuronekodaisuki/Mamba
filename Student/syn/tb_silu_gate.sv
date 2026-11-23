`timescale 1ns/1ps
// Self-checking testbench for silu_unit.sv and gate_unit.sv
//   python3 model/vec_golden.py --elem --out vec
module tb_silu_gate;
  localparam int NV = 4000;

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic               rst_n = 1'b0;
  logic               in_valid = 1'b0;
  logic signed [23:0] in_y = '0, in_z = '0;
  logic [15:0]        in_tag = '0;

  logic               s_valid, g_valid;
  logic signed [23:0] s_y;
  logic signed [31:0] g_g;
  logic [15:0]        s_tag, g_tag;

  silu_unit u_silu (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_x(in_z), .in_tag(in_tag),
                    .out_valid(s_valid), .out_y(s_y), .out_tag(s_tag));
  gate_unit u_gate (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_y(in_y), .in_z(in_z),
                    .in_tag(in_tag), .out_valid(g_valid), .out_g(g_g), .out_tag(g_tag));

  logic [47:0] vin [NV];
  logic [23:0] vs  [NV];
  logic [31:0] vg  [NV];
  initial begin
    $readmemh("vec/sg_in.hex", vin);
    $readmemh("vec/sg_s.hex",  vs);
    $readmemh("vec/sg_g.hex",  vg);
  end

  initial begin
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    for (int i = 0; i < NV; ) begin
      @(posedge clk);
      if ($urandom_range(0, 3) == 0) in_valid <= 1'b0;
      else begin
        in_valid <= 1'b1;
        in_y     <= vin[i][47:24];
        in_z     <= vin[i][23:0];
        in_tag   <= 16'(i);
        i++;
      end
    end
    @(posedge clk);
    in_valid <= 1'b0;
  end

  int si = 0, gi = 0, errors = 0;
  always @(posedge clk) begin
    if (s_valid) begin
      if (s_y !== $signed(vs[si]) || s_tag !== 16'(si)) begin
        errors++;
        if (errors <= 10) $display("SILU MISMATCH %0d got=%0d exp=%0d", si, s_y, $signed(vs[si]));
      end
      si++;
    end
    if (g_valid) begin
      if (g_g !== $signed(vg[gi]) || g_tag !== 16'(gi)) begin
        errors++;
        if (errors <= 10) $display("GATE MISMATCH %0d got=%0d exp=%0d", gi, g_g, $signed(vg[gi]));
      end
      gi++;
    end
  end

  initial begin
    wait (si == NV && gi == NV);
    repeat (2) @(posedge clk);
    if (errors == 0) $display("PASS: %0d SiLU and %0d gate outputs match the golden model", si, gi);
    else             $display("FAIL: %0d mismatches", errors);
    $finish;
  end

  initial begin
    #1_000_000;
    $display("TIMEOUT (silu %0d, gate %0d)", si, gi);
    $finish;
  end
endmodule
