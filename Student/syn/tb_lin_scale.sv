`timescale 1ns/1ps
// Self-checking testbench for lin_scale.sv
//   python3 model/lin_golden.py --scale --out vec
// Beats are streamed with random bubbles; Q.F_OUT results are checked bit for bit.
module tb_lin_scale;
  localparam int MAXB = 20000, MAXR = 10000;
  localparam int F_OUT = 12, W_OUT = 24;     // must match lin_golden.py --F / --W

  logic clk = 1'b0;
  always #2 clk = ~clk;

  logic               rst_n = 1'b0;
  logic               in_valid = 1'b0, in_first = 1'b0, in_last = 1'b0;
  logic signed [18:0] in_acc = '0;
  logic [15:0]        in_scale = '0;
  logic signed [7:0]  in_k = '0;
  logic [15:0]        in_tag = '0;
  logic               out_valid;
  logic [15:0]        out_tag;
  logic signed [W_OUT-1:0] out_y;

  lin_scale #(.F_OUT(F_OUT), .W_OUT(W_OUT)) dut (.*);

  logic [44:0] beat [MAXB];
  logic [W_OUT-1:0] yexp [MAXR];
  int nb = 0, nr = 0;

  initial begin
    int fd;
    logic [44:0] w;
    logic [W_OUT-1:0] y;
    fd = $fopen("vec/lsc_in.hex", "r");
    while ($fscanf(fd, "%h", w) == 1) beat[nb++] = w;
    $fclose(fd);
    fd = $fopen("vec/lsc_y.hex", "r");
    while ($fscanf(fd, "%h", y) == 1) yexp[nr++] = y;
    $fclose(fd);
    $display("loaded %0d beats, %0d rows", nb, nr);
  end

  initial begin
    int row = 0;
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    for (int i = 0; i < nb; ) begin
      @(posedge clk);
      if ($urandom_range(0, 3) == 0) begin
        in_valid <= 1'b0;
      end else begin
        in_valid <= 1'b1;
        in_first <= beat[i][44];
        in_last  <= beat[i][43];
        in_k     <= beat[i][42:35];
        in_scale <= beat[i][34:19];
        in_acc   <= beat[i][18:0];
        in_tag   <= 16'(row);
        if (beat[i][43]) row++;
        i++;
      end
    end
    @(posedge clk);
    in_valid <= 1'b0;
  end

  int idx = 0, errors = 0;
  always @(posedge clk) begin
    if (out_valid) begin
      if (out_y !== yexp[idx] || out_tag !== 16'(idx)) begin
        errors++;
        if (errors <= 10)
          $display("MISMATCH row %0d tag=%0d got=%h exp=%h", idx, out_tag, out_y, yexp[idx]);
      end
      idx++;
    end
  end

  initial begin
    wait (nr > 0 && idx == nr);
    repeat (2) @(posedge clk);
    if (errors == 0) $display("PASS: %0d rows match the golden model", idx);
    else             $display("FAIL: %0d / %0d mismatches", errors, idx);
    $finish;
  end

  initial begin
    #5_000_000;
    $display("TIMEOUT (received %0d outputs)", idx);
    $finish;
  end
endmodule
