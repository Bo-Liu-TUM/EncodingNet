// http://hdlbits.01xz.net/wiki/Iverilog?load=3I39iv
module top_module ();
    reg clk = 0;
    always #5 clk = ~clk;  // Create clock with period=10
    
    initial `probe_start;   // Start the timing diagram
    `probe(clk);        // Probe signal "clk"

    reg reset = 1;
    reg en    = 0;
    reg w_en  = 1;
    // A testbench
    reg  signed [ 7:0] a_left   = 8'd0;
    wire signed [ 7:0] a_right;
    reg  signed [ 7:0] w_in     = 8'd1;
    wire signed [ 7:0] w_out;
    reg  signed [16:0] sum_in   = 8'd0;
    wire signed [16:0] sum_out;

    initial begin
        #0 reset <= 1;
        #2 reset <= 0;
        #2 reset <= 1;
    end
    initial begin
        #0 en <= 0;
        #4 en <= 1;
    end
    initial begin
        #0  w_en <= 1;
        #10 w_en <= 0;
        #40 w_en <= 1;
        #10 w_en <= 0;
    end
    
    initial begin
        #0 w_in <= 1;
        #50 w_in <= -1;
    end
    
    wire signed [16:0] real_sum_out;
    assign real_sum_out = a_left * w_in + sum_in;
    initial begin
        #5
        #10 $strobe("a_left * w_in + sum_in = %d * %d + %d = %d, sum_out = %d (%h at %0d ps)", a_left, w_in, sum_in, real_sum_out, sum_out, sum_out, $time);
        #10 $strobe("a_left * w_in + sum_in = %d * %d + %d = %d, sum_out = %d (%h at %0d ps)", a_left, w_in, sum_in, real_sum_out, sum_out, sum_out, $time);
        #10 $strobe("a_left * w_in + sum_in = %d * %d + %d = %d, sum_out = %d (%h at %0d ps)", a_left, w_in, sum_in, real_sum_out, sum_out, sum_out, $time);
        #10 $strobe("a_left * w_in + sum_in = %d * %d + %d = %d, sum_out = %d (%h at %0d ps)", a_left, w_in, sum_in, real_sum_out, sum_out, sum_out, $time);
        #10
        #10 $strobe("a_left * w_in + sum_in = %d * %d + %d = %d, sum_out = %d (%h at %0d ps)", a_left, w_in, sum_in, real_sum_out, sum_out, sum_out, $time);
        #10 $strobe("a_left * w_in + sum_in = %d * %d + %d = %d, sum_out = %d (%h at %0d ps)", a_left, w_in, sum_in, real_sum_out, sum_out, sum_out, $time);
        #10 $strobe("a_left * w_in + sum_in = %d * %d + %d = %d, sum_out = %d (%h at %0d ps)", a_left, w_in, sum_in, real_sum_out, sum_out, sum_out, $time);
        #10 $strobe("a_left * w_in + sum_in = %d * %d + %d = %d, sum_out = %d (%h at %0d ps)", a_left, w_in, sum_in, real_sum_out, sum_out, sum_out, $time);
    end
    initial begin
        #10 a_left <= 1; sum_in <= 2;
        #10 a_left <= -2; sum_in <= -3;
        #10 a_left <= 3; sum_in <= -4;
        #10 a_left <= -4; sum_in <= 5;
        #10
        #10 a_left <= 1; sum_in <= 2;
        #10 a_left <= -2; sum_in <= -3;
        #10 a_left <= 3; sum_in <= -4;
        #10 a_left <= -4; sum_in <= 5;
        #10 $finish;            // Quit the simulation
    end
    
    `probe(reset);
    `probe(en);
    `probe(w_en);
    
    `probe(w_in);
    `probe(w_out);
    
    `probe(a_left);
    `probe(a_right);
    
    `probe(sum_in);
    `probe(sum_out);
    
    PE inst1 (.CLK(clk), .RESET(reset), .EN(en), .W_EN(w_en), 
              .active_left(a_left), .active_right(a_right),
              .in_weight_above(w_in), .out_weight_below(w_out),
              .in_sum(sum_in), .out_sum(sum_out)
             );

endmodule



module PE #(
    parameter  a_bit = 8,
    parameter  w_bit = 8,
    parameter  sum_bit = 17
    )(
    // interface to system
    input wire CLK,                         // CLK
    input wire RESET,                       // RESET, Negedge is active
    input wire EN,                          // enable signal for the accelerator, high for active
    input wire W_EN,                         // enable weight to flow
    // interface to PE row .....
    input wire signed [a_bit-1:0] active_left,
    output reg signed [a_bit-1:0] active_right,

    input wire signed [sum_bit-1:0] in_sum,
    output reg signed [sum_bit-1:0] out_sum,

    input wire signed [w_bit-1:0] in_weight_above,
    output reg signed [w_bit-1:0] out_weight_below
    );
    
    always @(negedge RESET or posedge CLK )begin
        if(~RESET) begin
            out_sum <= 0;
            active_right <= 0;
            out_weight_below <= 0;
        end else begin
            if(EN) begin
                if(W_EN) begin
                    out_weight_below <= in_weight_above;
                end else begin
                    active_right <= active_left;
                    out_sum <= out_weight_below * active_left + in_sum;
                end
            end
        end
    end

endmodule
