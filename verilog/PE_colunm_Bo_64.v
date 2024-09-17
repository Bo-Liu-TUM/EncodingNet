////////////////////////////////////////////////////////////////////////
// Created by: Bo Liu, Chair of Electronic Design Automation, TUM
// Version   : v1.0
// Date      : Tue May  7 17:52:19 2024
////////////////////////////////////////////////////////////////////////


module PE_column_Bo_64 # (
parameter a_bit = 8,
parameter w_bit = 8
)(
input CLK,
input RESET,
input EN,
input load_EN,
input [6 -1 : 0] address,
input [64*a_bit -1 : 0] a,
input [w_bit -1 : 0] w,
output reg [64*a_bit -1 : 0] a_right,
output reg [22-1 : 0] decoder_out
);


//reg [a_bit -1] a_reg [64-1 : 0];
reg  [64*w_bit -1 : 0] w_reg;
wire [64*64 -1 : 0] p;
wire [64*7-1 : 0] s_wire;
reg [64*7-1 : 0] s_reg;
wire [22-1 : 0] decoder_out_wire;


// searched approximate multiplier
generate
genvar gi;
for(gi=0; gi<64; gi=gi+1)begin:gen_muls
	MUL_Bo_64bit MUL_Bo_U (
		.x(    a[(gi+1)*a_bit-1 : gi*a_bit]),
		.w(w_reg[(gi+1)*w_bit-1 : gi*w_bit]),
		.c(    p[(gi+1)*64-1 : gi*64])
	);
end
endgenerate


// bit-wise accumulation
generate
genvar gj;
for(gj=0; gj<64; gj=gj+1)begin:gen_adds
	assign s_wire[(gj+1)*7-1 : gj*7] = p[gj+64*0] + 
		p[gj+64*1] + 
		p[gj+64*2] + 
		p[gj+64*3] + 
		p[gj+64*4] + 
		p[gj+64*5] + 
		p[gj+64*6] + 
		p[gj+64*7] + 
		p[gj+64*8] + 
		p[gj+64*9] + 
		p[gj+64*10] + 
		p[gj+64*11] + 
		p[gj+64*12] + 
		p[gj+64*13] + 
		p[gj+64*14] + 
		p[gj+64*15] + 
		p[gj+64*16] + 
		p[gj+64*17] + 
		p[gj+64*18] + 
		p[gj+64*19] + 
		p[gj+64*20] + 
		p[gj+64*21] + 
		p[gj+64*22] + 
		p[gj+64*23] + 
		p[gj+64*24] + 
		p[gj+64*25] + 
		p[gj+64*26] + 
		p[gj+64*27] + 
		p[gj+64*28] + 
		p[gj+64*29] + 
		p[gj+64*30] + 
		p[gj+64*31] + 
		p[gj+64*32] + 
		p[gj+64*33] + 
		p[gj+64*34] + 
		p[gj+64*35] + 
		p[gj+64*36] + 
		p[gj+64*37] + 
		p[gj+64*38] + 
		p[gj+64*39] + 
		p[gj+64*40] + 
		p[gj+64*41] + 
		p[gj+64*42] + 
		p[gj+64*43] + 
		p[gj+64*44] + 
		p[gj+64*45] + 
		p[gj+64*46] + 
		p[gj+64*47] + 
		p[gj+64*48] + 
		p[gj+64*49] + 
		p[gj+64*50] + 
		p[gj+64*51] + 
		p[gj+64*52] + 
		p[gj+64*53] + 
		p[gj+64*54] + 
		p[gj+64*55] + 
		p[gj+64*56] + 
		p[gj+64*57] + 
		p[gj+64*58] + 
		p[gj+64*59] + 
		p[gj+64*60] + 
		p[gj+64*61] + 
		p[gj+64*62] + 
		p[gj+64*63];
end
endgenerate


// define position weights, minor change = ±5
localparam Pos_Wt0 = -15'd16384;
localparam Pos_Wt1 =  15'd9424;
localparam Pos_Wt2 =  15'd9344;
localparam Pos_Wt3 =  15'd8192;
localparam Pos_Wt4 =  15'd8192;
localparam Pos_Wt5 = -15'd6560;
localparam Pos_Wt6 = -15'd4800;
localparam Pos_Wt7 =  15'd4096;
localparam Pos_Wt8 = -15'd3583;
localparam Pos_Wt9 = -15'd2464;
localparam Pos_Wt10 =  15'd2312;
localparam Pos_Wt11 = -15'd2048;
localparam Pos_Wt12 = -15'd2048;
localparam Pos_Wt13 = -15'd2048;
localparam Pos_Wt14 = -15'd1024;
localparam Pos_Wt15 = -15'd1024;
localparam Pos_Wt16 = -15'd1024;
localparam Pos_Wt17 =  15'd1024;
localparam Pos_Wt18 =  15'd1024;
localparam Pos_Wt19 = -15'd639;
localparam Pos_Wt20 = -15'd512;
localparam Pos_Wt21 =  15'd512;
localparam Pos_Wt22 =  15'd512;
localparam Pos_Wt23 = -15'd512;
localparam Pos_Wt24 = -15'd512;
localparam Pos_Wt25 =  15'd512;
localparam Pos_Wt26 = -15'd512;
localparam Pos_Wt27 =  15'd256;
localparam Pos_Wt28 = -15'd256;
localparam Pos_Wt29 = -15'd256;
localparam Pos_Wt30 = -15'd256;
localparam Pos_Wt31 =  15'd256;
localparam Pos_Wt32 =  15'd256;
localparam Pos_Wt33 =  15'd256;
localparam Pos_Wt34 =  15'd256;
localparam Pos_Wt35 = -15'd256;
localparam Pos_Wt36 =  15'd161;
localparam Pos_Wt37 = -15'd128;
localparam Pos_Wt38 = -15'd128;
localparam Pos_Wt39 =  15'd128;
localparam Pos_Wt40 =  15'd128;
localparam Pos_Wt41 =  15'd128;
localparam Pos_Wt42 = -15'd128;
localparam Pos_Wt43 =  15'd128;
localparam Pos_Wt44 =  15'd128;
localparam Pos_Wt45 = -15'd64;
localparam Pos_Wt46 = -15'd64;
localparam Pos_Wt47 = -15'd64;
localparam Pos_Wt48 = -15'd64;
localparam Pos_Wt49 = -15'd64;
localparam Pos_Wt50 = -15'd64;
localparam Pos_Wt51 =  15'd56;
localparam Pos_Wt52 = -15'd32;
localparam Pos_Wt53 = -15'd32;
localparam Pos_Wt54 = -15'd32;
localparam Pos_Wt55 = -15'd32;
localparam Pos_Wt56 = -15'd32;
localparam Pos_Wt57 =  15'd32;
localparam Pos_Wt58 = -15'd32;
localparam Pos_Wt59 = -15'd16;
localparam Pos_Wt60 = -15'd16;
localparam Pos_Wt61 = -15'd16;
localparam Pos_Wt62 = -15'd16;
localparam Pos_Wt63 =  15'd16;


// decoder: multiply position weights, then accumulate
assign decoder_out_wire = s_reg[(0+1)*7-1 : 0*7] * Pos_Wt0 + 
	s_reg[(1+1)*7-1 : 1*7] * Pos_Wt1 + 
	s_reg[(2+1)*7-1 : 2*7] * Pos_Wt2 + 
	s_reg[(3+1)*7-1 : 3*7] * Pos_Wt3 + 
	s_reg[(4+1)*7-1 : 4*7] * Pos_Wt4 + 
	s_reg[(5+1)*7-1 : 5*7] * Pos_Wt5 + 
	s_reg[(6+1)*7-1 : 6*7] * Pos_Wt6 + 
	s_reg[(7+1)*7-1 : 7*7] * Pos_Wt7 + 
	s_reg[(8+1)*7-1 : 8*7] * Pos_Wt8 + 
	s_reg[(9+1)*7-1 : 9*7] * Pos_Wt9 + 
	s_reg[(10+1)*7-1 : 10*7] * Pos_Wt10 + 
	s_reg[(11+1)*7-1 : 11*7] * Pos_Wt11 + 
	s_reg[(12+1)*7-1 : 12*7] * Pos_Wt12 + 
	s_reg[(13+1)*7-1 : 13*7] * Pos_Wt13 + 
	s_reg[(14+1)*7-1 : 14*7] * Pos_Wt14 + 
	s_reg[(15+1)*7-1 : 15*7] * Pos_Wt15 + 
	s_reg[(16+1)*7-1 : 16*7] * Pos_Wt16 + 
	s_reg[(17+1)*7-1 : 17*7] * Pos_Wt17 + 
	s_reg[(18+1)*7-1 : 18*7] * Pos_Wt18 + 
	s_reg[(19+1)*7-1 : 19*7] * Pos_Wt19 + 
	s_reg[(20+1)*7-1 : 20*7] * Pos_Wt20 + 
	s_reg[(21+1)*7-1 : 21*7] * Pos_Wt21 + 
	s_reg[(22+1)*7-1 : 22*7] * Pos_Wt22 + 
	s_reg[(23+1)*7-1 : 23*7] * Pos_Wt23 + 
	s_reg[(24+1)*7-1 : 24*7] * Pos_Wt24 + 
	s_reg[(25+1)*7-1 : 25*7] * Pos_Wt25 + 
	s_reg[(26+1)*7-1 : 26*7] * Pos_Wt26 + 
	s_reg[(27+1)*7-1 : 27*7] * Pos_Wt27 + 
	s_reg[(28+1)*7-1 : 28*7] * Pos_Wt28 + 
	s_reg[(29+1)*7-1 : 29*7] * Pos_Wt29 + 
	s_reg[(30+1)*7-1 : 30*7] * Pos_Wt30 + 
	s_reg[(31+1)*7-1 : 31*7] * Pos_Wt31 + 
	s_reg[(32+1)*7-1 : 32*7] * Pos_Wt32 + 
	s_reg[(33+1)*7-1 : 33*7] * Pos_Wt33 + 
	s_reg[(34+1)*7-1 : 34*7] * Pos_Wt34 + 
	s_reg[(35+1)*7-1 : 35*7] * Pos_Wt35 + 
	s_reg[(36+1)*7-1 : 36*7] * Pos_Wt36 + 
	s_reg[(37+1)*7-1 : 37*7] * Pos_Wt37 + 
	s_reg[(38+1)*7-1 : 38*7] * Pos_Wt38 + 
	s_reg[(39+1)*7-1 : 39*7] * Pos_Wt39 + 
	s_reg[(40+1)*7-1 : 40*7] * Pos_Wt40 + 
	s_reg[(41+1)*7-1 : 41*7] * Pos_Wt41 + 
	s_reg[(42+1)*7-1 : 42*7] * Pos_Wt42 + 
	s_reg[(43+1)*7-1 : 43*7] * Pos_Wt43 + 
	s_reg[(44+1)*7-1 : 44*7] * Pos_Wt44 + 
	s_reg[(45+1)*7-1 : 45*7] * Pos_Wt45 + 
	s_reg[(46+1)*7-1 : 46*7] * Pos_Wt46 + 
	s_reg[(47+1)*7-1 : 47*7] * Pos_Wt47 + 
	s_reg[(48+1)*7-1 : 48*7] * Pos_Wt48 + 
	s_reg[(49+1)*7-1 : 49*7] * Pos_Wt49 + 
	s_reg[(50+1)*7-1 : 50*7] * Pos_Wt50 + 
	s_reg[(51+1)*7-1 : 51*7] * Pos_Wt51 + 
	s_reg[(52+1)*7-1 : 52*7] * Pos_Wt52 + 
	s_reg[(53+1)*7-1 : 53*7] * Pos_Wt53 + 
	s_reg[(54+1)*7-1 : 54*7] * Pos_Wt54 + 
	s_reg[(55+1)*7-1 : 55*7] * Pos_Wt55 + 
	s_reg[(56+1)*7-1 : 56*7] * Pos_Wt56 + 
	s_reg[(57+1)*7-1 : 57*7] * Pos_Wt57 + 
	s_reg[(58+1)*7-1 : 58*7] * Pos_Wt58 + 
	s_reg[(59+1)*7-1 : 59*7] * Pos_Wt59 + 
	s_reg[(60+1)*7-1 : 60*7] * Pos_Wt60 + 
	s_reg[(61+1)*7-1 : 61*7] * Pos_Wt61 + 
	s_reg[(62+1)*7-1 : 62*7] * Pos_Wt62 + 
	s_reg[(63+1)*7-1 : 63*7] * Pos_Wt63;


// main sequential logic
always @(negedge RESET or posedge CLK )begin
	if(~RESET) begin
		w_reg <= 0;
		s_reg <= 0;
		decoder_out <= 0;
		a_right <= 0;
	end else begin
		if(EN) begin
			if(load_EN) begin
				// load weights
				case(address)
					2'b000000: w_reg[(0+1)*w_bit-1 : 0*w_bit] <= w;
					2'b000001: w_reg[(1+1)*w_bit-1 : 1*w_bit] <= w;
					2'b000010: w_reg[(2+1)*w_bit-1 : 2*w_bit] <= w;
					2'b000011: w_reg[(3+1)*w_bit-1 : 3*w_bit] <= w;
					2'b000100: w_reg[(4+1)*w_bit-1 : 4*w_bit] <= w;
					2'b000101: w_reg[(5+1)*w_bit-1 : 5*w_bit] <= w;
					2'b000110: w_reg[(6+1)*w_bit-1 : 6*w_bit] <= w;
					2'b000111: w_reg[(7+1)*w_bit-1 : 7*w_bit] <= w;
					2'b001000: w_reg[(8+1)*w_bit-1 : 8*w_bit] <= w;
					2'b001001: w_reg[(9+1)*w_bit-1 : 9*w_bit] <= w;
					2'b001010: w_reg[(10+1)*w_bit-1 : 10*w_bit] <= w;
					2'b001011: w_reg[(11+1)*w_bit-1 : 11*w_bit] <= w;
					2'b001100: w_reg[(12+1)*w_bit-1 : 12*w_bit] <= w;
					2'b001101: w_reg[(13+1)*w_bit-1 : 13*w_bit] <= w;
					2'b001110: w_reg[(14+1)*w_bit-1 : 14*w_bit] <= w;
					2'b001111: w_reg[(15+1)*w_bit-1 : 15*w_bit] <= w;
					2'b010000: w_reg[(16+1)*w_bit-1 : 16*w_bit] <= w;
					2'b010001: w_reg[(17+1)*w_bit-1 : 17*w_bit] <= w;
					2'b010010: w_reg[(18+1)*w_bit-1 : 18*w_bit] <= w;
					2'b010011: w_reg[(19+1)*w_bit-1 : 19*w_bit] <= w;
					2'b010100: w_reg[(20+1)*w_bit-1 : 20*w_bit] <= w;
					2'b010101: w_reg[(21+1)*w_bit-1 : 21*w_bit] <= w;
					2'b010110: w_reg[(22+1)*w_bit-1 : 22*w_bit] <= w;
					2'b010111: w_reg[(23+1)*w_bit-1 : 23*w_bit] <= w;
					2'b011000: w_reg[(24+1)*w_bit-1 : 24*w_bit] <= w;
					2'b011001: w_reg[(25+1)*w_bit-1 : 25*w_bit] <= w;
					2'b011010: w_reg[(26+1)*w_bit-1 : 26*w_bit] <= w;
					2'b011011: w_reg[(27+1)*w_bit-1 : 27*w_bit] <= w;
					2'b011100: w_reg[(28+1)*w_bit-1 : 28*w_bit] <= w;
					2'b011101: w_reg[(29+1)*w_bit-1 : 29*w_bit] <= w;
					2'b011110: w_reg[(30+1)*w_bit-1 : 30*w_bit] <= w;
					2'b011111: w_reg[(31+1)*w_bit-1 : 31*w_bit] <= w;
					2'b100000: w_reg[(32+1)*w_bit-1 : 32*w_bit] <= w;
					2'b100001: w_reg[(33+1)*w_bit-1 : 33*w_bit] <= w;
					2'b100010: w_reg[(34+1)*w_bit-1 : 34*w_bit] <= w;
					2'b100011: w_reg[(35+1)*w_bit-1 : 35*w_bit] <= w;
					2'b100100: w_reg[(36+1)*w_bit-1 : 36*w_bit] <= w;
					2'b100101: w_reg[(37+1)*w_bit-1 : 37*w_bit] <= w;
					2'b100110: w_reg[(38+1)*w_bit-1 : 38*w_bit] <= w;
					2'b100111: w_reg[(39+1)*w_bit-1 : 39*w_bit] <= w;
					2'b101000: w_reg[(40+1)*w_bit-1 : 40*w_bit] <= w;
					2'b101001: w_reg[(41+1)*w_bit-1 : 41*w_bit] <= w;
					2'b101010: w_reg[(42+1)*w_bit-1 : 42*w_bit] <= w;
					2'b101011: w_reg[(43+1)*w_bit-1 : 43*w_bit] <= w;
					2'b101100: w_reg[(44+1)*w_bit-1 : 44*w_bit] <= w;
					2'b101101: w_reg[(45+1)*w_bit-1 : 45*w_bit] <= w;
					2'b101110: w_reg[(46+1)*w_bit-1 : 46*w_bit] <= w;
					2'b101111: w_reg[(47+1)*w_bit-1 : 47*w_bit] <= w;
					2'b110000: w_reg[(48+1)*w_bit-1 : 48*w_bit] <= w;
					2'b110001: w_reg[(49+1)*w_bit-1 : 49*w_bit] <= w;
					2'b110010: w_reg[(50+1)*w_bit-1 : 50*w_bit] <= w;
					2'b110011: w_reg[(51+1)*w_bit-1 : 51*w_bit] <= w;
					2'b110100: w_reg[(52+1)*w_bit-1 : 52*w_bit] <= w;
					2'b110101: w_reg[(53+1)*w_bit-1 : 53*w_bit] <= w;
					2'b110110: w_reg[(54+1)*w_bit-1 : 54*w_bit] <= w;
					2'b110111: w_reg[(55+1)*w_bit-1 : 55*w_bit] <= w;
					2'b111000: w_reg[(56+1)*w_bit-1 : 56*w_bit] <= w;
					2'b111001: w_reg[(57+1)*w_bit-1 : 57*w_bit] <= w;
					2'b111010: w_reg[(58+1)*w_bit-1 : 58*w_bit] <= w;
					2'b111011: w_reg[(59+1)*w_bit-1 : 59*w_bit] <= w;
					2'b111100: w_reg[(60+1)*w_bit-1 : 60*w_bit] <= w;
					2'b111101: w_reg[(61+1)*w_bit-1 : 61*w_bit] <= w;
					2'b111110: w_reg[(62+1)*w_bit-1 : 62*w_bit] <= w;
					2'b111111: w_reg[(63+1)*w_bit-1 : 63*w_bit] <= w;
				endcase
			end else begin
				s_reg <= s_wire;
				decoder_out <= decoder_out_wire;
				a_right <= a;
			end
		end
	end
end


endmodule

////////////////////////////////////////////////////////////////////////
// Created by: Bo Liu, Chair of Electronic Design Automation, TUM
// Version   : v1.0
// Date      : Tue May  7 17:52:19 2024
////////////////////////////////////////////////////////////////////////


module MUL_Bo_64bit(x, w, c);
input  [7:0]  x;
input  [7:0]  w;
output [63:0] c;
// c = x * w


wire n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, 
	n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, 
	n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, 
	n30, n31, n32, n33, n34, n35, n36, n37, n38, n39, 
	n40, n41, n42, n43, n44, n45, n46, n47, n48, n49, 
	n50, n51, n52, n53, n54, n55, n56, n57, n58, n59, 
	n60, n61, n62, n63, n64, n65, n66, n67, n68, n69, 
	n70, n71, n72, n73, n74, n75, n76, n77, n78, n79, 
	n80, n81, n82, n83, n84, n85, n86, n87, n88, n89, 
	n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, 
	n100, n101, n102, n103, n104, n105, n106, n107, n108, n109, 
	n110, n111, n112, n113, n114, n115, n116, n117, n118, n119, 
	n120, n121, n122, n123, n124, n125, n126, n127, n128, n129, 
	n130, n131, n132, n133, n134, n135, n136, n137, n138, n139, 
	n140, n141, n142, n143, n144, n145, n146, n147, n148, n149, 
	n150, n151, n152, n153, n154, n155, n156, n157, n158, n159, 
	n160, n161, n162, n163, n164, n165, n166, n167, n168, n169, 
	n170, n171, n172, n173, n174, n175, n176, n177, n178, n179, 
	n180, n181, n182, n183, n184, n185, n186, n187, n188, n189, 
	n190, n191, n192, n193, n194, n195, n196, n197, n198, n199, 
	n200, n201, n202, n203, n204, n205, n206, n207, n208, n209, 
	n210, n211, n212, n213, n214, n215, n216, n217, n218, n219, 
	n220, n221, n222, n223, n224, n225, n226, n227, n228, n229, 
	n230, n231, n232, n233, n234, n235, n236, n237, n238, n239, 
	n240, n241, n242, n243, n244, n245, n246, n247, n248, n249, 
	n250, n251, n252, n253, n254, n255, n256, n257, n258, n259, 
	n260, n261, n262, n263, n264, n265, n266, n267, n268, n269, 
	n270, n271, n272, n273, n274, n275, n276, n277, n278, n279, 
	n280, n281, n282, n283, n284, n285, n286, n287, n288, n289, 
	n290, n291, n292, n293, n294, n295, n296, n297, n298, n299, 
	n300, n301, n302, n303, n304, n305, n306, n307, n308, n309, 
	n310, n311, n312, n313, n314, n315, n316, n317, n318, n319, 
	n320, n321, n322, n323, n324, n325, n326, n327, n328, n329, 
	n330, n331, n332, n333, n334, n335, n336, n337, n338, n339, 
	n340, n341, n342, n343, n344, n345, n346, n347, n348, n349, 
	n350, n351, n352, n353, n354, n355, n356, n357, n358, n359, 
	n360, n361, n362, n363, n364, n365, n366, n367, n368, n369, 
	n370, n371, n372, n373, n374, n375, n376, n377, n378, n379, 
	n380, n381, n382, n383, n384, n385, n386, n387, n388, n389, 
	n390, n391, n392, n393, n394, n395, n396, n397, n398, n399, 
	n400, n401, n402, n403, n404, n405, n406, n407, n408, n409, 
	n410, n411, n412, n413, n414, n415, n416, n417, n418, n419, 
	n420, n421, n422, n423, n424, n425, n426, n427, n428, n429, 
	n430, n431, n432, n433, n434, n435, n436, n437, n438, n439, 
	n440, n441, n442, n443, n444, n445, n446, n447, n448, n449, 
	n450, n451, n452, n453, n454, n455, n456, n457, n458, n459, 
	n460, n461, n462, n463, n464, n465, n466, n467, n468, n469, 
	n470, n471, n472, n473, n474, n475, n476, n477, n478, n479, 
	n480, n481, n482, n483, n484, n485, n486, n487, n488, n489, 
	n490, n491, n492, n493, n494, n495, n496, n497, n498, n499, 
	n500, n501, n502, n503, n504, n505, n506, n507, n508, n509, 
	n510, n511;


// hidden nodes
assign n0 = ~(x[5] & w[1]);
assign n1 = w[7];
assign n2 = ~(w[7] & w[6]);
assign n6 = ~x[4];
assign n8 = ~(x[6] | w[1]);
assign n10 = ~(w[5] ^ w[0]);
assign n11 = ~(x[4] | w[7]);
assign n14 = ~x[1];
assign n15 = ~(w[0] ^ x[6]);
assign n17 = ~(x[4] | w[4]);
assign n22 = ~(w[0] | x[4]);
assign n27 = ~w[7];
assign n30 = ~(w[1] & w[6]);
assign n32 = w[4] & w[4];
assign n33 = ~(x[4] | x[3]);
assign n34 = ~(w[3] & x[6]);
assign n36 = x[1] ^ w[7];
assign n41 = ~(w[6] | x[1]);
assign n44 = ~(x[3] & w[2]);
assign n45 = w[3] | x[6];
assign n46 = ~(x[5] & w[2]);
assign n52 = ~(x[1] & w[4]);
assign n55 = w[3] & w[1];
assign n56 = ~(w[6] | x[5]);
assign n58 = ~(x[1] & w[3]);
assign n61 = ~(x[6] | w[2]);
assign n65 = x[6] ^ x[5];
assign n66 = ~(x[5] | w[3]);
assign n67 = ~(x[1] & w[5]);
assign n68 = ~w[0];
assign n69 = x[0] ^ w[6];
assign n71 = ~(x[4] | w[4]);
assign n72 = ~(w[5] | x[4]);
assign n73 = x[7];
assign n76 = ~(w[4] | x[7]);
assign n78 = w[6] & w[3];
assign n79 = ~(x[3] & w[3]);
assign n81 = w[6] ^ x[5];
assign n83 = ~(x[2] ^ x[6]);
assign n85 = ~(x[2] & w[7]);
assign n88 = ~(x[6] & w[7]);
assign n95 = x[1] ^ w[6];
assign n96 = w[5] ^ x[6];
assign n103 = w[2];
assign n106 = w[0] | w[4];
assign n108 = w[4] | x[4];
assign n110 = ~(x[3] & w[1]);
assign n112 = ~(w[3] | x[7]);
assign n113 = w[3] & x[7];
assign n118 = w[5];
assign n119 = ~(w[1] ^ x[2]);
assign n122 = ~(x[3] | w[5]);
assign n124 = ~(w[2] | x[7]);
assign n126 = ~w[6];
assign n129 = ~(w[3] & x[4]);
assign n132 = x[4];
assign n133 = w[3] & w[1];
assign n134 = ~(w[6] & x[6]);
assign n138 = ~(x[2] & w[6]);
assign n139 = x[0] ^ x[3];
assign n140 = x[0];
assign n141 = w[2];
assign n145 = ~(w[4] & x[2]);
assign n147 = ~(w[5] & w[4]);
assign n148 = ~(x[3] ^ x[4]);
assign n150 = ~(x[1] | w[6]);
assign n151 = x[6];
assign n152 = w[3] & x[3];
assign n153 = ~(x[0] ^ w[7]);
assign n156 = ~(x[7] | w[7]);
assign n159 = x[7];
assign n160 = w[3] | w[5];
assign n163 = x[0] | w[4];
assign n165 = w[0] ^ x[6];
assign n167 = w[7];
assign n168 = w[4];
assign n169 = ~w[2];
assign n170 = ~(x[4] | w[1]);
assign n171 = ~(x[6] & x[6]);
assign n174 = ~(x[1] ^ w[0]);
assign n175 = x[7] & w[5];
assign n183 = x[7];
assign n186 = ~(w[6] & w[7]);
assign n187 = ~(w[6] & w[5]);
assign n189 = w[0];
assign n190 = x[3];
assign n191 = ~(w[0] & x[5]);
assign n193 = ~(x[3] & w[4]);
assign n194 = ~(x[7] & w[6]);
assign n195 = ~(x[2] & w[2]);
assign n198 = x[0] ^ w[5];
assign n199 = ~(w[0] & x[3]);
assign n201 = x[6] | x[3];
assign n204 = ~(x[7] & w[0]);
assign n205 = ~(w[2] & x[4]);
assign n207 = ~(w[4] | w[3]);
assign n209 = x[4];
assign n210 = x[3] & x[3];
assign n213 = ~(x[4] | w[3]);
assign n216 = ~(w[3] & w[7]);
assign n220 = ~(x[2] & w[3]);
assign n222 = ~(x[4] | w[6]);
assign n224 = ~(x[0] ^ x[3]);
assign n225 = x[5] ^ w[5];
assign n227 = ~(x[2] & w[5]);
assign n228 = x[3] ^ w[3];
assign n230 = ~(w[1] & x[7]);
assign n231 = ~w[6];
assign n235 = ~w[7];
assign n236 = ~(w[7] | x[4]);
assign n238 = ~(w[7] & x[7]);
assign n239 = w[4] & w[1];
assign n240 = ~(x[4] ^ w[6]);
assign n242 = x[3];
assign n243 = ~(w[7] ^ x[5]);
assign n245 = w[4] ^ x[5];
assign n246 = ~(w[6] | x[6]);
assign n247 = x[3] ^ w[6];
assign n249 = ~(w[4] | x[6]);
assign n254 = w[4] | w[2];
assign n257 = n56;
assign n265 = n156 | n213;
assign n268 = ~(n83 & n152);
assign n276 = ~(n240 | n235);
assign n278 = n163 | n11;
assign n304 = ~(n69 & n30);
assign n305 = n106 & w[4];
assign n307 = ~(n242 | n239);
assign n310 = ~(n175 | n71);
assign n328 = ~(n199 | n210);
assign n331 = n170;
assign n334 = n138;
assign n337 = n224 | n44;
assign n340 = ~(n73 ^ n201);
assign n341 = ~(n129 | n15);
assign n355 = n96;
assign n373 = ~(n160 | n169);
assign n375 = ~(n113 & n118);
assign n389 = ~(n201 ^ n46);
assign n410 = ~(n69 & n10);
assign n412 = ~(n228 | n55);
assign n417 = n41 & n95;
assign n421 = ~(n118 & n159);
assign n429 = n46;
assign n433 = n67;
assign n437 = ~(n32 | n27);
assign n438 = ~(w[6] | n14);
assign n442 = ~(n78 & n113);
assign n444 = n119 ^ n207;
assign n446 = ~(n183 ^ n239);
assign n457 = ~(n134 & n216);
assign n458 = n126 & n239;
assign n463 = n145;
assign n466 = n1 ^ n190;
assign n470 = ~(n132 & n32);
assign n473 = w[3];
assign n475 = ~(n103 | n231);
assign n478 = ~n108;
assign n493 = ~(n147 ^ n78);
assign n496 = n85;
assign n498 = n249;
assign n506 = n148 & n171;


// output nodes
assign c[0] = n238; // -16384
assign c[1] = n6; // 9424
assign c[2] = n209; // 9344
assign c[3] = n88; // 8192
assign c[4] = n194; // 8192
assign c[5] = n134; // -6560
assign c[6] = n167; // -4800
assign c[7] = n421; // 4096
assign c[8] = n73; // -3583
assign c[9] = n246; // -2464
assign c[10] = w[5]; // 2312
assign c[11] = n243; // -2048
assign c[12] = n76; // -2048
assign c[13] = n236; // -2048
assign c[14] = n81; // -1024
assign c[15] = n96; // -1024
assign c[16] = n112; // -1024
assign c[17] = n498; // 1024
assign c[18] = n222; // 1024
assign c[19] = n473; // -639
assign c[20] = n225; // -512
assign c[21] = n466; // 512
assign c[22] = n72; // 512
assign c[23] = n124; // -512
assign c[24] = n34; // -512
assign c[25] = n496; // 512
assign c[26] = n168; // -512
assign c[27] = n17; // 256
assign c[28] = n334; // -256
assign c[29] = n245; // -256
assign c[30] = n247; // -256
assign c[31] = n66; // 256
assign c[32] = n230; // 256
assign c[33] = n61; // 256
assign c[34] = n122; // 256
assign c[35] = n141; // -256
assign c[36] = w[1]; // 161
assign c[37] = n227; // -128
assign c[38] = n193; // -128
assign c[39] = n36; // 128
assign c[40] = n204; // 128
assign c[41] = n8; // 128
assign c[42] = n429; // -128
assign c[43] = n213; // 128
assign c[44] = n150; // 128
assign c[45] = n433; // -64
assign c[46] = n463; // -64
assign c[47] = n79; // -64
assign c[48] = n0; // -64
assign c[49] = n205; // -64
assign c[50] = n153; // -64
assign c[51] = n189; // 56
assign c[52] = n52; // -32
assign c[53] = n220; // -32
assign c[54] = n44; // -32
assign c[55] = n69; // -32
assign c[56] = n165; // -32
assign c[57] = n331; // 32
assign c[58] = n191; // -32
assign c[59] = n58; // -16
assign c[60] = n195; // -16
assign c[61] = n110; // -16
assign c[62] = n198; // -16
assign c[63] = n22; // 16


endmodule

