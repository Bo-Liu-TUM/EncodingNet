import os
import re
import torch
import numpy as np
import pandas as pd
import pickle

# def gen_uniform_levels_input_encode(bit, signed=True, norm_to_one=True):
#     l = np.arange(2 ** bit)
#     scale = (2 ** (bit - 1) - 1) if signed else (2 ** bit - 1)
#     l = l - 2 ** (bit - 1) if signed else l
#     # c = bin(ll & (2 ** bit - 1))[2:]  # generate two's complement
#     # t = '0' * (bit - len(c)) + c  # add 0 to the high significant bit
#     code = ['0' * (bit - len(bin(ll & (2 ** bit - 1))[2:])) + bin(ll & (2 ** bit - 1))[2:] for ll in l]
#     # code = []
#     # for ll in l:
#     #     c = bin(ll & (2 ** bit - 1))[2:]  # generate two's complement
#     #     t = '0' * (bit - len(c)) + c  # add 0 to the high significant bit
#     #     code.append(t)
#     l = l / scale if norm_to_one else l
#     return l, code


def gen_uniform_levels_input_encode(bit, signed=True, norm_to_one=True):
    l = torch.arange(2 ** bit)
    scale = (2 ** (bit - 1) - 1) if signed else (2 ** bit - 1)
    l = l - 2 ** (bit - 1) if signed else l
    # c = bin(ll & (2 ** bit - 1))[2:]  # generate two's complement
    # t = '0' * (bit - len(c)) + c  # add 0 to the high significant bit
    code = ['0' * (bit - len(bin(ll & (2 ** bit - 1))[2:])) + bin(ll & (2 ** bit - 1))[2:] for ll in l]
    # code = l.reshape(-1, 1) >> np.arange(bit - 1, -1, -1) & 1
    # code = []
    # for ll in l:
    #     c = bin(ll & (2 ** bit - 1))[2:]  # generate two's complement
    #     t = '0' * (bit - len(c)) + c  # add 0 to the high significant bit
    #     code.append(t)
    l = l / scale if norm_to_one else l
    return l, code


# def gen_input_matrix(code_x, code_w, x, w):
#     input_code_matrix = [[int(i) for i in list(c_x + c_w)] for c_x in code_x for c_w in code_w]
#     input_value_matrix = [[v_x.item(), v_w.item()] for v_x in x for v_w in w]
#     # for c_x, v_x in zip(code_x, x):
#     #     for c_w, v_w in zip(code_w, w):
#     #         input_code_matrix.append([int(i) for i in list(c_x + c_w)])
#     #         input_value_matrix.append([v_x.item(), v_w.item()])
#     input_code_matrix = np.array(input_code_matrix, dtype=bool).T
#     inputs_list = [e for e in input_code_matrix]
#     return inputs_list, input_value_matrix


def gen_input_matrix(code_x, code_w, x, w):
    input_code_matrix = [[int(i) for i in list(c_x + c_w)] for c_x in code_x for c_w in code_w]
    input_value_matrix = [[v_x.item(), v_w.item()] for v_x in x for v_w in w]
    # for c_x, v_x in zip(code_x, x):
    #     for c_w, v_w in zip(code_w, w):
    #         input_code_matrix.append([int(i) for i in list(c_x + c_w)])
    #         input_value_matrix.append([v_x.item(), v_w.item()])
    input_value_matrix = torch.tensor(input_value_matrix, dtype=torch.float32)
    input_code_matrix = torch.tensor(input_code_matrix, dtype=torch.bool).T
    inputs_list = [e for e in input_code_matrix]
    return inputs_list, input_value_matrix


def solve_position_weight(code, value_true):
    position_weight = value_true @ code.T @ torch.linalg.inv(code @ code.T)
    return position_weight


def solve_max_re_rmse(position_weight, code, value_true):
    value_pred = position_weight @ code
    errors = value_pred - value_true
    # maximal relative error
    relative_errors = torch.abs(errors) / max(value_true.max(), -value_true.min())
    maximal_relative_error = relative_errors.max()
    # root mean squared error
    squared_errors = errors ** 2
    root_mean_squared_error = torch.sqrt(torch.mean(squared_errors))
    return value_pred, maximal_relative_error, root_mean_squared_error


def gen_inputs_value(bit=8, signed=True, norm_to_one=False):
    x_bit, w_bit = bit, bit
    x_signed, w_signed, norm_to_one = signed, signed, norm_to_one
    x, code_x = gen_uniform_levels_input_encode(x_bit, signed=x_signed, norm_to_one=norm_to_one)
    w, code_w = gen_uniform_levels_input_encode(w_bit, signed=w_signed, norm_to_one=norm_to_one)
    value = torch.kron(x, w).reshape(1, -1).to(torch.float32)
    inputs, input_value_matrix = gen_input_matrix(code_x, code_w, x, w)
    constant_ones = torch.ones_like(inputs[0])
    return value, inputs, constant_ones


def get_code(running_info, inputs, constant_ones):
    champion = running_info['pop.champion']
    f = champion.to_func()
    outputs = f(*inputs)
    outputs.append(constant_ones)  # add one column of ones as the constant
    outputs = torch.stack(outputs).type(torch.float32)  #
    code = outputs[champion.custom_attr_valid_output_bits]
    return code


def get_1_tcl(hdl_fpath, hdl_fname):
    hdl_path = os.path.join(hdl_fpath, hdl_fname)
    netlish_path = hdl_path[:-2] + '_netlist.v'
    tcl_code = f"#set target_library ./tech_lib/NangateOpenCellLibrary_typical.db\n" \
               f"#set link_library ./tech_lib/NangateOpenCellLibrary_typical.db\n" \
               f"set target_library ./tech_lib/NanGate_15nm_OCL_typical_conditional_nldm.db\n" \
               f"set link_library ./tech_lib/NanGate_15nm_OCL_typical_conditional_nldm.db\n" \
               f"suppress_message UID-401\n" \
               f"read_verilog {hdl_path}\n" \
               f"link\n" \
               f"set timing_enable_through_paths true\n" \
               f"#set timing_through_path_max_segements 10\n" \
               f"link\n" \
               f"check_design\n" \
               f"create_clock CLK -period 1000\n" \
               f"compile_ultra\n" \
               f"optimize_registers\n" \
               f"optimize_netlist -area\n" \
               f"report_timing -delay max\n" \
               f"report_area\n" \
               f"report_power\n" \
               f"\n" \
               f"#write -f verilog -hierarchy -output {netlish_path}\n" \
               f"\n" \
               f"\n"
    return tcl_code


def match(begin, end, string):
    result = re.findall(r'' + begin + '(.+?)' + end + '', string)
    if len(result) == 0:
        raise ValueError('nothing found')
    else:
        return [r.replace(' ', '') for r in result]


def trans_power_unit(unit, power):
    if isinstance(power, str):
        power = float(power)
    if unit == 'uW':
        return str(power)
    elif unit == 'mW':
        return str(power*1000)
    elif unit == 'nW':
        return str(power/1000)
    elif unit == 'pW':
        return str(power/1000000)
    else:
        raise ValueError('unit error')


def extract_report(report):
    Design_names = ['Design'] + match('Design :', '\n', report)[0::3]

    # power
    Dynamic_Power = ['Dynamic Power (uW)'] + match('Total Dynamic Power    =', '\(', report)
    # Dynamic_Power_Unit = ['Dynamic Power Unit']
    for i in range(1, len(Dynamic_Power)):
        unit = Dynamic_Power[i][-2:]
        power = Dynamic_Power[i][0:-2]
        Dynamic_Power[i] = trans_power_unit(unit, power)
        # Dynamic_Power_Unit.append('uW')

    Leakage_Power = ['Leakage Power (uW)'] + match('Cell Leakage Power     =', '\n', report)
    # Leakage_Power_Unit = ['Leakage Power Unit']
    for i in range(1, len(Leakage_Power)):
        unit = Leakage_Power[i][-2:]
        power = Leakage_Power[i][0:-2]
        Leakage_Power[i] = trans_power_unit(unit, power)
        # Leakage_Power_Unit.append('uW')

    # performance
    data_arrival_time = ['data arrival time'] + match('data arrival time', '\n', report)[0::2]
    library_setup_time = ['library setup time'] + match('library setup time', '\n', report)
    for i in range(1, len(library_setup_time)):
        library_setup_time[i] = library_setup_time[i][1:-7]
    clock_CLK = ['clock CLK (ps)'] + match('clock network delay \(ideal\)              0.00', '\n', report)[1::2]

    # area
    Comb_area = ['Combinational area'] + match('Combinational area:', '\n', report)
    Buf_Inv_area = ['Buf/Inv area'] + match('Buf/Inv area:', '\n', report)
    Noncombinational_area = ['Noncombinational area'] + match('Noncombinational area:', '\n', report)
    Total_cell_area = ['Total cell area'] + match('Total cell area:', '\n', report)
    return (Design_names,
            Dynamic_Power,
            # Dynamic_Power_Unit,
            Leakage_Power,
            # Leakage_Power_Unit,
            data_arrival_time, library_setup_time, clock_CLK,
            Comb_area, Buf_Inv_area, Noncombinational_area, Total_cell_area)


def read_area_power():
    # csv_file = './verilog/decoder_multiplier_16rowSA_15bit_run_1GHz.csv'
    csv_file = './verilog/decoder_multiplier_64rowSA_15bit_run_5GHz.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, header=0,
                         usecols=['Total area', 'Total power'])
        df = np.array(df)
        area = df[:, 0]
        power = df[:, 1]
        return area, power
    else:
        print(csv_file, 'open failed')
        exit()


def apply_delta(position_weights, area, delta=0):
    pos_wgts = position_weights.clone()
    if delta != 0:
        for i in range(pos_wgts.numel()):
            idx = pos_wgts[:, i].item() + 16384
            area_ori = area[idx]
            idx = torch.arange(idx - delta, idx + delta + 1)
            idx[idx < 0] = 0
            idx[idx >= len(area)] = len(area) - 1
            min_area = area[idx].min()
            min_p = idx[area[idx].argmin()] - 16384
            pow_2 = []
            for e in (2 ** torch.arange(14)).tolist() + (-2 ** torch.arange(15)).tolist() + [0]:
                if e in idx - 16384:
                    pow_2.append(e)
            if pow_2:
                pow_2_err = torch.tensor(pow_2) - pos_wgts[:, i]
                pos_wgts[:, i] = pow_2[pow_2_err.argmin()]
            elif min_area < area_ori:
                pos_wgts[:, i] = min_p
    return pos_wgts


def normalize(position_weights, bit=15):
    return (position_weights / position_weights.abs().max() * (2 ** (bit - 1) - 1)).round().int()


# read_running_info(path, th, name)
# read_running_info(path, th, rows=256, cols=2, target=64, search)
def read_running_info(path: [str, None] = None,  #
                      th: [float, None] = None,
                      name: [str, None] = None,
                      rows: [int, None] = None, cols: [int, None] = None,
                      target: [int, None] = None, search: [int, None] = None):
    assert path is not None, 'path is None'
    assert th is not None, 'th is None'
    # path='/home/ge26rem/lrz-nashome/LRZ/SourceCode/CGP_search/running_cache/',
    # th = 0.1,
    # name='data-256row-2col-64bit-128b.pickle',
    # rows=256, cols=2, target=64, search=128,
    if name is not None and None in [rows, cols, target, search]:
        by_name = True
    elif name is None and None not in [rows, cols, target, search]:
        by_name = False
    elif name is not None and None not in [rows, cols, target, search]:
        by_name = True
    else:
        raise ValueError("name is None, and None in [rows, cols, target, search]")
    running_path = os.path.join(path, f'th{th}%/')
    file_fullname = name if by_name else f"data-{rows}row-{cols}col-{target}bit-{search}b.pickle"
    file_address = os.path.join(running_path, file_fullname)
    if os.path.exists(file_address) and os.path.isfile(file_address):
        try:
            with open(file_address, 'rb') as f:
                running_info = pickle.load(f)  # load
            return running_info
        except Exception as e:
            print(file_fullname, 'open failed')
            exit()
    else:
        print(file_fullname, 'not exist')
        exit()


def convert_searched_results(path='/home/ge26rem/lrz-nashome/LRZ/SourceCode/CGP_search/running_cache/', th=0.1,
                             target=64, search=128, cols=2, rows=256):
    x_signed, w_signed, norm_to_one = True, True, False
    running_info = read_running_info(path=path, th=th,
                                     rows=rows, cols=cols, target=target, search=search)
    value, inputs, constant_ones = gen_inputs_value(bit=8, signed=True, norm_to_one=False)
    code = get_code(running_info, inputs, constant_ones)
    position_weights = solve_position_weight(code, value)
    value_appr, maximal_relative_error, rmse = solve_max_re_rmse(position_weights, code, value)

    searched_info = []
    scale_x, scale_w = 2 ** (8 - x_signed), 2 ** (8 - w_signed)
    scale_v = scale_x * scale_w
    searched_info.append({
        'quantization type': 'Uniform',
        # 'signed of activation': x_signed, 'signed of weight': w_signed,
        'bit-width of activation': 8, 'bit-width of weight': 8,
        'bit-width of product': position_weights.numel(),
        'root mean square error': rmse / scale_v,
        # 'maximal relative error': maximal_relative_error,
        'digit-weight of product code': position_weights / scale_v,
        # 'value of activation': x / scale_x,
        # 'code of activation': torch.tensor([[int(i) for i in list(c_x)] for c_x in code_x]).bool(),
        # 'value of weight': w / scale_w,
        # 'code of weight': torch.tensor([[int(i) for i in list(c_w)] for c_w in code_w]).bool(),
        # 'input value matrix': input_value_matrix,
        # 'input code of product': inputs,
        'output code of product': code,
        # 'real value of product': value / scale_v,
        'approximate value of product': value_appr / scale_v,
        'method to generate product': 'kronecker product of activation and weight: v=torch.kron(x, w)',
    })
    del value, inputs, constant_ones
    return searched_info


class LogicOP(torch.nn.Module):
    def __init__(self, i: [int, None], j: [int, None], f_str: str):
        super(LogicOP, self).__init__()
        id = [7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0]
        self.i, self.j, self.f_str, self.id = i, j, f_str, id
        if f_str == 'torch.ones(torch.broadcast_shapes(x.shape, w.shape), dtype=torch.bool)':
            self.op = lambda x, w: eval(f_str)
        else:
            self.f = lambda x_0, x_1: eval(f_str)  # f_str = 'torch.logical_not(torch.logical_xor(x_0, x_1))'
            if self.i < 8 <= self.j:
                self.op = lambda x, w: self.f(*torch.broadcast_tensors(x >> id[i] & 1, w >> id[j] & 1))
            elif self.i >= 8 > self.j:
                self.op = lambda x, w: self.f(*torch.broadcast_tensors(w >> id[i] & 1, x >> id[j] & 1))
            elif self.i < 8 and self.j < 8:
                if f_str in ['x_0', 'torch.logical_not(x_0)']:
                    self.op = lambda x, w: self.f(
                        (x >> id[i] & 1).broadcast_to(torch.broadcast_shapes(x.shape, w.shape)), None)
                else:
                    self.op = lambda x, w: self.f(
                        (x >> id[i] & 1).broadcast_to(torch.broadcast_shapes(x.shape, w.shape)),
                        (x >> id[j] & 1).broadcast_to(torch.broadcast_shapes(x.shape, w.shape))
                    )  # *torch.broadcast_tensors(x >> id[i] & 1, x >> id[j] & 1)
            else:
                if f_str in ['x_0', 'torch.logical_not(x_0)']:
                    self.op = lambda x, w: self.f(
                        (w >> id[i] & 1).broadcast_to(torch.broadcast_shapes(x.shape, w.shape)), None)
                else:
                    self.op = lambda x, w: self.f(
                        (w >> id[i] & 1).broadcast_to(torch.broadcast_shapes(x.shape, w.shape)),
                        (w >> id[j] & 1).broadcast_to(torch.broadcast_shapes(x.shape, w.shape))
                    )  # *torch.broadcast_tensors(w >> id[i] & 1, w >> id[j] & 1)

    def forward(self, x, w):
        return self.op(x, w)


def get_OP_OP_info(graph, champion):
    OP, OP_info = None, None
    if graph._n_columns == 1:
        OP = []
        OP_info = []
        for o_i in champion.custom_attr_valid_output_bits:
            if o_i < graph.output_nodes.__len__():
                gi = graph.output_nodes[o_i].addresses[0]
                if gi >= 16:
                    ij = graph.hidden_nodes[gi - 16].addresses
                    if len(ij) == 2:
                        op_i, op_j, op_f_str = ij[0], ij[1], graph.hidden_nodes[gi - 16]._def_output
                        # OP.append(LogicOP(i=ij[0], j=ij[1], f_str=graph.hidden_nodes[gi - 16]._def_output))
                    elif len(ij) == 1:
                        op_i, op_j, op_f_str = ij[0], ij[0], graph.hidden_nodes[gi - 16]._def_output
                        # OP.append(LogicOP(i=ij[0], j=ij[0], f_str=graph.hidden_nodes[gi - 16]._def_output))
                    else:
                        raise ValueError('invalid ij')
                else:
                    op_i, op_j, op_f_str = gi, gi, 'x_0'
                    # OP.append(LogicOP(i=gi, j=gi, f_str='x_0'))
            else:
                op_i, op_j, op_f_str = None, None, 'torch.ones(torch.broadcast_shapes(x.shape, w.shape), dtype=torch.bool)'
                # OP.append(LogicOP(i=None, j=None, f_str='torch.ones(torch.broadcast_shapes(x.shape, w.shape), dtype=torch.bool)'))
            OP.append(LogicOP(i=op_i, j=op_j, f_str=op_f_str))
            OP_info.append({'i': op_i, 'j': op_j, 'f_str': op_f_str})

    return OP, OP_info


def apply_delta_for_finetune(searched_info, delta=0):
    assert len(searched_info) == 1, 'length of searched_info > 1'
    position_weights = searched_info[0]['digit-weight of product code'] * 16384
    max_position_weight = position_weights.abs().max()
    position_weights = normalize(position_weights, bit=15)
    area, _ = read_area_power()
    position_weights_new = apply_delta(position_weights, area, delta)
    searched_info[0][
        'digit-weight of product code'] = position_weights_new * max_position_weight / 16383 / 16384
    searched_info[0]['approximate value of product'] = searched_info[0]['digit-weight of product code'] @ \
                                                       searched_info[0]['output code of product'].cpu()
    return searched_info


def get_approx_product(searched_info, a_bit=8, w_bit=8, product_bit=64, f_info='../backup/searched_info.pickle'):
    def test_fc(approx_product):
        def p(x_l, w_l, x_sign=True, w_sign=True):
            x_c = (x_l * 127).int() & 255 if x_sign else (x_l * 255).int()
            w_c = (w_l * 127).int() & 255 if w_sign else (w_l * 255).int()
            idx = x_c * 256 + w_c
            return approx_product[idx].sum(2)

        def q(x, sign=True):
            m = x.abs().max()
            s = 127 if sign else 255
            return x.div(m).mul(s).round().div(s).mul(m)

        x, w = torch.rand(2, 4) * 2 - 1, torch.rand(3, 4) * 2 - 1
        x, w = x.unsqueeze(1), w.unsqueeze(0)
        fc_real = (x * w).sum(2)
        m_x, m_w = x.abs().max(), w.abs().max()
        x_q, w_q = q(x, sign=True), q(w, sign=True)
        fc_quant = (x_q * w_q).sum(2)
        x_l, w_l = x_q / m_x, w_q / m_w
        fc_approx = p(x_l, w_l, x_sign=True, w_sign=True) * m_x * m_w

        print('real   product: {}\n'
              'quant  product: {}\n'
              'approx product: {}\n'.format(fc_real, fc_quant, fc_approx))

    def test_fc_by_code(approx_product_value_2d, approx_product_code_2d, digit_weight):
        def p(x_l, w_l, x_sign=True, w_sign=True):
            x_c = (x_l * 127).int() + 128 if x_sign else (x_l * 255).int()
            w_c = (w_l * 127).int() + 128 if w_sign else (w_l * 255).int()
            # return approx_product_value_2d[x_c, w_c].sum(2)
            return (approx_product_code_2d[x_c, w_c].sum(2) * digit_weight).sum(2)

        def q(x, sign=True):
            m = x.abs().max()
            s = 127 if sign else 255
            return x.div(m).mul(s).round().div(s).mul(m)

        x, w = torch.rand(2, 4) * 2 - 1, torch.rand(3, 4) * 2 - 1
        x, w = x.unsqueeze(1), w.unsqueeze(0)
        fc_real = (x * w).sum(2)
        m_x, m_w = x.abs().max(), w.abs().max()
        x_q, w_q = q(x, sign=True), q(w, sign=True)
        fc_quant = (x_q * w_q).sum(2)
        x_l, w_l = x_q / m_x, w_q / m_w
        fc_approx = p(x_l, w_l, x_sign=True, w_sign=True) * m_x * m_w

        print('real   product: {}\n'
              'quant  product: {}\n'
              'approx product: {}\n'.format(fc_real, fc_quant, fc_approx))

    def test_conv(approx_product, bias=None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
        import torch.nn.functional as F

        def conv2d(x, w, stride, padding, approx=False, x_sign=True, w_sign=True):
            d, c, k, j = w.shape
            x_pad = F.pad(x, padding * 2, value=0.)
            x_pad = x_pad.unfold(2, k, stride[0])
            x_pad = x_pad.unfold(3, j, stride[1])
            x_pad = x_pad.unsqueeze(1)
            w = w.unsqueeze(2).unsqueeze(2).unsqueeze(0)
            if approx:
                return p(x_pad, w, x_sign=x_sign, w_sign=w_sign)  # .sum(dim=(5,6,2))
            else:
                return x_pad.mul(w).sum(dim=(5, 6, 2))

        def p(x_l, w_l, x_sign=True, w_sign=True):
            x_c = (x_l * 127).int() & 255 if x_sign else (x_l * 255).int()
            w_c = (w_l * 127).int() & 255 if w_sign else (w_l * 255).int()
            idx = x_c * 256 + w_c
            return approx_product[idx].sum(dim=(5, 6, 2))
            # x_c = x_c * 256
            # xw_c = []
            # for di in range(w_c.shape[1]):
            #     xw_c.append(approx_product[w_c[:, di:di + 1, :, :, :, :, :] + x_c].sum(dim=(5, 6, 2)))
            # xw_c = torch.concat(xw_c, dim=1)
            # return xw_c
            # idx = x_c + w_c
            # return approx_product[idx]

        def q(x, sign=True):
            m = x.abs().max()
            s = 127 if sign else 255
            return x.div(m).mul(s).round().div(s).mul(m)

        # bias, stride, padding, dilation, groups = None, (1, 1), (1, 1), (1, 1), 1
        x, w = torch.rand(2, 3, 4, 4), torch.rand(3, 3, 2, 2) * 2 - 1
        m_x, m_w = x.abs().max(), w.abs().max()
        # real
        # F.conv2d(x,w,bias,stride,padding,dilation,groups) == x_pad.mul(w).sum(dim=(5,6,2))
        conv_real = conv2d(x, w, stride, padding)  # .sum(dim=(5,6,2))
        # quant
        x_q, w_q = q(x, sign=True), q(w, sign=True)
        conv_quant = conv2d(x_q, w_q, stride, padding)
        # approx
        x_l, w_l = x_q / m_x, w_q / m_w
        conv_approx = conv2d(x_l, w_l, stride, padding, approx=True, x_sign=True, w_sign=True) * m_x * m_w

        print('real   product: {}\n'
              'quant  product: {}\n'
              'approx product: {}\n'.format(conv_real, conv_quant, conv_approx))

    def test_conv_by_code(approx_product_value_2d, approx_product_code_2d, digit_weight,
                          bias=None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
        import torch.nn.functional as F

        def conv2d(x, w, stride, padding, approx=False, x_sign=True, w_sign=True):
            d, c, k, j = w.shape
            x_pad = F.pad(x, padding * 2, value=0.)
            x_pad = x_pad.unfold(2, k, stride[0])
            x_pad = x_pad.unfold(3, j, stride[1])
            x_pad = x_pad.unsqueeze(1)
            w = w.unsqueeze(2).unsqueeze(2).unsqueeze(0)
            if approx:
                return p(x_pad, w, x_sign=x_sign, w_sign=w_sign)  # .sum(dim=(5,6,2))
            else:
                return x_pad.mul(w).sum(dim=(5, 6, 2))

        def p(x_l, w_l, x_sign=True, w_sign=True):
            x_c = (x_l * 127).int() + 128 if x_sign else (x_l * 255).int()
            w_c = (w_l * 127).int() + 128 if w_sign else (w_l * 255).int()
            # return approx_product_value_2d[x_c, w_c].sum(dim=(5, 6, 2))
            return (approx_product_code_2d[x_c, w_c, :].sum(dim=(5, 6, 2)) * digit_weight).sum(4)
            # xw_c = []
            # for di in range(w_c.shape[1]):
            #     xw_c.append(approx_product[w_c[:, di:di + 1, :, :, :, :, :] + x_c].sum(dim=(5, 6, 2)))
            # xw_c = torch.concat(xw_c, dim=1)
            # return xw_c
            # idx = x_c + w_c
            # return approx_product[idx]

        def q(x, sign=True):
            m = x.abs().max()
            s = 127 if sign else 255
            return x.div(m).mul(s).round().div(s).mul(m)

        # bias, stride, padding, dilation, groups = None, (1, 1), (1, 1), (1, 1), 1
        x, w = torch.rand(2, 3, 4, 4), torch.rand(3, 3, 2, 2) * 2 - 1
        m_x, m_w = x.abs().max(), w.abs().max()
        # real
        # F.conv2d(x,w,bias,stride,padding,dilation,groups) == x_pad.mul(w).sum(dim=(5,6,2))
        conv_real = conv2d(x, w, stride, padding)  # .sum(dim=(5,6,2))
        # quant
        x_q, w_q = q(x, sign=True), q(w, sign=True)
        conv_quant = conv2d(x_q, w_q, stride, padding)
        # approx
        x_l, w_l = x_q / m_x, w_q / m_w
        conv_approx = conv2d(x_l, w_l, stride, padding, approx=True, x_sign=True, w_sign=True) * m_x * m_w

        print('real   product: {}\n'
              'quant  product: {}\n'
              'approx product: {}\n'.format(conv_real, conv_quant, conv_approx))

    approx_product_value_2d, approx_product_code_2d, digit_weight = None, None, None
    if product_bit == 0:
        # return None, 0.0
        return None, None, None, 0.0
    else:
        # import pickle
        # # f_info = '../backup/searched_info.pickle'
        # with open(f_info, 'rb') as f:
        #     searched_info = pickle.load(f)
        # # print(product_bit, type(product_bit))
        for info in searched_info:
            if info['bit-width of product'] == product_bit:
                print('bit-width of product: {}\t'
                      'root mean square error:{:.2e}'.format(info['bit-width of product'],
                                                             info['root mean square error']))
                # (info['code of activation'] * 2 ** torch.arange(0, 8).flip(0).view(1, -1)).sum(1)
                # (info['code of weight'] * 2 ** torch.arange(0, 8).flip(0).view(1, -1)).sum(1)
                assert info['bit-width of weight'] == w_bit
                assert info['bit-width of activation'] == a_bit
                # idx = (info['input code of product'] * 2 ** torch.arange(0, 2 * bit).flip(0).view(-1, 1)).sum(0)
                # approx_product = torch.zeros_like(idx).float()
                # approx_product[idx] = info['approximate value of product']
                approx_product_code_2d = info['output code of product'].view(product_bit, 256, 256).permute(1, 2, 0)
                approx_product_value_2d = info['approximate value of product'].view(256, 256)
                digit_weight = info['digit-weight of product code']
                # approx_product[idx] = info['real value of product']
                # test_fc(approx_product)
                # test_conv(approx_product)
                # test_fc_by_code(approx_product_value_2d, approx_product_code_2d, digit_weight)
                # test_conv_by_code(approx_product_value_2d, approx_product_code_2d, digit_weight)
                # return approx_product, info['root mean square error']
                return approx_product_value_2d, approx_product_code_2d, digit_weight, info['root mean square error']
        if None in (approx_product_value_2d, approx_product_code_2d, digit_weight):
            raise ValueError('did not find product_bit={} in {}'.format(product_bit, f_info))



def main():
    pass


if __name__ == '__main__':
    main()
