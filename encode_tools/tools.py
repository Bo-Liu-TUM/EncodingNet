import os
import re
import torch
import numpy as np
import pandas as pd
import pickle


def gen_uniform_levels_input_encode(bit, signed=True, norm_to_one=True):
    l = torch.arange(2 ** bit)
    scale = (2 ** (bit - 1) - 1) if signed else (2 ** bit - 1)
    l = l - 2 ** (bit - 1) if signed else l
    code = ['0' * (bit - len(bin(ll & (2 ** bit - 1))[2:])) + bin(ll & (2 ** bit - 1))[2:] for ll in l]
    l = l / scale if norm_to_one else l
    return l, code


def gen_input_matrix(code_x, code_w, x, w):
    input_code_matrix = [[int(i) for i in list(c_x + c_w)] for c_x in code_x for c_w in code_w]
    input_value_matrix = [[v_x.item(), v_w.item()] for v_x in x for v_w in w]
    input_value_matrix = torch.tensor(input_value_matrix, dtype=torch.float32)
    input_code_matrix = torch.tensor(input_code_matrix, dtype=torch.bool).T
    inputs_list = [e for e in input_code_matrix]
    return inputs_list, input_value_matrix


def solve_position_weight(code, value_true, alpha=0.):
    position_weight = value_true @ code.T @ torch.linalg.inv(code @ code.T + alpha * torch.eye(code.shape[0]).to(code.device))
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


# read_running_info(path, th, name), name is generated by get_file_name() + '.pickle'
def read_running_info(path: [str, None] = None,  #
                      th: [float, None] = None,
                      name: [str, None] = None):
    assert path is not None, 'path is None'
    assert th is not None, 'th is None'
    running_path = os.path.join(path, f'th{th}%/')
    file_fullname = name
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
                             filename='', mode='Exact-INT'):
    if mode == 'FP32' or mode == 'Exact-INT':
        return None
    else:
        x_signed, w_signed, norm_to_one = True, True, False
        running_info = read_running_info(path=path, th=th, name=filename)
        value, inputs, constant_ones = gen_inputs_value(bit=8, signed=True, norm_to_one=False)
        code = get_code(running_info, inputs, constant_ones)
        position_weights = solve_position_weight(code, value, alpha=0.1)
        value_appr, maximal_relative_error, rmse = solve_max_re_rmse(position_weights, code, value)

        searched_info = []
        scale_x, scale_w = 2 ** (8 - x_signed), 2 ** (8 - w_signed)
        scale_v = scale_x * scale_w
        searched_info.append({
            'quantization type': 'Uniform',
            'bit-width of activation': 8, 'bit-width of weight': 8,
            'bit-width of product': position_weights.numel(),
            'root mean square error': rmse / scale_v,
            'digit-weight of product code': position_weights / scale_v,
            'output code of product': code,
            'approximate value of product': value_appr / scale_v,
            'method to generate product': 'kronecker product of activation and weight: v=torch.kron(x, w)',
        })
        del value, inputs, constant_ones
        return searched_info


def apply_delta_for_finetune(searched_info, delta=0):
    if searched_info is not None and delta > 0:
        assert len(searched_info) == 1, 'length of searched_info > 1'
        position_weights = searched_info[0]['digit-weight of product code'] * 16384
        max_position_weight = position_weights.abs().max()
        position_weights = normalize(position_weights, bit=16)
        area, _ = read_area_power()
        position_weights_new = apply_delta(position_weights, area, delta)
        searched_info[0][
            'digit-weight of product code'] = position_weights_new * max_position_weight / 16383 / 16384
        searched_info[0]['approximate value of product'] = searched_info[0]['digit-weight of product code'] @ \
                                                           searched_info[0]['output code of product'].cpu()
    return searched_info


def get_file_name(col=2, row=256, target=64, search=128, idx=0, n_parents=10, n_offsprings=50, n_champions=2,
                  mutate_strategy='dynamic', mutate_rate=0.1):
    return f"data-" \
           f"{row}row-" \
           f"{col}col-" \
           f"{target}bit-" \
           f"{search}b-" \
           f"idx{idx}-" \
           f"{n_parents}pars-" \
           f"{n_offsprings}offs-" \
           f"{n_champions}chas-"\
           f"{mutate_strategy}-" \
           f"{mutate_rate}mutate"


def get_approx_product(searched_info, a_bit=8, w_bit=8, product_bit=64, mode='Approx-INT'):
    approx_product_value_2d, rmse = None, 0.0
    if mode == 'Approx-INT':
        assert len(searched_info) == 1, 'length of searched_info > 1'
        info = searched_info[0]
        assert info['bit-width of product'] == product_bit
        assert info['bit-width of weight'] == w_bit
        assert info['bit-width of activation'] == a_bit
        print(f"bit-width of product: {info['bit-width of product']}\t"
              f"root mean square error:{info['root mean square error']:.2e}")
        approx_product_value_2d = info['approximate value of product'].view(256, 256)
        # digit_weight = info['digit-weight of product code']
        rmse = info['root mean square error']
    return approx_product_value_2d, rmse


def main():
    pass


if __name__ == '__main__':
    main()
