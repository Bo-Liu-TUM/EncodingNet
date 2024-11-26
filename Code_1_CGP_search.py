"""
this is for encoding search using cartesian genetic programming
"""

import cgp
from cgp.ea import dynamic_mutation_rate
from cgp import Identity, NOT, NOR2, NAND2, XNOR2, OR2, AND2, XOR2

import os
import torch
import logging
import encode_tools as tools
from kuai_log import get_logger

import argparse


def solve_position_weight(code, value_true, alpha=0.):
    position_weight = value_true @ code.T @ torch.linalg.inv(code @ code.T)
    position_weight = position_weight.round()
    value_pred = position_weight @ code
    errors = value_pred - value_true
    relative_errors = torch.abs(errors) / max(value_true.max(), -value_true.min())
    maximal_relative_error = relative_errors.max()
    return position_weight, value_pred, maximal_relative_error


def get_total_area(addresses, graph, active_nodes={}):
    # according to the address, get the node info
    # if it is input node, return 0
    # else save idx and area of this node, return input address of this node
    for address in addresses:
        hidden_idx = address - graph.input_nodes.__len__()
        if hidden_idx >= 0:
            node = graph.hidden_nodes[hidden_idx]
            active_nodes.update({node.idx: node.custom_attr_area})
            active_nodes = get_total_area(node.addresses, graph, active_nodes)
    return active_nodes


# Define an objective function.
# The objective function takes an individual as an argument and updates the fitness of the individual.
def objective(individual):
    if not individual.fitness_is_None():
        return individual

    graph = cgp.CartesianGraph(individual.genome)
    # print(graph.pretty_str())
    f = individual.to_func()
    outputs = f(*inputs)
    outputs.append(constant_ones)  # add one column of ones as the constant
    range_idx = len(outputs)
    outputs = torch.stack(outputs).type(torch.float32)  #
    # outputs is B, value is v, target_bit_width is M
    # position_weights: solve position weights with ridge regression
    position_weights = tools.solve_position_weight(outputs, value, alpha=0.1)
    # outputs_idx: identify the M largest absolute values in position weights
    outputs_idx = position_weights.abs().view(-1).argsort(descending=True)[0:target_bit_width]
    # select the output nodes with the index, solve maximal relative error, and area
    value_pred, max_re, _ = tools.solve_max_re_rmse(position_weights.round()[:, outputs_idx], outputs[outputs_idx], value)

    # here is another way to solve the best idx, but it is very time-consuming, if you are interested, please have a try
    # ind_cols_idx = find_maximal_independent_bits(outputs, max_rank=None)
    # ind_cols_idx = drop_meaningless_bits(outputs[ind_cols_idx], value, ind_cols_idx, non_zero_th=1e-1, alpha=0.)
    # position_weights, value_pred, max_re, outputs_idx = solve_and_drop_bits(outputs[ind_cols_idx], value, ind_cols_idx, target_bit_width, alpha=0.)

    active_nodes = {}
    for idx in outputs_idx:
        # print(graph.output_nodes[idx].addresses)
        if idx < range_idx - 1:  # added constant one cannot be indexed
            active_nodes = get_total_area(graph.output_nodes[idx].addresses, graph, active_nodes)
    area = sum(active_nodes.values())

    if max_re <= maximal_relative_error_th:
        area_term = area
        max_re_term = maximal_relative_error_th
    else:
        area_term = area_max
        max_re_term = max_re.item()

    fitness = area_term + max_re_term
    individual.fitness = -float(fitness)
    individual.custom_attr_area = area
    individual.custom_attr_outputs_bit_width = outputs_idx.__len__()
    individual.custom_attr_valid_output_bits = outputs_idx.tolist()
    individual.custom_attr_maximal_relative_error = max_re.item()
    individual.custom_attr_maximal_relative_error_th = maximal_relative_error_th
    return individual


def drop_one_bit(position_weights, sample_idx):
    order = position_weights.abs().view(-1).argsort(descending=True)[:-1]  # drop the smallest one
    return sample_idx[order]


def find_maximal_independent_bits(matrix, max_rank=None):
    max_rank = torch.linalg.matrix_rank(matrix) if max_rank is None else max_rank
    num_rows, num_cols = matrix.shape
    matrix = matrix.T if num_rows < num_cols else matrix
    Q, R = torch.linalg.qr(matrix, mode='complete')
    diag_abs = torch.abs(torch.diag(R))
    sorted_indices = torch.argsort(diag_abs, descending=True)
    independent_cols_idx = sorted_indices[0:max_rank]
    return independent_cols_idx


def drop_meaningless_bits(code, value_true, independent_cols_idx, non_zero_th=1e-1, alpha=0.):
    position_weights, _, _ = solve_position_weight(code, value_true, alpha=alpha)
    selected = torch.nonzero(position_weights.view(-1).abs() > non_zero_th).flatten()
    return independent_cols_idx[selected]


def solve_and_drop_bits(code_sample, value_true, independent_cols_idx, target_bit_width=1, alpha=0.):
    rank_list = list(range(code_sample.shape[0], target_bit_width, -1))
    sample_idx = torch.arange(code_sample.shape[0])
    if torch.cuda.is_available():
        sample_idx = sample_idx.cuda(args.gpu)
    position_weights, value_pred, max_re = solve_position_weight(code_sample, value_true, alpha=alpha)
    for rank in rank_list:
        sample_idx = drop_one_bit(position_weights, sample_idx)
        position_weights, value_pred, max_re = solve_position_weight(code_sample[sample_idx], value_true, alpha=alpha)
        # print('rank: {}\t'
        #       'rmse: {:.3f}\t'
        #       'max_re: {:.3f}\t'.format(rank, rmse, max_re))
    return position_weights, value_pred, max_re, independent_cols_idx[sample_idx]


def main():
    # Define parameters for the population, the genome, the evolutionary algorithm and the evolve function.
    population_params = {"n_parents": n_parents, "seed": None}  # seed used for the reproduction 8188211

    genome_params = {
        "n_inputs": 16,
        "n_outputs": output_bit_width_during_search,
        "n_columns": gate_levels,
        "n_rows": gate_rows,
        "levels_back": gate_levels,
        "primitives": primitives,  # optional operations in each node
    }

    ea_params = {"n_offsprings": n_offsprings, "mutation_rate": mutate_rate,
                 "tournament_size": n_champions, "n_processes": 1,
                 # "reorder_genome": True  # only valid when n_rows == 1 and levels_back == n_columns
                 }

    evolve_params = {"max_generations": max_generations, "termination_fitness": 0.0}

    # Initialize a population and an evolutionary algorithm instance
    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)
    if continue_running and os.path.exists(running_file):
        import pickle
        with open(running_file, 'rb') as f:
            running_info = pickle.load(f)  # load
        for e in running_info["pop.parents"]:
            e.reset_fitness()  # individual.reset_fitness()
        pop.parents = running_info["pop.parents"]
        pop.generation = running_info["pop.generation"]
        history = running_info["history"]
        if pop.generation > max_generations:
            exit()
    else:
        # Define a callback function to record information about the progress of the evolution:
        history = {
            # "fitness_parents": [],  # too large if it is saved
            "champion.fitness": [],
            "champion.area": [],
            "champion.maximal_relative_error": [],
            "champion.maximal_relative_error_th": [],
            "champion.outputs_bit_width": [],
            "mutate_rate": [],
        }

    def recording_callback(pop, ea):
        # history["fitness_parents"].append(pop.fitness_parents())
        history["champion.fitness"].append(pop.champion.fitness)
        history["champion.area"].append(pop.champion.custom_attr_area)
        history["champion.maximal_relative_error"].append(pop.champion.custom_attr_maximal_relative_error)
        history["champion.maximal_relative_error_th"].append(pop.champion.custom_attr_maximal_relative_error_th)
        history["champion.outputs_bit_width"].append(pop.champion.custom_attr_outputs_bit_width)
        history["mutate_rate"].append(ea._mutation_rate)
        running_info = {
            'population_params': population_params,
            'genome_params': genome_params,
            'ea_params': ea_params,
            'evolve_params': evolve_params,
            'history': history,
            'pop.generation': pop.generation,
            'pop.parents': pop.parents,
            'pop.champion': pop.champion,
            # 'ea': ea
        }
        import pickle
        print(running_file, 'saving---')
        with open(running_file, 'wb') as f:
            pickle.dump(running_info, f, protocol=pickle.HIGHEST_PROTOCOL)  # save
        print(running_file, 'saved!')

        if mutate_strategy == 'dynamic':
            max_re = pop.champion.custom_attr_maximal_relative_error
            ea._mutation_rate = dynamic_mutation_rate(max_re)

    # Use the evolve function that ties everything together and executes the evolution:
    cgp.evolve(pop, objective, ea, **evolve_params,
               print_progress=True, callback=recording_callback, logger=logger)

    print("done!")


parser = argparse.ArgumentParser(description='PyTorch CGP Encode Searching')
parser.add_argument('--gpu', type=int, default=0, choices=[0, 1, 2, 3])
parser.add_argument('--target', type=int, default=64)
parser.add_argument('--search', type=int, default=128)
parser.add_argument('--cols', type=int, default=2)
parser.add_argument('--rows', type=int, default=256)
parser.add_argument('--th', type=float, default=0.1)
parser.add_argument('--gen', type=int, default=2500)
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--n-parents', type=int, default=10)
parser.add_argument('--n-offsprings', type=int, default=50)
parser.add_argument('--n-champions', type=int, default=2)
parser.add_argument('--mutate-strategy', type=str, default='dynamic', choices=['dynamic', 'fixed'])
parser.add_argument('--mutate-rate', type=float, default=0.1)
args = parser.parse_args()
args.running_cache = './running_cache/'

if __name__ == '__main__':
    max_generations = args.gen
    n_parents = args.n_parents
    n_idx = args.idx
    n_offsprings = args.n_offsprings
    n_champions = args.n_champions
    mutate_strategy = args.mutate_strategy
    mutate_rate = args.mutate_rate
    maximal_relative_error_th = args.th / 100  # 0.01%, 0.5%, 1.0%, 2.0%, 5.0%, 10.0%, 20.0%
    target_bit_width = args.target  # 48  # 64, 63, ..., 16, 15
    output_bit_width_during_search = args.search  # 256
    gate_levels = args.cols  # 1
    gate_rows = args.rows  # 256

    total_gates = gate_rows * gate_levels
    primitives = (NAND2, NOR2, XNOR2, AND2, OR2, XOR2, NOT, Identity)
    area_max = total_gates * max(e.custom_attr_area for e in primitives)

    continue_running = True
    running_path = os.path.join(args.running_cache, f'th{args.th}%')   # backup/

    if not os.path.exists(running_path):
        os.makedirs(running_path)

    running_file = os.path.join(running_path,
                                f"data-{gate_rows}row-{gate_levels}col-"
                                f"{target_bit_width}bit-{output_bit_width_during_search}b-"
                                f"idx{n_idx}-"
                                f"{n_parents}pars-{n_offsprings}offs-{n_champions}chas-"
                                f"{mutate_strategy}-{mutate_rate}mutate.pickle")

    running_log = f"data-{gate_rows}row-{gate_levels}col-" \
                  f"{target_bit_width}bit-{output_bit_width_during_search}b-" \
                  f"idx{n_idx}-" \
                  f"{n_parents}pars-{n_offsprings}offs-{n_champions}chas-" \
                  f"{mutate_strategy}-{mutate_rate}mutate.log"

    logger = get_logger(name='gcp', level=logging.INFO, log_filename=running_log,
                        log_path=running_path, is_add_file_handler=True,
                        formatter_template='{host}-cuda:' + str(args.gpu) + '-{levelname}-{message}')

    if torch.cuda.is_available():
        logger.info("GPU is available, congratulations!")
        logger.info(f"Total devices: {torch.cuda.device_count()}\t"
                    f"Allowed devices: {list(range(torch.cuda.device_count()))}")
        if args.gpu < torch.cuda.device_count():
            logger.info(f"Assigned device: {args.gpu}, start running, good luck!")
        else:
            logger.error(f"Assigned device: {args.gpu}, out of allowed range.")
            exit()
    else:
        logger.error("GPU is not available, using CPU will be very time-consuming.")
        if 'y' in input("continue? [yes/no]"):
            logger.info(f"Using CPU, let's have a try!")
        else:
            exit()

    value, inputs, constant_ones = tools.gen_inputs_value(bit=8, signed=True, norm_to_one=False)
    if torch.cuda.is_available():
        value = value.to(torch.float32).cuda(args.gpu)
        inputs = [e.cuda(args.gpu) for e in inputs]
        constant_ones = constant_ones.to(torch.bool).cuda(args.gpu)

    logger.info(f"max_re_th:{maximal_relative_error_th:.2%}\t\t"
                f"[{gate_rows}rows * {gate_levels}cols]\t"
                f"[target {target_bit_width}bit / search {output_bit_width_during_search}bit]\t"
                f"idx{n_idx}\t"
                f"[{n_parents}pars-{n_offsprings}offs-{n_champions}chas]\t"
                f"{mutate_strategy}{mutate_rate}% mutate")
    main()
else:
    print('Code_1_CGP_search.py imported')

