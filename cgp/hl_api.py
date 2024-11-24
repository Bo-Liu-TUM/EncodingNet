import warnings
from typing import Callable, Optional

import numpy as np

from .ea import MuPlusLambda
from .individual import IndividualBase
from .population import Population


import time
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def evolve(
    pop: Population,
    objective: Callable[[IndividualBase], IndividualBase],
    ea: MuPlusLambda,
    min_fitness: float = np.inf,
    termination_fitness: float = np.inf,
    max_generations: int = np.inf,
    max_objective_calls: int = np.inf,
    print_progress: Optional[bool] = False,
    callback: Optional[Callable[[Population, MuPlusLambda], None]] = None,
    logger = None
) -> None:
    """
    Evolves a population and returns the history of fitness of parents.

    Parameters
    ----------
    pop : Population
        A population class that will be evolved.
    objective : Callable
        An objective function used for the evolution. Needs to take an
        individual (Individual) as input parameter and return
        a modified individual (with updated fitness).
    ea : EA algorithm instance
        The evolution algorithm. Needs to be a class instance with an
        `initialize_fitness_parents` and `step` method.
    min_fitness : float
        Minimum fitness at which the evolution is stopped.
        Warning: This argument is deprecated and will be removed in the 0.4
        release. Please use `termination_fitness` instead.
    termination_fitness : float
        Minimum fitness at which the evolution is terminated
    max_generations : int
        Maximum number of generations.
        Defaults to positive infinity.
        Either this or `max_objective_calls` needs to be set to a finite value.
    max_objective_calls: int
        Maximum number of function evaluations.
        Defaults to positive infinity.
    print_progress : boolean, optional
        Switch to print out the progress of the algorithm. Defaults to False.
    callback :  callable, optional
        Called after each iteration with the population instance.
        Defaults to None.

    Returns
    -------
    None
    """
    if np.isfinite(min_fitness) and np.isfinite(termination_fitness):
        raise RuntimeError(
            "Both `min_fitness` and `termination_fitness` have been set. The "
            "`min_fitness` argument is deprecated and will be removed in the 0.4 "
            "release. Please use `termination_fitness` instead."
        )

    if np.isfinite(min_fitness):
        warnings.warn(
            DeprecationWarning(
                "The `min_fitness` argument is deprecated and "
                "will be removed in the 0.4 release. Please use "
                "`termination_fitness` instead."
            )
        )
        termination_fitness = min_fitness

    if np.isinf(max_generations) and np.isinf(max_objective_calls):
        raise ValueError("Either max_generations or max_objective_calls must be finite.")

    ea.initialize_fitness_parents(pop, objective)
    if callback is not None:
        callback(pop, ea)

    # perform evolution
    max_fitness = np.finfo(float).min
    # Main loop: -1 offset since the last loop iteration will still increase generation by one
    generation_time = AverageMeter('Time', ':6.3f')
    end = time.time()

    while pop.generation < max_generations - 1 and ea.n_objective_calls < max_objective_calls:

        pop = ea.step(pop, objective)

        # progress printing, recording, checking exit condition etc.; needs to
        # be done /after/ new parent population was populated from combined
        # population and /before/ new individuals are created as offsprings
        assert isinstance(pop.champion.fitness, float)
        if pop.champion.fitness > max_fitness:
            max_fitness = pop.champion.fitness

        generation_time.update(time.time() - end)
        end = time.time()
        if print_progress:
            if np.isfinite(max_generations):  # \033[K
                content = f"[{pop.generation + 1}/{max_generations}]\t"\
                          f"Time: {generation_time.avg:.2f}s\t"\
                          f"MaxFit: {max_fitness:.2f}\t"\
                          f"Size: [{pop._genome_params.get('n_rows')}r*{pop._genome_params.get('n_columns')}c]\t"\
                          f"Out.bit: [{pop.champion.custom_attr_outputs_bit_width}/{pop._genome_params.get('n_outputs')}]-b\t"\
                          f"Max.re: [{pop.champion.custom_attr_maximal_relative_error:.4%}/{pop.champion.custom_attr_maximal_relative_error_th:.2%}]\t"\
                          f"Area: {pop.champion.custom_attr_area:.2f}\t"\
                          f"Mut.rate: {ea._mutation_rate:.2%}\t"
                # print(content)  # print(# end="",# flush=False,)
                logger.info(content)
            elif np.isfinite(max_objective_calls):
                content = f"[{ea.n_objective_calls}/{max_objective_calls}]\t"\
                          f"Time: {generation_time.avg:.2f}s\t"\
                          f"MaxFit: {max_fitness:.2f}\t"\
                          f"Size: [{pop._genome_params.get('n_rows')}r*{pop._genome_params.get('n_columns')}c]\t"\
                          f"Out.bit: [{pop.champion.custom_attr_outputs_bit_width}/{pop._genome_params.get('n_outputs')}]-b\t"\
                          f"Max.re: [{pop.champion.custom_attr_maximal_relative_error:.4%}/{pop.champion.custom_attr_maximal_relative_error_th:.2%}]\t"\
                          f"Area: {pop.champion.custom_attr_area:.2f}\t"\
                          f"Zeros: {pop.champion.custom_attr_zeros_meets_ratio:.2%}\t"\
                          f"Mut.rate: {ea._mutation_rate:.2%}\t"
                # print(content)  # print(# end="",# flush=False,)
                logger.info(content)
            else:
                assert False  # should never be reached

        if callback is not None:
            callback(pop, ea)

        if pop.champion.fitness + 1e-10 >= termination_fitness:
            break

    if print_progress:
        print()
