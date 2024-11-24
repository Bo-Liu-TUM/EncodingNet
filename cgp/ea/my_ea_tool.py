"""
my user defined tool for ea
"""


def dynamic_mutation_rate(max_re):
    if 0.9 < max_re:
        mutation_rate = 0.3
    elif 0.8 < max_re <= 0.9:
        mutation_rate = 0.2
    elif 0.7 < max_re <= 0.8:
        mutation_rate = 0.2
    elif 0.6 < max_re <= 0.7:
        mutation_rate = 0.1
    elif 0.5 < max_re <= 0.6:
        mutation_rate = 0.1
    elif 0.4 < max_re <= 0.5:
        mutation_rate = 0.1
    elif 0.3 < max_re <= 0.4:
        mutation_rate = 0.1
    elif 0.2 < max_re <= 0.3:
        mutation_rate = 0.1
    elif 0.1 < max_re <= 0.2:
        mutation_rate = 0.05
    elif 0.05 < max_re <= 0.1:
        mutation_rate = 0.0285
    elif 0.04 < max_re <= 0.05:
        mutation_rate = 0.0214
    elif 0.03 < max_re <= 0.04:
        mutation_rate = 0.0143
    elif 0.025 < max_re <= 0.03:
        mutation_rate = 0.0133
    elif 0.02 < max_re <= 0.025:
        mutation_rate = 0.0123
    elif 0.01 < max_re <= 0.02:
        mutation_rate = 0.0113
    elif 0.005 < max_re <= 0.01:
        mutation_rate = 0.0100
    elif max_re <= 0.005:
        mutation_rate = 0.0100
    return mutation_rate

