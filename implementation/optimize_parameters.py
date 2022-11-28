import numpy as np
from scipy.integrate import quad as integrate
import json
import time
from sympy.ntheory import factorint
import itertools
import math
import pandas as pd


def _false_positive_probability(threshold, b_1, b_2, r_1, r_2):
    _probability = lambda s : 1 - (1 - (1 - (1 - s**float(r_1))**float(b_1))**float(r_2))**float(b_2)
    a, err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b_1, b_2, r_1, r_2):
    _probability = lambda s : 1 - (1 - (1 - (1 - (1 - s**float(r_1))**float(b_1))**float(r_2))**float(b_2))
    a, err = integrate(_probability, threshold, 1.0)
    return a

def compute_weighted_average_error(threshold, b_1, b_2, r_1, r_2, false_positive_weight=0.5, false_negative_weight=0.5 ):
    fp = _false_positive_probability(threshold, b_1, b_2, r_1, r_2)
    fn = _false_negative_probability(threshold, b_1, b_2, r_1, r_2)
    error = fp * false_positive_weight + fn * false_negative_weight
    return error, fp, fn


def optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight, amplified, minimum_r1=None):
    '''
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    '''
    min_error = float("inf")
    last_error = float("inf")
    opt = ((), ())
    all_outcomes = []
    min_error_b_0 = {}
    for b_0 in range(num_perm, 0, -1):
        print(f'b_0: {b_0}')
        max_r_1 = int(num_perm / b_0)
        min_error_r1 = {}
        for r_1 in range(max_r_1, 0, -1):
            # if minimum_r1 and r_1 < minimum_r1:
            #     continue
            b_1_start_range = 1 if amplified else b_0
            for b_1 in range(b_1_start_range, b_0 + 1):
                n_2 = int(num_perm / (b_1 * r_1))
                for b_2 in range(1, n_2 + 1):
                    r_2 = int(n_2 / b_2)
                    # if error > last_error:
                    #     break
                    error, _, _ = compute_weighted_average_error(threshold, b_1, b_2, r_1, r_2)
                    dict_ = {}
                    dict_['params'] = ((r_1, b_1), (r_2, b_2))
                    dict_['r1'] = r_1
                    dict_['b1'] = b_1
                    dict_['r2'] = r_2
                    dict_['b2'] = b_2
                    dict_['error'] = error
                    all_outcomes.append(dict_)
                    # if b_0 not in min_error_b_0.keys():
                    #     min_error_b_0[b_0] = error
                    # if error < min_error_b_0[b_0]:
                    #     min_error_b_0[b_0] = error
                    if error < min_error:
                        min_error = error
                        b = (b_1, b_2)
                        r = (r_1, r_2)
                        opt = (b, r)
    print(f'optimal_parameters: {opt}')
    return opt, all_outcomes, min_error_b_0


def _get_divisors(number):
    factors = factorint(number)
    # return_set = {1, 16}
    list_factors = [key for key, value in factors.items() for i in range(1, value + 1)]
    return_set = {1, number}
    for i in range(len(list_factors)):
        combis = itertools.combinations(list_factors, i)
        for com in combis:
            return_set.add(math.prod(com))
    # for key, value in factors.items():
    #     for i in range(1, value + 1):
    #         return_set.add(key ** i)
    # return_set.update({1, number})
    return return_set


def manual_get_divisor(number):
    return_set = set()
    for i in range(1, int(number / 2) + 1):
        if number % i == 0:
            return_set.add(i)
            return_set.add(int(number / i) )
    return return_set

def optimal_param_new(threshold, num_perm, false_positive_weight, false_negative_weight, amplified, minimum_r1=1):
    '''
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    '''
    start_time = time.time()
    min_error = float("inf")
    opt = ((), ())
    # all_outcomes = []
    count = 0
    for r_1 in range(minimum_r1, num_perm + 1):
        max_b_0 = int(num_perm / r_1)
        # min_error_b_0 = {}
        for b_0 in range(max_b_0, 0, -1):
            b_1_options = _get_divisors(b_0) if amplified else {b_0}
            for b_1 in b_1_options:
                n_2 = int(b_0 / b_1)
                b_2_options = _get_divisors(n_2) if amplified else {1}
                for b_2 in b_2_options:
                    count += 1
                    r_2 = int(n_2 / b_2)
                    fp = _false_positive_probability(threshold, b_1, b_2, r_1, r_2)
                    fn = _false_negative_probability(threshold, b_1, b_2, r_1, r_2)
                    error = fp * false_positive_weight + fn * false_negative_weight
                    if error < min_error:
                        min_error = error
                        b = (b_1, b_2)
                        r = (r_1, r_2)
                        opt = (b, r)
    print(f'optimal_parameters: {opt}')
    print(f'evaluation took {time.time() - start_time} seconds')
    print(f'threshold estimated: {threshold_comp(r[0],b[0])}')
    return opt, min_error


def threshold_comp(r,b):
    return ((r - 1) / (r * b - 1)) ** (1/r)

if __name__ == "__main__":
    thresholds = list(np.arange(start=0.05, stop=1, step=0.05))
    num_perms = [16, 32, 64, 128, 256, 512, 1024]
    amplified_list = [True, False]

    list_configurations = [{'threshold': threshold, 'num_perm': num_perm, 'amplified': amplified}
                           for threshold in thresholds for num_perm in num_perms for amplified in amplified_list]

    with open('./results/parameter_config.json', 'r') as file_:
        existing_config = json.load(file_)

    existing_config = []


    print(f'num configs: {len(list_configurations)}')

    for index, config in enumerate(list_configurations):
        # search in existing config
        params = next((x['params'] for x in existing_config if  (abs(x['threshold'] - config['threshold']) < 0.01
                                                                and x['num_perm'] == config['num_perm']
                                                                and x['amplified'] is config['amplified'])), "not_found")
        print(config)
        print(params)
        if params == 'not_found':

            # optimize parmameters
            params, error = optimal_param_new(threshold=config['threshold'],
                                   num_perm=config['num_perm'],
                                   false_positive_weight=0.5,
                                   false_negative_weight=0.5,
                                   amplified=config['amplified'])

            config['params'] = params
            config['threshold'] = round(config['threshold'], 2)
            list_configurations[index] = config
            print(config)

    with open('./results/parameter_config.json', 'w') as file_:
        json.dump(list_configurations, file_)
