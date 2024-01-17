from multiprocessing import Pool
import os
import seaborn as sns; sns.set_style('white')
from itertools import product

def run_one_exp(params):
    lr, momentum, weight_decay = params
    output_filename = 'output_lr' + str(lr) + '_m' + str(momentum) + '_wd' + str(weight_decay) + '.txt'
    cmd = f'python -u hyperparameter.py --lr {lr} --momentum {momentum} --wd {weight_decay} > output/{output_filename}'
    print(cmd)
    os.system(cmd)

def run_in_parallel():
    learning_rate_values = [0.01, 0.005, 0.001]
    momentum_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    weight_decay_values = [0, 1e-4, 1e-3]
    parameters = list(product(learning_rate_values, momentum_values, weight_decay_values))
    with Pool(processes=6) as pool:
        pool.map(run_one_exp, parameters)


if __name__ == '__main__':
    run_in_parallel()

