#
# Script for running gplearn comparison
#

import sys

from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from generate_random_expression import generate_random_expression


random_states = [11284, 11964, 15795, 21575, 22118, 23654, 29802,  5390,  6265, 860]

random_state_idx = int(sys.argv[1])
random_state = random_states[random_state_idx]
pop_size = int(sys.argv[2])
n_gens = int(sys.argv[3])
benchmark_init = (sys.argv[4] == "True")

results_file = f'results/gplearn_r{random_state_idx}_p{pop_size}_g{n_gens}_b{benchmark_init}.csv'

def read_file(filename, label='target', sep=None):

    if filename.endswith('gz'):
        compression = 'gzip'
    else:
        compression = None

    print('compression:',compression)
    print('filename:',filename)

    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(filename, sep=sep, compression=compression,
                engine='python')

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert(X.shape[1] == feature_names.shape[0])

    return X, y, feature_names

# Load dataset

dataset_name = 'banana'

if dataset_name == 'banana':
    dataset_file_path = '~/elm/pmlb/datasets/banana/banana.tsv.gz'
elif dataset_name == '523_analcatdata_neavote':
    dataset_file_path = 'pmlb/datasets/523_analcatdata_neavote/523_analcatdata_neavote.tsv.gz'
elif dataset_name == '228_elusage':
    dataset_file_path = 'pmlb/datasets/228_elusage/228_elusage.tsv.gz'
elif dataset_name == '663_rabe_266':
    dataset_file_path = 'pmlb/datasets/663_rabe_266/663_rabe_266.tsv.gz'
else:
    print(f'Invalid dataset name: {dataset_name}')
    raise

features, labels, feature_names = read_file(dataset_file_path, sep='\t')
X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=random_state)
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
y_test_scaled = sc_y.fit_transform(y_test.reshape(-1,1)).flatten()

# Create r2 fitness function
def r2_function(y, y_pred, w):
    return r2_score(y, y_pred)

r2_fitness_function = make_fitness(function=r2_function, greater_is_better=True)


if benchmark_init:

    # Do one generation of gplearn gplearn, using default params from SRBench,
    # but with `abs` added to the function set, since it is in the benchmark exps
    # and supported by gplearn.
    # https://github.com/cavalab/srbench/blob/master/experiment/methods/gplearn.py
    # (Accessed: 4/20/2023)
    est = SymbolicRegressor(
                            tournament_size=20,
                            init_depth=(2, 6),
                            init_method='half and half',
                            metric=r2_fitness_function, #metric='mean absolute error',
                            stopping_criteria=1.0,
                            parsimony_coefficient=0.001,
                            p_crossover=0.9,
                            p_subtree_mutation=0.01,
                            p_hoist_mutation=0.01,
                            p_point_mutation=0.01,
                            p_point_replace=0.05,
                            max_samples=1.0,
                            function_set= ('add', 'sub', 'mul', 'div', 'log',
                                            'sqrt', 'sin','cos', 'abs'),
                            population_size=1000,
                            generations=1, # The rest will be done after manual population overwrite|
                            verbose=1
                           )
    est.fit(X_train_scaled, y_train_scaled)

    # Overwrite programs with benchmark expressions
    print("REPLACING PROGRAMS")
    final_gen_programs = est._programs[-1]
    for program in final_gen_programs:
        random_expression = generate_random_expression(program)
        program.program = random_expression
        program.fitness = program.fitness()

    # Continue running evolution with warm start
    est.set_params(generations=n_gens, warm_start=True)
    est.fit(X_train_scaled, y_train_scaled)

else:

    # Do gplearn, using default params from SRBench
    # https://github.com/cavalab/srbench/blob/master/experiment/methods/gplearn.py (Accessed: 4/20/2023)
    est = SymbolicRegressor(
                            tournament_size=20,
                            init_depth=(2, 6),
                            init_method='half and half',
                            metric=r2_fitness_function, #metric='mean absolute error',
                            stopping_criteria=1.0,
                            parsimony_coefficient=0.001,
                            p_crossover=0.9,
                            p_subtree_mutation=0.01,
                            p_hoist_mutation=0.01,
                            p_point_mutation=0.01,
                            p_point_replace=0.05,
                            max_samples=1.0,
                            function_set= ('add', 'sub', 'mul', 'div', 'log',
                                            'sqrt', 'sin','cos'),
                            population_size=pop_size, #1000,
                            generations=n_gens, # 500
                            verbose=1
                           )
    est.fit(X_train_scaled, y_train_scaled)

print(est._program)

test_score = est.score(X_test_scaled, y_test_scaled)
print(test_score)

results_df = pd.DataFrame(est.run_details_)
results_df['test'] = test_score
results_df.to_csv(results_file)

