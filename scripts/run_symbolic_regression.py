"""
Sample script for running LMX symbolic regression experiments.

This script assumes PMLR datasets have been downloaded to ./pmlr/datasets/

Results will be saved under ./results, so make sure this directory exists.
"""


import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import sys

import argparse
import pandas as pd
import numpy as np
from numpy import sin, cos, arcsin, arccos, exp, pi, tan, tanh, e
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from transformers import pipeline

random_states = [11284, 11964, 15795, 21575, 22118, 23654, 29802,  5390,  6265, 860]

parser = argparse.ArgumentParser()
parser.add_argument('--device_idx', type=int, default=0)
parser.add_argument('--dataset_name', default='banana')
parser.add_argument('--max_generations', type=int, default=5000)
parser.add_argument('--lm_name', default='galactica')
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--exp_id', required=True)
args = parser.parse_args()

dataset_name = args.dataset_name
lm_name = args.lm_name
device_idx = args.device_idx
split_idx = args.split_idx
exp_id = args.exp_id


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


# Set up initial set of expressions to draw from
# Clean it up for runnability and consistency
expr_list = [
 '0.3*x1*sin(2*3.14*x1)',
 'pow(x1,3)*exp(-x1)*cos(x1)*sin(x1)*(pow(sin(x1),2)*cos(x1)-1)',
 'div(30*x1*x3,(x1-10)*pow(x2,2))',
 'div(x1*(x1+1),2)',
 'log(x1)',
 'sqrt(x1)',
 'log(x1+sqrt(pow(x1,2)+1))',
 'pow(x1,x2)',
 'x1*x2+sin((x1-1)*(x2-1))',
 'pow(x1,4)-pow(x1,3)+div(pow(x2,2),2)-x2',
 '6*sin(x1)*cos(x2)',
 'div(8,2+pow(x1,2)+pow(x2,2))',
 'div(pow(x1,3),5)+div(pow(x2,3),2)-x2-x1',
 '1.57+24.3*x4',
 '0.23+14.2*div((x4+x2),(3*x5))',
 '4.9*div((x4-x1+div(x2,x5)),(3*x5))-5.41',
 '0.13*sin(x3)-2.3',
 '3+2.13*log(abs(x5))',
 '1.3+0.13*sqrt(abs(x1))',
 '2.1380940889*(1-exp(-0.54723748542*x1))',
 '6.87+11*sqrt(abs(7.23*x1*x4*x5))',
 'div(sqrt(abs(x1)),log(abs(x2)))*div(exp(x3),pow(x4,2))',
 '0.81+24.3*div(2*x2+3*pow(x3,2),((4*pow(x4,3)+5*pow(x5,4))))',
 '6.87+11*cos(7.23*pow(x1,3))',
 '2-2.1*cos(9.8*x1)*sin(1.3*x5)',
 '32.0-(3.0*((tan(x1)/tan(x2))*(tan(x3)/tan(x4))))',
 '22.0-(4.2*((cos(x1)-tan(x2))*(tanh(x3)/sin(x4))))',
 '12.0-(6.0*((tan(x1)/exp(x2))*(log(x3)-tan(x4))))',
 'pow(x1,5)-2*pow(x1,3)+x1',
 'pow(x1,6)-2*pow(x1,4)+pow(x1,2)',
 'div(pow(x1,2)*pow(x2,2),(x1+x2))',
 'div(pow(x1,5),pow(x2,3))',
 'pow(x1,3)+pow(x1,2)+x1',
 'pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'pow(x1,6)+pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'sin(pow(x1,2))*cos(x1)-1',
 'sin(x1)+sin(x1+pow(x1,2))',
 'log(x1+1)+log(pow(x1,2)+1)',
 'sin(x1)+sin(pow(x2,2))',
 '2*sin(x1)*cos(x2)',
 'pow(x1,x2)',
 'pow(x1,4)-pow(x1,3)+div(pow(x2,2),2)-x2',
 'pow(x1,4)-pow(x1,3)+div(pow(x2,2),2)-x2',
 '3.39*pow(x1,3)+2.12*pow(x1,2)+1.78*x1',
 'sin(pow(x1,2))*cos(x1)-0.75',
 'sin(1.5*x1)*cos(0.5*x2)',
 '2.7*pow(x1,x2)',
 'sqrt(1.23*x1)',
 'pow(x1,0.426)',
 '2*sin(1.3*x1)*cos(x2)',
 'log(x1+1.4)+log(pow(x1,2)+1.3)',
 '1./3+x1+sin(pow(x1,2))',
 'sin(pow(x1,2))*cos(x1)-2',
 'sin(pow(x1,3))*cos(pow(x1,2))-1',
 'log(x1+1)+log(pow(x1,2)+1)+log(x1)',
 'pow(x1,4)-pow(x1,3)+pow(x2,2)-x2',
 '4*pow(x1,4)+3*pow(x1,3)+2*pow(x1,2)+x1',
 'div(exp(x1)-exp(-1*x1),2)',
 'div(exp(x1)+exp(-1*x1),2)',
 'pow(x1,9)+pow(x1,8)+pow(x1,7)+pow(x1,6)+pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 '6*sin(x1)*cos(x2)',
 'div(pow(x1,2)*pow(x2,2),(x1+x2))',
 'div(pow(x1,5),pow(x2,3))',
 'pow(x1,1/3)',
 'pow(x1,3)+pow(x1,2)+x1+sin(x1)+sin(pow(x2,2))',
 'pow(x1,1/5)',
 'pow(x1,2/3)',
 '4*sin(x1)*cos(x2)',
 'sin(pow(x1,2))*cos(x1)-5',
 'pow(x1,5)+pow(x1,4)+pow(x1,2)+x1',
 'exp(-1*pow(x1,2))',
 'pow(x1,8)+pow(x1,7)+pow(x1,6)+pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'exp(-0.5*pow(x1,2))',
 'div(1,(1+pow(x1,-4)))+div(1,(1+pow(x2,-4)))',
 'pow(x1,9)+pow(x1,8)+pow(x1,7)+pow(x1,6)+pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'x1*x2+x3*x4+x5*x6+x1*x7*x9+x3*x6*x8',
 'div(pow(x1+1,3),pow(x1,2)-x1+1)',
 'div((pow(x1,5)-3*pow(x1,3)+1),(pow(x1,2)+1))',
 'div((pow(x1,6)+pow(x1,5)),(pow(x1,4)+pow(x1,3)+pow(x1,2)+x1+1))',
 'div(pow(x1+1,3),pow(x1,2)-x1+1)',
 'div((pow(x1,5)-3*pow(x1,3)+1),(pow(x1,2)+1))',
 'div((pow(x1,6)+pow(x1,5)),(pow(x1,4)+pow(x1,3)+pow(x1,2)+x1+1))',
 'sin(x1)+sin(x1+pow(x1,2))',
 'div(exp(-pow(x1-1,2)),(1.2+pow((x2-2.5),2)))',
 'exp(-x1)*pow(x1,3)*cos(x1)*sin(x1)*(cos(x1)*pow(sin(x1),2)-1)',
 'exp(-x1)*pow(x1,3)*cos(x1)*sin(x1)*(cos(x1)*pow(sin(x1),2)-1)*(x2-5)',
 'div(10,(5+(pow((x1-3),2)+pow((x2-3),2)+pow((x3-3),2)+pow((x4-3),2)+pow((x5-3),2))))',
 '30*(x1-1)*div(x3-1,(x1-10)*pow(x2,2))',
 '6*sin(x1)*cos(x2)',
 '(x1-3)*(x2-3)+2*sin(x1-4)*(x2-4)',
 'div(pow((x1-3),4)+pow((x2-3),3)-(x2-3),pow((x2-2),4)+10)',
 '2.5*pow(x1,4)-1.3*pow(x1,3)+0.5*pow(x2,2)-1.7*x2',
 '8.0*pow(x1,2)+8.0*pow(x2,3)-15.0',
 '0.2*pow(x1,3)+0.5*pow(x2,3)-1.2*x2-0.5*x1',
 '1.5*exp(x1)+5.0*cos(x2)',
 '6.0*sin(x1)*cos(x2)',
 '1.35*x1*x2+5.5*sin((x1-1.0)*(x2-1.0))',
 'pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'sin(pow(x1,2))*cos(x1)-1',
 'log(x1+1)+log(pow(x1,2)+1)',
 '2*sin(x1)*cos(x2)',
 '0.58 + log(x1) + 0.5/x1 - 1./(12*x1**2) + 1./(120*x1**4)',
 '2-2.1*cos(9.8*x1)*sin(1.3*x2)',
 'div(exp(-pow(x1-1,2)),(1.2+pow((x2-2.5),2)))',
 'div(1,(1+pow(x1,-4)))+div(1,(1+pow(x2,-4)))',
 '1./3+x1+sin(pow(x1,2))',
 '3.14*x1*x1']
benchmark_expr_list = list(set(expr_list))

def log(x):
    if x == 0:
        return 0
    else:
        return np.log(np.abs(x))

def sqrt(x):
    return x**0.5

def div(x, y):
    if y == 0:
        return 0
    else:
        return x / y

def randomly_labeled_benchmark_expr(benchmark_expr_list, n_vars=2):
    random_expr = np.random.choice(benchmark_expr_list).copy()
    for source_var_idx in range(1, 11):
        target_var_idx = np.random.randint(1, n_vars + 1)
        random_expr = random_expr.replace(f'x{source_var_idx}', f'x{target_var_idx}')
    return random_expr

def create_crossover_prompt(pop, examples=7):
    prompt = "Below are 10 expressions that approximate the dataset:\n"
    parents = np.random.choice(pop, size=examples, replace=False)
    for parent in parents:
        prompt += parent + '\n'
    return prompt

def do_crossover(pop, generator, examples=7, temp=0.8, text_length=500):
    prompt = create_crossover_prompt(pop, examples=examples)
    model_output = generator(prompt,
                             do_sample=True,
                             min_length=text_length,
                             max_length=text_length,
                             temperature=temp,
                             return_full_text=False)
    return model_output[0]['generated_text']

def process_output(output, n_take=3): # Assumes 2 variables; todo: generalize
    lines = output.split('\n')
    candidates = []
    for line in lines:
        if ('x1' in line) or ('x2' in line):
            candidate = line.strip().strip('+').lower().replace('^', '**')
            try:
                simp_candidate = str(parse_expr(candidate))
                candidates.append(simp_candidate)
            except:
                pass
        if len(candidates) == n_take:
            break
    return candidates

def compute_fitness(eq, X, true_Y): # Assumes 2 variables; todo: generalize
    try:
        pred_Y = []
        for i in range(X.shape[0]):
            x1, x2 = X[i]
            y = eval(eq).real
            pred_Y.append(y)
        return r2_score(true_Y, pred_Y)
    except:
        return None

# Load dataset
if dataset_name == 'banana':
    dataset_file_path = 'pmlb/datasets/banana/banana.tsv.gz'
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
                                                    random_state=random_states[split_idx])
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
y_test_scaled = sc_y.fit_transform(y_test.reshape(-1,1)).flatten()

# Load generator
assert lm_name in ['galactica', 'pythia']
if lm_name == 'galactica':
    lm_tag = 'facebook/galactica-1.3b'
elif lm_name == 'pythia':
    lm_tag = 'EleutherAI/pythia-1.4b-deduped'

generator13b = pipeline('text-generation',
                        model='facebook/galactica-1.3b',
                        device=device_idx)

# Set up results file
results_file = f'results/{dataset_name}_{lm_name}_split{split_idx}_{exp_id}_results.csv'

#
# Run core evolution loop on training set
#

pop_size = 50
max_generations = args.max_generations
random_candidate_prob = 0.05
initial_pop_size = 1000

# Initialize population
pop = []
fit = []
while len(pop) < initial_pop_size:
    candidate = randomly_labeled_benchmark_expr(benchmark_expr_list)
    fitness = compute_fitness(candidate, X_train_scaled, y_train_scaled)
    if fitness is not None:
        pop.append(candidate)
        fit.append(fitness)

med_fit_chart = []
avg_fit_chart = []
max_fit_chart = []
max_fit_cand = []
test_fit_chart = []

for gen in range(max_generations):

    # Update stats
    avg_fit = sum(fit) / pop_size
    max_fit = max(fit)
    med_fit = np.median(fit)

    avg_fit_chart.append(avg_fit)
    max_fit_chart.append(max_fit)
    med_fit_chart.append(med_fit)

    print('gen ', gen, len(pop))
    print('avg fit ', avg_fit, 'max fit', max_fit, 'med fit', med_fit)
    best_cand = pop[fit.index(max_fit)]
    print(best_cand)
    max_fit_cand.append(best_cand)
    test_fit = compute_fitness(best_cand, X_test_scaled, y_test_scaled)
    test_fit_chart.append(test_fit)

    # Update resultls file
    results_df = pd.DataFrame({
            'best_expr': max_fit_cand,
            'test_fitness': test_fit_chart,
            'max_fitness': max_fit_chart,
            'mean_fitness': avg_fit_chart,
            'median_fitness': med_fit_chart
        })

    results_df.to_csv(results_file)

    # Create offspring
    off_pop = []
    off_fit = []
    while len(off_pop) < pop_size:
        if np.random.random() < random_candidate_prob:
            candidate = randomly_labeled_benchmark_expr(benchmark_expr_list)
            if (candidate not in off_pop) and (candidate not in pop):
                fitness = compute_fitness(candidate, X_train_scaled, y_train_scaled)
                if fitness is not None:
                    off_pop.append(candidate)
                    off_fit.append(fitness)
        else:
            out = do_crossover(pop, generator13b)
            candidates = process_output(out)
            for candidate in candidates:
                if (candidate not in off_pop) and (candidate not in pop):
                    fitness = compute_fitness(candidate, X_train_scaled, y_train_scaled)
                    if fitness is not None:
                        off_pop.append(candidate)
                        off_fit.append(fitness)

    merged_pop = off_pop + pop
    merged_fit = off_fit + fit

    # Do tournament selection to get back down to pop-size
    while len(merged_pop) > pop_size:
        c1i, c2i = np.random.choice(np.arange(len(merged_pop)),
                                    size=2, replace=False)
        if merged_fit[c1i] > merged_fit[c2i]:
            idx_to_delete = c2i
        else:
            idx_to_delete = c1i

        del merged_pop[idx_to_delete]
        del merged_fit[idx_to_delete]

    pop = merged_pop
    fit = merged_fit
    for p, f in zip(pop, fit):
        print(f, p)
