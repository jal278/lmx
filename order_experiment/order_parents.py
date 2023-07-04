"""Experiment scripts to show that order of parents matters in LMX

- We generate a set of parents and then sort them in ascending, descending, or random order
- Run LMX on the sorted parents save the results (see lmx_bitstring.py)
- Plotting in notebook: (see plot_ordered_parents.ipynb)
- Batch run script in: (see run_sort_exp.sh)
"""

import numpy as np
from lmx_bitstring import LMX_bitstring
import json
import fire
from tqdm import tqdm

# -- Hyperparameters ------------------------------------------------------ -- #
POP_SIZE = 10
N_BITS   = 10
#REPS     = 100
REPS     = 2
lmx      = LMX_bitstring(max_length=256)


# -- Helper functions ----------------------------------------------------- -- #
def generate_population(pop_size=32, n_bits=N_BITS):
    return np.random.choice([0, 1], size=(pop_size, n_bits), p=[0.5, 0.5])

def evaluate_fitness(population):
    """Function to evaluate the fitness of a population of bitstrings for the OneMax problem."""
    return np.sum(population, axis=1)

def evaluate_leading_ones(population):
    """Function to evaluate the fitness of a population of bitstrings for the LeadingOnes problem."""
    def get_leading_ones(bitstring):
        """Function to count the number of ones before first zero in a bitstring."""
        if 0 not in bitstring:
            return np.sum(bitstring) # All ones
        return np.argmax(bitstring == 0)
    return np.array([get_leading_ones(bitstring) for bitstring in population])


# -- Experiment functions ------------------------------------------------- -- #
def experiment(parents, evaluate, parent_sorting='random'):
    """Run the LMX experiment with sorted parents."""
    if parent_sorting == 'ascending':
        sorted_indices = np.argsort(evaluate(parents))
    elif parent_sorting == 'descending':
        sorted_indices = np.argsort(evaluate(parents))[::-1]
    elif parent_sorting == 'random':
        sorted_indices = np.random.permutation(len(parents))
    else:
        raise ValueError("Invalid parent_sorting argument. Choose 'random', 'ascending', or 'descending'.")
    parents = parents[sorted_indices]
    
    # Apply LMX and return the fitness of the parents and children
    children = lmx(parents, n_children=POP_SIZE)
    return evaluate_fitness(children)

def run_sort_experiment(evaluate, output_file):
    """Main function to run the experiment multiple times and store results."""
    methods = ['random', 'ascending', 'descending']
    results = {method: [] for method in methods}
    parent_fitness = []
    
    # Identify the type of experiment being run based on the evaluate_fitness function
    experiment_name = "Fitness" if evaluate == evaluate_fitness else "Leading Ones"
    
    # Run the experiment multiple times
    for _ in tqdm(range(REPS), desc=f"Experiment Reps ({experiment_name})"):
        parents = generate_population(pop_size=POP_SIZE, n_bits=N_BITS)
        parent_fitness.append(evaluate(parents))

        for method in methods:
            child_fitness = experiment(parents, evaluate, parent_sorting=method)
            results[method].extend(child_fitness)
   
    # Save the results to a json file, convert to list for json serialization
    with open(output_file, 'w') as f:
        json.dump(results, f, default=lambda x: x.tolist())

class ExperimentRunner:
    def one_max(self):
        run_sort_experiment(evaluate_fitness, '_ordered_fitness.json')

    def leading_ones(self):
        run_sort_experiment(evaluate_leading_ones, '_ordered_ones.json')

def main():
    fire.Fire(ExperimentRunner)

if __name__ == "__main__":
    main()