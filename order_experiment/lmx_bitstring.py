import numpy as np
from transformers import pipeline
import re

class LMX_bitstring():
    def __init__(self, device=0, max_length=150):
        self.generator = init_model(device)
        self.max_length = max_length

    def __call__(self, parents, n_children=1):
        """Simple LMX recombination operator
        - Takes a population binary string parents
        - Computes probability vector from parents
        - Samples from probability vector to create children
        """
        # Compute probability vector
        string_parents = ["".join(map(str, row)) for row in parents]
        prompt = "\n".join(string_parents)

        # Sample from probability vector until you have enough valid children
        problem_size = len(parents[0])
        children = []
        while len(children) < n_children:
            model_output = self.generator([prompt],
                                        batch_size=1,
                                        do_sample=True,
                                        top_p=0.8,
                                        top_k=30,
                                        max_length=self.max_length,
                                        temperature=1.0,
                                        return_full_text=False,
                                        pad_token_id=self.generator.tokenizer.eos_token_id)
            gen_output = [x[0]['generated_text'] for x in model_output]    

            # Process output and extend children list
            for output in gen_output:
                processed_output = self.process_output(output, problem_size)
                children.extend(processed_output)
            #print(f"{len(children)}/{n_children} children generated")
            if len(children) >= n_children:
                children = children[:n_children]
        return self.text_to_genome(children)

    def process_output(self, output, n_vals, take_offspring=8):
        """ Process the text generator output, extract the top take_offspring number of child values, and return a list of candidates.
        """        
        genomes = output.strip().split("\n")
        candidates = [genome for genome in genomes[:take_offspring] if self.is_bitstring_list(genome, n_vals)]
        return list(set(candidates))

    @staticmethod
    def is_bitstring_list(s, n_vals, verbose=False):
        """Check if the input string is a valid bitstring."""
        if len(s) != n_vals:
            if verbose: print(s, "has the wrong length")
            return False
        if not re.match(r'^[01]+$', s):
            if verbose: print(s, "is not a valid bitstring")
            return False
        return True

    @staticmethod
    def text_to_genome(children):
        if isinstance(children, str):
            return np.array([[int(bit) for bit in children]])
        elif isinstance(children, list):
            return np.array([[int(bit) for bit in bitstring] for bitstring in children])
        else:
            raise ValueError("Invalid input type. Expected a single bitstring or a list of bitstrings.")

def init_model(model_name="EleutherAI/pythia-1.4b-deduped", device=0):
    """Initialize the model."""
    generator = pipeline('text-generation', model=model_name, device=device)
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
    return generator

def parents_to_prompt(parents):
    """Convert a list of bitstrings to a prompt."""
    string_parents = ["".join(map(str, row)) for row in parents]
    prompt = "\n".join(string_parents)
    return prompt