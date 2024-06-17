# Evolve stable diffusion prompts using language model crossover (LMX)
# Mark Nelson, 2022-2023
# (main GA loop adapted from Elliot Meyerson)

# Paper:
#   Elliot Meyerson, Mark J. Nelson, Herbie Bradley, Arash Moradi, Amy K.
#     Hoover, Joel Lehman (2023). Language Model Crossover: Variation Through
#     Few-Shot Prompting. arXiv preprint. https://arxiv.org/abs/2302.12170

import pandas as pd
import numpy as np
import pickle
import os
import os.path
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm.auto import tqdm
import graphviz
import textwrap

max_prompt_tokens = 75  # a token is roughly 4 chars; SD prompts can be up to 75-77 tokens
sd_seed = 99 # make SD generate deterministically by using a fixed RNG seed

def get_seed_prompts(n, initprompts, rng_seed=None):
  seed_prompts = initprompts.sample(n=n, random_state=rng_seed)['Prompt'].tolist()
  return [p[:max_prompt_tokens*4] for p in seed_prompts]

# assumes seed_prompts have already been truncated to approx max_prompt_tokens
def new_prompt(seed_prompts, generator, use_prefix=True, use_header=False, color='g'):
  if use_prefix:
    seed_prompt = "\n".join(["Prompt: " + seed_prompt for seed_prompt in seed_prompts]) + "\nPrompt:"
  else:
    seed_prompt = "\n".join(seed_prompts) + "\n"

  if use_header:
    if color == 'g':
        color_name = 'green'
    elif color == 'r':
        color_name = 'red'
    elif color == 'b':
        color_name = 'blue'
    else:
        print("Invalid color", color)
        raise
  seed_prompt = f"List text-to-image prompts that generate images containing the most {color_name}:\n" + seed_prompt

  output = generator(
    seed_prompt,
    do_sample=True,
    temperature=0.9,
    max_new_tokens=max_prompt_tokens,
    return_full_text=False
  )

  # return only the first line, without leading/trailing whitespace
  return output[0]['generated_text'].partition('\n')[0].strip()

# One point crossover baseline
def one_point_crossover(parent_prompts):
    parent1, parent2 = parent_prompts[:2]
    parent_list1 = parent1.split()
    parent_list2 = parent2.split()
    split_idx = np.random.randint(len(parent_list1))
    child_list = parent_list1[:split_idx] + parent_list2[split_idx:]
    child = ' '.join(child_list)
    return child


def sd_generate(prompt, sd, num_inference_steps=50):
    torch.manual_seed(sd_seed)
    image = sd(prompt, num_inference_steps=num_inference_steps).images[0]
    return image

# fitness is "excess R/G/B" for one of R/G/B
def compute_fitness(image, rgb):
  img = np.asarray(image, dtype=np.float32)
  if rgb=="r":
    return np.sum(img[:,:,0]) - 0.5*np.sum(img[:,:,1]) - 0.5*np.sum(img[:,:,2])
  elif rgb=="g":
    return np.sum(img[:,:,1]) - 0.5*np.sum(img[:,:,0]) - 0.5*np.sum(img[:,:,2])
  # else "b"
  return np.sum(img[:,:,2]) - 0.5*np.sum(img[:,:,0]) - 0.5*np.sum(img[:,:,1])

def run_experiment(name,
                   fitness_fun,
                   pop_size,
                   max_generations,
                   num_parents_for_crossover,
                   baseline=False,
                   use_prefix=True,
                   use_header=False,
                   init_seed=9999, # initial population can have a noticeable impact on performance
                                   # ..so use the same initial pop when comparing hyperparameters
                   sd_inference_steps=10, # 50 is default, reduced for quicker iteration
                   color='g',
                   random_candidate_prob=0.05):

  # Initialize LLM
  llm = "EleutherAI/pythia-2.8b-deduped"

  # draw the initial population from a dataset of prompts from lexica.art:
  #   https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts
  initprompts = pd.read_parquet("train.parquet")

  # initialize the LLM
  generator = pipeline(task="text-generation", model=llm, device=0)

  # initialize SD
  sd = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)

  sd = sd.to("cuda")




  # add more generations to an existing run if {name}_pop.pickle exists
  restart = os.path.isfile(f"results/{name}_pop.pickle")
  if restart:
    with open(f"results/{name}_pop.pickle", "rb") as f:
      pop = pickle.load(f)
    with open(f"results/{name}_provenance.pickle", "rb") as f:
      provenance = pickle.load(f)
    results_df = pd.read_csv(f"results/{name}_results.csv")
    max_fit_cand = results_df['best_prompt'].tolist()
    max_fit_chart = results_df['max_fitness'].tolist()
    avg_fit_chart = results_df['mean_fitness'].tolist()
    med_fit_chart = results_df['median_fitness'].tolist()
  # otherwise, initialize from the human prompts dataset
  else:
    pop = get_seed_prompts(pop_size, initprompts, rng_seed=init_seed)
    # dict mapping prompt -> [parent_prompts]
    #  where a None value means the prompt is from the seed dataset, not LLM-generated
    provenance = {p: None for p in pop}
    med_fit_chart = []
    avg_fit_chart = []
    max_fit_chart = []
    max_fit_cand = []

  img = [sd_generate(p, sd, sd_inference_steps) for p in pop]
  fit = [fitness_fun(im) for im in img]

  for gen in tqdm(range(max_generations)):
    # Update stats
    avg_fit = sum(fit) / pop_size
    max_fit = max(fit)
    med_fit = np.median(fit)

    print('gen ', gen, len(pop))
    print('avg fit ', avg_fit, 'max fit', max_fit, 'med fit', med_fit)
    best_cand = pop[fit.index(max_fit)]

    for c, f in zip(pop, fit):
        print(f, c)
    print('best: ', best_cand)

    # skip storing this the first iteration when restarting a run, to avoid duplicates
    if not (restart and gen==0):
      avg_fit_chart.append(avg_fit)
      max_fit_chart.append(max_fit)
      med_fit_chart.append(med_fit)
      max_fit_cand.append(best_cand)

    # Update results file
    results_df = pd.DataFrame({
        'best_prompt': max_fit_cand,
        'max_fitness': max_fit_chart,
        'mean_fitness': avg_fit_chart,
        'median_fitness': med_fit_chart
        })
    results_df.to_csv(f"results/{name}_results.csv")

    # save the best image
    best_idx = np.argmax(fit)
    img[best_idx].save(f"results/{name}_best_{gen}.png")

    # Create offspring
    off_pop = []
    off_img = []
    off_fit = []
    while len(off_pop) < pop_size:
      if np.random.random() < random_candidate_prob:
        parents = None
        candidate = get_seed_prompts(1, initprompts)[0]
      else:
        parents = random.sample(pop, num_parents_for_crossover)
        if baseline:
            candidate = one_point_crossover(parents)
        else:
            candidate = new_prompt(parents, generator, use_prefix=use_prefix, use_header=use_header, color=color)

      if ((candidate not in off_pop) and (candidate not in pop)) or (num_parents_for_crossover == 0):
        image = sd_generate(candidate, sd, sd_inference_steps)
        fitness = fitness_fun(image)
        off_pop.append(candidate)
        off_img.append(image)
        off_fit.append(fitness)
        if candidate not in provenance:
          provenance[candidate] = parents

    merged_pop = off_pop + pop
    merged_img = off_img + img
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
      del merged_img[idx_to_delete]
      del merged_fit[idx_to_delete]
    pop = merged_pop
    img = merged_img
    fit = merged_fit

  # Update stats from the last iteration
  avg_fit = sum(fit) / pop_size
  max_fit = max(fit)
  med_fit = np.median(fit)

  best_cand = pop[fit.index(max_fit)]

  avg_fit_chart.append(avg_fit)
  max_fit_chart.append(max_fit)
  med_fit_chart.append(med_fit)
  max_fit_cand.append(best_cand)

  # Update results file
  results_df = pd.DataFrame({
      'best_prompt': max_fit_cand,
      'max_fitness': max_fit_chart,
      'mean_fitness': avg_fit_chart,
      'median_fitness': med_fit_chart
      })
  results_df.to_csv(f"results/{name}_results.csv")

  # save the best image
  best_idx = np.argmax(fit)
  img[best_idx].save(f"results/{name}_best.png")

  # save the population and provenance in case we want to run more generations
  with open(f"results/{name}_pop.pickle", "wb") as f:
    pickle.dump(pop, f, pickle.HIGHEST_PROTOCOL)
  with open(f"results/{name}_provenance.pickle", "wb") as f:
    pickle.dump(provenance, f, pickle.HIGHEST_PROTOCOL)

  return

  # draw a provenance graph
  wrapwidth = 25
  maxdepth = 5

  graph = graphviz.Digraph()
  visited = set()

  def label(p):
    return textwrap.fill(p, width=wrapwidth)

  def graph_parents(p, depth=0):
    prov = provenance[p]
    if prov:
      graph.node(label(p), style="filled", fillcolor="lightblue")
      for pr in prov:
        if pr not in visited:
          visited.add(pr)
          if depth < maxdepth:
            graph_parents(pr, depth+1)
        graph.edge(label(p), label(pr))
    else:
      graph.node(label(p))

  # Uncomment to draw graphs if graphviz is properly installed.
  #graph_parents(pop[best_idx])
  #graph.render(f"results/{name}_prompthist", format="pdf")

# experiments
if __name__ == "__main__":

    import argparse

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process some genetic algorithm parameters.")

    # Adding arguments
    parser.add_argument("--pop_size", type=int, default=20, help="Population size")
    parser.add_argument("--max_generations", type=int, default=100, help="Maximum number of generations")
    parser.add_argument("--num_parents_for_crossover", type=int, default=4, help="Number of parents for crossover")
    parser.add_argument("--baseline", action='store_true', help="Use baseline algorithm")
    parser.add_argument("--use_prefix", action='store_true', help="Use prefix for processing")
    parser.add_argument("--use_header", action='store_true', help="Use header at top of prompt")
    parser.add_argument("--random_seed", type=int, default=99, help="Random seed; use different ones for independent runs")
    parser.add_argument("--sd_steps", type=int, default=10, help="Num SD inference steps")
    parser.add_argument("--device", type=int, default=0, help="Index of GPU")
    parser.add_argument("--id", type=str, default='', help="Experiment id (can use to disambiguate)")
    parser.add_argument("--random_candidate_prob", type=float, default=0.05, help="Prob of random candidate for each individual")


    # Parse the arguments
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    for c in ("r", "g", "b"):
        exp_name = f"{args.id}excess_{c}_ps{args.pop_size}_np{args.num_parents_for_crossover}_bl{args.baseline}_pr{args.use_prefix}_rs{args.random_seed}_sd{args.sd_steps}_hd{args.use_header}_rcb{args.random_candidate_prob}"
        run_experiment(name=exp_name,
                       fitness_fun=lambda image: compute_fitness(image, c),
                       pop_size=args.pop_size,
                       max_generations=args.max_generations,
                       num_parents_for_crossover=args.num_parents_for_crossover,
                       baseline=args.baseline,
                       use_prefix=args.use_prefix,
                       use_header=args.use_header,
                       init_seed=args.random_seed,
                       sd_inference_steps=args.sd_steps,
                       color=c,
                       random_candidate_prob=args.random_candidate_prob)

    print("Done. Thank you.")
