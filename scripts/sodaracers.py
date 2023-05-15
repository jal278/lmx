import functools
import json
import os
import time
from itertools import permutations
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from openelm.codegen.codegen_utilities import sample, truncate
from openelm.constants import SRC_PATH
from openelm.environments.environments import Sodaracer
from openelm.environments.sodaracer.walker import Walker
from openelm.map_elites import Map
from openelm.utils.code_eval import pool_exec_processes

CIRCLE = """
def make_circle(wc, cx, cy, radius, num_points):
    \"\"\"Approximate a circle with center (cx,cy) square with num_points points.\"\"\"
    joints = []

    tot_ang = 3.14 * 2.0

    for idx in range(num_points):
        ang = (tot_ang / num_points) * idx
        x = math.cos(ang) * radius + cx
        y = math.sin(ang) * radius + cy
        joints.append(wc.add_joint(x, y))

    return joints

"""

RADIAL = """
def make_walker():
    \"\"\"Create a radial walker.\"\"\"
    wc = walker_creator()

    num_points = 8
    rad = 5.0
    cx, cy = (5, 5)
    # the main body is a square
    points = make_circle(wc, cx, cy, rad, num_points)
    center = wc.add_joint(cx, cy)

    for k in range(num_points):
        wc.add_muscle(points[k], points[(k + 1) % num_points])
        wc.add_muscle(points[k], center, float(k) / num_points, float(k) / num_points)

    return wc.get_walker()

"""

WHEEL = """
def make_walker():
    \"\"\"Create a wheel walker.\"\"\"
    wc = walker_creator()
    num_points = 8
    rad = 3.0
    cx, cy = (11, 5)
    points = make_circle(wc, 0.6, -0.5, rad / 2, num_points)
    center = wc.add_joint(cx + 1, cy + 1)
    for j in range(num_points):
        for i in range(num_points - 5):
            wc.add_muscle(points[j], points[(i + j) % num_points],
                          0.0, 1.0, (j + 1) / num_points)
        wc.add_muscle(points[j], center, 3, (j + 1) / num_points)
    return wc.get_walker()

"""

SQUARE_PREREQ = """
def make_square(wc, x0, y0, x1, y1):
    \"\"\"Make a square with top left x0,y0 and top right x1,y1.\"\"\"
    j0 = wc.add_joint(x0, y0)
    j1 = wc.add_joint(x0, y1)
    j2 = wc.add_joint(x1, y1)
    j3 = wc.add_joint(x1, y0)
    return j0, j1, j2, j3

"""

SQUARE = """
def make_walker():
    \"\"\"Create a square walker.\"\"\"
    wc = walker_creator()

    # the main body is a square
    sides = make_square(wc, 0, 0, 10, 10)
    center = wc.add_joint(5, 5)

    # connect the square with distance muscles
    for k in range(len(sides) - 1):
        wc.add_muscle(sides[k], sides[k + 1])
    wc.add_muscle(sides[3], sides[0])

    # one prong of the square is a distance muscle
    wc.add_muscle(sides[3], center)

    # the other prongs from the center of the square are active
    wc.add_muscle(sides[0], center, 5.0, 0.0)
    wc.add_muscle(sides[1], center, 10.0, 0.0)
    wc.add_muscle(sides[2], center, 2.0, 0.0)

    return wc.get_walker()

"""

GALLOPER_PREREQ = """
def make_sensor(wc, x0, y0, x1, y1, d):
    return (
        wc.add_joint(x0, y0),
        wc.add_joint(x1, y1),
        wc.add_joint(x1, y0),
        wc.add_joint(x0, y1),
        wc.add_joint(d, 0.5),
        wc.add_joint(x1, 0.5),
    )

"""

GALLOPER = """
def make_walker(
    dx=0.0,
    dy=0.0,
    ddr=0,
    ddc=1.6,
):
    wc = walker_creator()
    ends = [
        make_sensor(wc, 5 + dx, -1 + dy, ddr, ddc, 4.5),
        make_sensor(wc, 0, -0.1, sid, 9.5, 0.03),
        make_sensor(wc, 5.5, -0.001, 5.0, 4.86 + 0.8, 0.07),
        make_sensor(wc, 5.5, -3.0, 6.0, 4.86 + 0.8, 0.07),
        make_sensor(wc, 0, dx, ddr, ddc, 1.0),
    ]

    sides = ends[0] + ends[1] + ends[2] + ends[-1] + ends[-2] + ends[-3]

    center = wc.add_joint(dx, dy)

    # connect the square with distance muscles
    for k in range(len(sides) - 6):
        wc.add_muscle(sides[k], sides[k + 1], 30, 0.5)
    wc.add_muscle(sides[2], sides[4], 4.0, 0.8)
    for k in range(len(sides) - 2):
        wc.add_muscle(sides[k], sides[k + 2], 18.0, 60.0 / 5.5)

    for k in reversed(range(len(sides) - 6)):
        wc.add_muscle(sides[k], sides[k + 5], 4.0, 20.0 / 9.0)

    wc.add_muscle(center, sides[7], 2, 0, 90.0 / 9.0)
    return wc.get_walker()

"""

QUERY_CPPN = """
def query_cppn(wc, xgrid, ygrid, scale, connect_func, amp_func, phase_func):
    \"\"\"Create a grid of points and functionally connect them.\"\"\"
    joints = {}
    for x in range(xgrid):
        for y in range(ygrid):
            joints[(x, y)] = wc.add_joint(x * scale, y * scale)
    for x1 in range(xgrid):
        for y1 in range(ygrid):
            for x2 in range(x1, xgrid):
                for y2 in range(y1, ygrid):
                    if x1 == y1 and x2 == y2:
                        continue
                    if connect_func(x1, y1, x2, y2):
                        amp = amp_func(x1, y1, x2, y2)
                        phase = phase_func(x1, y1, x2, y2)
                        wc.add_muscle(joints[(x1, y1)], joints[(x2, y2)], amp, phase)
    return joints

"""

CPPN_FIXED = """
def make_walker():
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) > 4.5:
            return False
        return True

    def amp(x1, y1, x2, y2):
        return max(abs(x1 - x2), abs(y1 - y2))

    def phase(x1, y1, x2, y2):
        return np.sign(x1)

    _ = query_cppn(wc, 8, 3, 1.5, connect, amp, phase)

    return wc.get_walker()

"""

CPPN_MUTABLE = """
def make_walker():
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) > 4.5:
            return False
        return True

    def amp(x1, y1, x2, y2):
        return max(abs(x1 - x2), abs(y1 - y2))

    def phase(x1, y1, x2, y2):
        return x1 if x1 % 2 == 1 else -x1

    _ = query_cppn(wc, 8, 3, 1.5, connect, amp, phase)

    return wc.get_walker()

"""

RUNNER = """
def make_walker(p_scale=1):  # acrylic of current (m)
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if -2 * x1 + x2 * 2 > 2:
            return True
        return x1 <= abs(y1 - y2)

    def amp(x, y, x2, y2):
        return abs(x - x2) + abs(y - y2)

    def phase(x1, y1, x2, y2):
        return -x1 / 2 - math.cos(math.pi / 9)

    joints = query_cppn(wc, 5, 7 + p_scale, 2, connect, amp, phase)
    return wc.get_walker()

"""

IMPORTS = """
from openelm.environments.sodaracer.walker import walker_creator
import math

"""
# Test version without instruction.
# Try CodeGen 16B
INSTRUCTION_ONE = [
    "#Combine the ",
    "starting programs above to make a new program.\ndef make_walker():\n",
]

INSTRUCTION_TWO = [
    "#Create a new walker by modifying the starting function above.\ndef make_walker():\n"
]

INSTRUCTION_THREE = ["def make_walker():\n"]

INSTRUCTION_FOUR = [""]

SEEDS_DICT = {
    "wheel": WHEEL,
    "radial": RADIAL,
    "square": SQUARE,
    "cppn_fixed": CPPN_FIXED,
    "cppn_mutable": CPPN_MUTABLE,
    "galloper": GALLOPER,
    "runner": RUNNER,
}


class CrossoverBenchmark:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reverse_seeds = {v: k for k, v in SEEDS_DICT.items()}

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = torch.device("cuda" if cfg.cuda else "cpu")
        self.config = AutoConfig.from_pretrained(cfg.model)
        self.config.use_cache = True
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = 50256

        if cfg.fp16:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model, config=self.config
            ).to(self.device)

    def construct_prompt(self, seeds, instruction):
        prompt_str = IMPORTS
        seeds = [SEEDS_DICT[seed] for seed in seeds]
        # if SQUARE in seeds:
        #     prompt_str += SQUARE_PREREQ
        # if GALLOPER in seeds:
        #     prompt_str += GALLOPER_PREREQ
        if RADIAL in seeds or WHEEL in seeds:
            prompt_str += CIRCLE
        if CPPN_FIXED in seeds or CPPN_MUTABLE in seeds or RUNNER in seeds:
            prompt_str += QUERY_CPPN
        instruction_str = instruction[0]
        for seed in seeds:
            if seed == SQUARE:
                prompt_str += SQUARE_PREREQ
            if seed == GALLOPER:
                prompt_str += GALLOPER_PREREQ
            prompt_str += seed
            if instruction == INSTRUCTION_ONE:
                instruction_str += self.reverse_seeds[seed] + ", "
        if instruction == INSTRUCTION_ONE:
            instruction_str += instruction[-1]
        return prompt_str + instruction_str

    def to_mapindex(self, b, bins):
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, bins))
        )

    def benchmark_seeds(self, seed, instruction):
        prompt = self.construct_prompt(seed, instruction)
        encoding = self.tokenizer(
            [prompt],
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=2048,
        ).to(self.device)
        token_len = encoding.input_ids.shape[1]
        results, valid_fitnesses = [], []
        # Map setup
        n_bins = 12
        genotype_space = np.array([[0, 1000], [0, 1000], [0, 2000]]).T
        bins = np.linspace(*genotype_space, n_bins + 1)[1:-1].T  # type: ignore
        fitness_map = Map(
            dims=(n_bins,) * genotype_space.shape[1],
            fill_value=-np.inf,
            dtype=float,
        )
        print("Benchmarking seeds: ", ", ".join(seed))
        print("Prompt length: ", token_len, " tokens.")
        for _ in tqdm(range(self.cfg.n_trials // self.cfg.batch_size)):
            completions = sample(
                self.cfg,
                self.model,
                self.tokenizer,
                encoding,
                starting_idx=token_len - 1,
            )
            trunc = functools.partial(truncate, only_local_scope=True)
            truncations = list(prompt + trunc for trunc in map(trunc, completions))
            #            if seed == ('square'):
            # for i in range(4):
            #     print(truncations[i])
            #     print("--------------")
            #     print(completions[i])
            #     print("--------------")
            execution_results = pool_exec_processes(
                truncations,
                # func_name="make_walker",
                processes=self.cfg.processes,
                debug=self.cfg.debug,
            )
            for i, result in enumerate(execution_results):
                try:
                    if isinstance(result, Walker) and result.validate():
                        sodaracer = Sodaracer(
                            program_str=truncations[i],
                            result_obj=result.to_dict(),
                            error_code=0,
                        )
                        if sodaracer.valid:
                            fitness = sodaracer.evaluate(1000)
                            if fitness is not None:
                                valid_fitnesses.append(fitness)
                                map_idx = self.to_mapindex(
                                    sodaracer.to_phenotype(), bins=bins
                                )
                                if fitness > fitness_map[map_idx]:
                                    fitness_map[map_idx] = fitness
                                results.append(1)
                    else:
                        if self.cfg.debug:
                            print("Failed execution, type:", result)
                        results.append(result)
                except Exception as e:
                    if self.cfg.debug:
                        print(e, "Exception:")
                    results.append(6)
        valid_rate = (results.count(1) / len(results)) * 100
        avg_fitnesses = np.nanmean(valid_fitnesses)
        qd_score = fitness_map.qd_score
        niches_filled = fitness_map.niches_filled
        if len(valid_fitnesses) != results.count(1):
            print("Length mismatch ", len(valid_fitnesses), results.count(1))
        print(f"Valid rate for {seed}: {valid_rate}%")
        print(f"Average fitness: {avg_fitnesses}")
        print(f"QD score: {qd_score}")
        print(f"Niches filled: {niches_filled}")
        return valid_rate, valid_fitnesses, qd_score, niches_filled

    def run_benchmark(self):
        # instruction = INSTRUCTION_ONE
        if len(self.cfg.seeds) == 1:
            instruction = INSTRUCTION_TWO
        instruction = INSTRUCTION_FOUR
        perm = list(permutations(self.cfg.seeds))
        print("Permutations: ", perm)
        valid_rates, all_fitnesses, qd_scores, niches = [], {}, [], []
        for seeds in perm:
            valid_rate, fitnesses, qd_score, niches_filled = self.benchmark_seeds(
                seeds, instruction
            )
            valid_rates.append(valid_rate)
            all_fitnesses[", ".join(seeds)] = fitnesses
            qd_scores.append(qd_score)
            niches.append(niches_filled)
        valid_stats = (np.nanmean(valid_rates), np.nanstd(valid_rates))
        qd_stats = (np.nanmean(qd_scores), np.nanstd(qd_scores))
        niche_stats = (np.nanmean(niches), np.nanstd(niches))
        print(f"Validity stats: {valid_stats[0]:.2f}, {valid_stats[1]:.2f}")
        print(f"QD stats: {qd_stats[0]:.2f}, {qd_stats[1]:.2f}")
        print(f"Niche stats: {niche_stats[0]:.2f}, {niche_stats[1]:.2f}")
        results_dct = {
            "rates": valid_rates,
            "fitnesses": all_fitnesses,
            "qd_scores": qd_scores,
            "niches": niches,
            "valid_stats": valid_stats,
            "qd_stats": qd_stats,
            "niche_stats": niche_stats,
            "config": OmegaConf.to_container(self.cfg),
            "permutations": perm,
        }
        json_path = Path(
            "/fsx/home-hyperion/OpenELM/data", f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        json_path.write_text(json.dumps(results_dct))


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=str(SRC_PATH / "config"),
    config_name="benchmark_crossover_cfg",
    version_base="1.2",
)
def main(cfg):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------  End -----------------")

    crossover = CrossoverBenchmark(cfg)
    crossover.run_benchmark()


if __name__ == "__main__":
    main()
