import csv
import itertools
import time
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from sudoku_solver import SudokuGA
import random
from collections import defaultdict


def parse_puzzles(txt_path):
    with open(txt_path, 'r') as f:
        puzzles = [line.strip() for line in f if line.strip()]

    def parse_puzzle(puzzle):
        return np.array([[0 if ch == '.' else int(ch) for ch in puzzle[i:i + 9]] for i in range(0, 81, 9)])

    return [parse_puzzle(puzzle) for puzzle in puzzles]


def run_single_simulation(args):
    puzzle_idx, puzzle, params = args
    ga = SudokuGA(puzzle,
                  population_size=params["population_size"],
                  mutation_rate=params["mutation_rate"],
                  generations=params["generations"],
                  selection_rate=params["selection_rate"],
                  # random_selection_rate=params["random_selection_rate"],
                  nb_children=params["nb_children"],
                  restart_after_n_stagnant=params["restart_after_n_stagnant"])
    start_time = time.time()
    solution = ga.solve()
    duration = time.time() - start_time
    fitness = ga.fitness(solution) if solution is not None else -1
    return {
        "puzzle_id": puzzle_idx,
        "solved": int(fitness == 0),
        "final_fitness": fitness,
        "time_seconds": round(duration, 2),
        "generations_run": getattr(ga, "generations_run", -1),
        **params
    }


def run_full_simulation(puzzle_file, output_csv):
    # Load puzzles
    puzzles = parse_puzzles(puzzle_file)
    print(f"Loaded {len(puzzles)} puzzles.")

    # Define base parameter grids
    population_sizes = [5000]
    mutation_rates = [0.1, 0.2, 0.3, 0.4]
    generations_list = [1000]
    selection_rates = [0.1, 0.2, 0.4, 0.5]
    # random_selection_rates = [0.1, 0.2, 0.3, 0.4]

    # Generate valid param combinations while ensuring (sel + rand)/2 * nb_children ≈ 1
    param_combinations = []
    for sel_rate in selection_rates:
        average_pairing = sel_rate / 2
        nb_children = int(round(1 / average_pairing))
        pairing_test = average_pairing * nb_children
        if not (0.99 <= pairing_test <= 1.01):
            continue

        for pop_size in population_sizes:
            for mut_rate in mutation_rates:
                for gen in generations_list:
                    param_dict = {
                        "population_size": pop_size,
                        "mutation_rate": mut_rate,
                        "generations": gen,
                        "selection_rate": sel_rate,
                        "nb_children": nb_children,
                        "restart_after_n_stagnant": 150
                    }
                    param_combinations.append(param_dict)

    # Target number of runs per param combo
    runs_per_combo = 10

    # Prepare sampled run list
    all_args = []
    combo_counter = defaultdict(int)

    # Seed for reproducibility
    random.seed(42)

    while len(all_args) < runs_per_combo * len(param_combinations):
        params = random.choice(param_combinations)
        if combo_counter[tuple(sorted(params.items()))] >= runs_per_combo:
            continue  # already hit 10 runs for this param combo

        puzzle_idx = random.randint(0, len(puzzles) - 1)
        puzzle = puzzles[puzzle_idx]

        all_args.append((puzzle_idx, puzzle, params))
        combo_counter[tuple(sorted(params.items()))] += 1

    print(f"Running {len(all_args)} simulations...")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        param_keys = list(param_combinations[0].keys()) if param_combinations else []
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ["puzzle_id", "solved", "final_fitness", "time_seconds", "generations_run"] + param_keys
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in tqdm(pool.imap_unordered(run_single_simulation, all_args), total=len(all_args)):
                writer.writerow(result)


if __name__ == "__main__":
    run_full_simulation("puzzles/hard95.txt", "hard95_improved_simulation_results.csv")
