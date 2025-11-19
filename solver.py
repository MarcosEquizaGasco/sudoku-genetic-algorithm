import csv
import time
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from sudoku_solver import SudokuGA


# Function to parse puzzle .txt files
def parse_puzzles(txt_path):
    with open(txt_path, 'r') as f:
        puzzles = [line.strip() for line in f if line.strip()]

    def parse_puzzle(puzzle):
        return np.array([[0 if ch == '.' else int(ch) for ch in puzzle[i:i + 9]] for i in range(0, 81, 9)])

    return [parse_puzzle(puzzle) for puzzle in puzzles]


# Function to run one simulation
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
        "time_seconds": round(duration, 2),
        "generations_run": getattr(ga, "generations_run", -1),
        "fitness": int(fitness)
    }


# Function so solve puzzle
def solve_all_with_params(puzzle_file, output_csv, params):
    puzzles = parse_puzzles(puzzle_file)
    print(f"Loaded {len(puzzles)} puzzles.")

    all_args = [(i, puzzle, params) for i, puzzle in enumerate(puzzles)]
    print(f"Running {len(all_args)} simulations...")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ["puzzle_id", "solved", "time_seconds", "generations_run", "fitness"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in tqdm(pool.imap_unordered(run_single_simulation, all_args), total=len(all_args)):
                writer.writerow(result)


if __name__ == "__main__":
    params = {
        "population_size": 5000,
        "mutation_rate": 0.3,
        "generations": 1000,
        "selection_rate": 0.5,
        # "random_selection_rate": 0.1,
        "nb_children": 4,
        "restart_after_n_stagnant": 150
    }

    solve_all_with_params("puzzles/hard95.txt", "hard95_improved_solved_results.csv", params)
