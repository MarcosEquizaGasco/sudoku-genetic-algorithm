# Sudoku Solver Using a Genetic Algorithm  
A stochastic optimization approach to solving Sudoku puzzles using evolutionary search.

## Overview  
This project implements a Genetic Algorithm (GA) to solve Sudoku puzzles by treating the puzzle as a stochastic search and optimization problem. The solver evolves a population of candidate solutions using:

- Rank-based parent selection  
- Subgrid-wise crossover  
- Mutation via swapping non-fixed values  
- Stagnation-based restarts  

Two rounds of experiments were conducted:  
1. An elitist + random-selection approach  
2. A rank-based selection approach (final model)

The final implementation significantly improves efficiency and reduces convergence time.

---

## Features

### Genetic Algorithm Solver  
- Conflict-based fitness function  
- Rank-based parent selection  
- Subgrid-level crossover  
- Swap mutation restricted to non-fixed cells  
- Automatic restarts  
- Supports multiple puzzle formats and difficulties  

### Parallel Simulation Framework  
- Multiprocessing support  
- Random puzzle sampling  
- Logs results to CSV  
- Tracks: solved/unsolved, runtime, final fitness, generations run

### Puzzle Formats Supported  
Each line in a text file represents a puzzle:
- Using digits and zeros (4..5..7.)
- Or digits and dots (40040070)

---

## Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn tqdm
```

Python 3.10+ recommended.

---

## Usage
**Solving a Single Puzzle**

The `tester.py` file allows the user to input their unsolved puzzle, and running it will output the solved version of the sudoku.

**Running Multiple Simulations**

The `simulator.py` file allows the user to run multiple simulations of the genetic algorithm to solve multiple puzzles. 

**Analyzing Simulation Results**

After solving with multiple simulations, the results of each run can be analyzed with the Jupyter Notebooks in the `analysis` folder. These results include:

- Parameter effect plots

- Solved vs unsolved charts

- Percent-solved comparisons

- Runtime and generation analysis

---

## Genetic Algorithm Details

### Encoding and Initialization
- Candidate = 9×9 grid  
- Fixed cells are never changed  
- Each 3×3 subgrid is filled with digits 1–9 (minus the given clues), then randomly shuffled  

### Fitness Function
- Counts duplicate digits across rows and columns  
- **Fitness = number of conflicts** (lower is better; **0 = solved**)  

### Rank-Based Selection
- Individuals are sorted by fitness  
- Selection probability is proportional to rank  
- Preserves diversity and reduces premature convergence  

### Crossover
- Subgrid-wise crossover  
- A crossover point (0–8) is chosen  
- Subgrids before the point come from **Parent 1**, the remaining from **Parent 2**  

### Mutation
- Swap two mutable cells within a randomly chosen 3×3 subgrid  
- Fixed cells are never altered  

### Restarts
- If there is no improvement for **N generations**, the population is reinitialized  
- The best-so-far solution is preserved  

---

## CSV Output Format

Each row in the results CSV logs a single GA run:

- `puzzle_id`  
- `solved` (1 = solved, 0 = unsolved)  
- `final_fitness`  
- `time_seconds`  
- `generations_run`  
- `population_size`  
- `mutation_rate`  
- `selection_rate`  
- `nb_children`  
- `restart_after_n_stagnant`  

---

## References

Chambers, L., & Michalewicz, Z. (1995). *Genetic Algorithms for Permutation Problems.* In **Practical Handbook of Genetic Algorithms** (Vol. 1, pp. 123–148). CRC Press.

Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning.* Addison-Wesley.

Lewis, R. (2007). Metaheuristics can solve Sudoku puzzles. *Journal of Heuristics, 13*(4), 387–401.

Mantere, T., & Koljonen, J. (2007). Solving, rating, and generating Sudoku puzzles with GA. In **IEEE Congress on Evolutionary Computation** (pp. 1382–1389).

Palm, R. B., Paquet, U., & Winther, O. (2017). *Recurrent Relational Networks.* arXiv:1711.08028.

Russell, S. J., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

Whitley, D. (1994). A genetic algorithm tutorial. *Statistics and Computing, 4*(2), 65–85.

Yato, T., & Seta, T. (2003). Complexity and completeness of finding another solution and its application to puzzles. *IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences, E86-A*(5), 1052–1060.
