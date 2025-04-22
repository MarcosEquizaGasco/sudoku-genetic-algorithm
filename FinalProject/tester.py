import numpy as np
from sudoku_solver import SudokuGA
from pprint import pprint

puzzle = [
    [0, 2, 0, 0, 0, 0, 5, 0, 8],
    [9, 0, 5, 6, 0, 0, 3, 0, 2],
    [0, 8, 0, 2, 0, 0, 0, 0, 7],
    [0, 0, 0, 0, 7, 9, 6, 8, 5],
    [5, 9, 8, 1, 0, 2, 7, 0, 0],
    [0, 6, 4, 0, 0, 0, 2, 1, 0],
    [0, 0, 1, 3, 4, 5, 0, 0, 0],
    [0, 0, 9, 0, 0, 6, 8, 5, 1],
    [0, 0, 0, 0, 9, 1, 0, 0, 3],
]


solver = SudokuGA(puzzle, population_size=5000, generations=1000, restart_after_n_stagnant=150,
                  selection_rate=0.25, mutation_rate=0.25)
solution = solver.solve()
pprint(np.array(solution))


