import numpy as np
import random
import matplotlib.pyplot as plt


# Class for Sudoku Genetic ALgorithm
class SudokuGA:
    def __init__(self, puzzle,
                 population_size=1000,
                 mutation_rate=0.25,
                 generations=1000,
                 selection_rate=0.25,
                 # random_selection_rate=0.25,
                 nb_children=4,
                 restart_after_n_stagnant=100):

        self.original = np.array(puzzle)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.selection_rate = selection_rate
        # self.random_selection_rate = random_selection_rate
        self.nb_children = nb_children
        self.restart_after_n_stagnant = restart_after_n_stagnant

        self.grid_size = 9
        self.subgrid_size = 3
        self.population = []

        # Fixed mask for non-mutable cells
        self.fixed_mask = self.original > 0

    def generate_candidate(self):
        """Generate a candidate with valid subgrids (3x3), may have row/column duplicates"""
        candidate = np.copy(self.original)
        for r in range(0, self.grid_size, 3):
            for c in range(0, self.grid_size, 3):
                block = candidate[r:r+3, c:c+3].flatten()
                fixed = set(block)
                missing = [n for n in range(1, 10) if n not in fixed]
                random.shuffle(missing)
                idx = 0
                for i in range(3):
                    for j in range(3):
                        if candidate[r+i, c+j] == 0:
                            candidate[r+i, c+j] = missing[idx]
                            idx += 1
        return candidate

    def initialize_population(self):
        self.population = [self.generate_candidate() for _ in range(self.population_size)]

    def fitness(self, grid):
        """Calculate fitness score"""
        duplicates = 0
        for i in range(self.grid_size):
            row_counts = np.bincount(grid[i], minlength=10)[1:]
            col_counts = np.bincount(grid[:, i], minlength=10)[1:]
            duplicates += sum(count - 1 for count in row_counts if count > 1)
            duplicates += sum(count - 1 for count in col_counts if count > 1)
        return duplicates
    '''
    def select_parents(self):
        # Top selection_rate % of population
        scored = [(self.fitness(ind), ind) for ind in self.population]
        scored.sort(key=lambda x: x[0])  # lower fitness is better
        top_n = int(self.selection_rate * self.population_size)
        selected = [x[1] for x in scored[:top_n]]

        # Randomly add some others for genetic diversity
        remaining = [x[1] for x in scored[top_n:]]
        random.shuffle(remaining)
        n_random = int(self.random_selection_rate * self.population_size)
        selected += remaining[:n_random]

        return selected

    '''
    def select_parents(self):
        """Select parents to create new grids"""
        # Rank-based selection: sort by fitness
        scored = [(self.fitness(ind), ind) for ind in self.population]
        scored.sort(key=lambda x: x[0])  # lower fitness is better
    
        # Assign ranks (best = rank 1)
        ranked_individuals = [x[1] for x in scored]
        ranks = list(range(len(ranked_individuals), 0, -1))  # N to 1
    
        # Normalize weights
        total_rank = sum(ranks)
        weights = [r / total_rank for r in ranks]
    
        # Select based on rank weights
        num_selected = int(self.selection_rate * self.population_size)
        selected_indices = np.random.choice(
            len(ranked_individuals),
            size=min(num_selected, len(ranked_individuals)),
            replace=False,
            p=weights
        )
        selected = [ranked_individuals[i] for i in selected_indices]

        return selected

    def crossover(self, p1, p2):
        """Crossover at the subgrid level using m randomly chosen grids from the mother."""
        child = np.copy(p1)

        # Randomly choose how many subgrids to take from the mother (from 1 to 8)
        m = random.randint(1, 8)

        # Randomly choose m grid indices to take from mother
        mother_grids = random.sample(range(9), m)

        # Subgrids are indexed from 0 to 8 (left-to-right, top-to-bottom)
        for idx in range(9):
            block_row = (idx // 3) * 3
            block_col = (idx % 3) * 3

            if idx in mother_grids:
                # Take this subgrid from the mother (p1)
                child[block_row:block_row + 3, block_col:block_col + 3] = p1[block_row:block_row + 3,
                                                                          block_col:block_col + 3]
            else:
                # Take this subgrid from the father (p2)
                child[block_row:block_row + 3, block_col:block_col + 3] = p2[block_row:block_row + 3,
                                                                          block_col:block_col + 3]

        '''
        # Calculate fitness score of child and reject if worse than the best parent
        if self.fitness(p1) <= self.fitness(p2):
            best_parent = p1
        else:
            best_parent = p2

        if self.fitness(child) <= self.fitness(best_parent):
            return child
        else:
            return best_parent
        '''

        return child

    def mutate(self, individual):
        """Mutate by swapping two non-fixed values within a subgrid"""
        mutated = np.copy(individual)
        if random.random() <= self.mutation_rate:
            r = random.randint(0, 2) * 3
            c = random.randint(0, 2) * 3

            cells = [(i, j) for i in range(r, r+3) for j in range(c, c+3)
                     if not self.fixed_mask[i, j]]
            if len(cells) >= 2:
                (i1, j1), (i2, j2) = random.sample(cells, 2)
                mutated[i1, j1], mutated[i2, j2] = mutated[i2, j2], mutated[i1, j1]
        return mutated

    def solve(self):
        """Solve the sudoku"""
        self.initialize_population()
        best_fitness = float('inf')
        best_grid = None
        stagnant = 0
        fitness_history = []
        local_best_fitness = float('inf')

        for gen in range(self.generations):

            # Find parents for next population, mutate them
            next_population = []
            parents = self.select_parents()
            next_population.extend([self.mutate(parent) for parent in parents])

            # Determine the best candidate and score
            scored = [(self.fitness(ind), ind) for ind in self.population]
            scored.sort(key=lambda x: x[0])
            best_candidate = scored[0][1]
            top_score = scored[0][0]
            fitness_history.append(top_score)

            # Compare current score to best local score for stagnation computation
            if top_score < local_best_fitness:
                local_best_fitness = top_score
                stagnant = 0
            else:
                stagnant += 1

            if top_score < best_fitness:
                best_fitness = top_score
                best_grid = best_candidate

            if best_fitness == 0:
                print(f"✅ Solved at generation {gen}")

                plt.plot(fitness_history)
                plt.title("Best Fitness Over Generations")
                plt.xlabel("Generation")
                plt.ylabel("Fitness")
                plt.savefig("fitness_progress.png")
                return best_grid

            # Restart population if stagnation occurs
            if stagnant >= self.restart_after_n_stagnant:
                # print(f"🔁 Restarting due to stagnation at generation {gen}")
                self.initialize_population()
                stagnant = 0
                local_best_fitness = float('inf')
                continue

            while len(next_population) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                for _ in range(self.nb_children):
                    child = self.crossover(p1, p2)
                    child = self.mutate(child)
                    next_population.append(child)
                    if len(next_population) >= self.population_size:
                        break

            self.population = next_population

            self.generations_run = gen + 1

            # if gen % 50 == 0:
                # print(f"Gen {gen}, Best fitness: {best_fitness}, Best local best fitness: {local_best_fitness}")

        print("No full solution found.")
        plt.plot(fitness_history)
        plt.title("Best Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.savefig("fitness_progress.png")
        self.best_fitness = best_fitness
        return best_grid
