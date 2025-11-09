import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

INITIAL_POP_SIZE = 100
VALID_PERM_VALUES = '01234567'


class QueensGA:
    def __init__(self, n=8, pop_size=100, mutation_prob=0.5,
                 recombination_rate=1.0, max_evaluations=10000):
        self._n = n
        self._pop_size = pop_size
        self._mutation_prob = mutation_prob
        self._recombination_rate = recombination_rate
        self._max_evaluations = max_evaluations
        self._evaluations = 0
        self._generation = 0
        self._best_fitness_history = []
        self._avg_fitness_history = []
        self._pop = {}

        # Parents
        self._parent1 = None
        self._parent2 = None

        # ############################### Initialization ###############################

    def initialize_population(self):
        def scramble_string(s):
            arr = list(s)
            random.shuffle(arr)
            return ''.join(arr)

        print("=" * 50)
        print("          INITIALIZING POPULATION")
        print("=" * 50)

        for i in range(self._pop_size):
            chromosome = scramble_string(VALID_PERM_VALUES)
            indiv = Individual(chromosome)
            # Key: indiv's ID, Value: indiv object
            self._pop[indiv.get_id()] = indiv

    def print_pop(self):
        for id, indiv in self._pop.items():
            pprint(f"Individual ID {id}: {vars(indiv)}")

    def get_pop_size(self):
        return self._pop_size

    def get_pop(self):
        return self._pop

    def get_parent1(self):
        return self._parent1

    def get_parent2(self):
        return self._parent2

    def set_pop(self, value):
        self._pop = value

    def get_mutation_prob(self):
        return self._mutation_prob

    def set_mutation_prob(self, value):
        self._mutation_prob = value

    def get_recombination_rate(self):
        return self._recombination_rate

    def set_recombination_rate(self, value):
        self._recombination_rate = value

    def get_max_evaluations(self):
        return self._max_evaluations

    def get_evaluations(self):
        return self._evaluations

    def increment_evaluations(self):
        self._evaluations += 1

    def get_generation(self):
        return self._generation

    def increment_generation(self):
        self._generation += 1

# ############################### Fitness Function ###############################

    def get_fitness(self, indiv):
        if indiv.is_fitness_calculated():
            return indiv.get_fitness()
        chromosome = indiv._chromosome
        fitness = 0
        for ith_queen in range(self._n):
            ith_queen_pos = int(chromosome[ith_queen])
            for j in range(ith_queen + 1, self._n):
                check_factor = (j - ith_queen)
                gene_to_compare = int(chromosome[j])
                if gene_to_compare == ith_queen_pos + check_factor or gene_to_compare == ith_queen_pos - check_factor:
                    fitness -= 1
        indiv.set_fitness(fitness)
        return fitness

    def get_population_fitness(self):
        fitness_pairs = []
        for id, indiv in self._pop.items():
            fitness = self.get_fitness(indiv)
            fitness_pairs.append((id, fitness))
        return fitness_pairs

    def get_best_fitness_history(self):
        return self._best_fitness_history

    def get_avg_fitness_history(self):
        return self._avg_fitness_history

# ############################### Parent Selection #############################

    def select_parents(self):
        all_ids = np.array(list(self._pop.keys()))

        # Randomly select 5 individuals
        if len(all_ids) < 5:
            selected_ids = all_ids
        else:
            # pick 5 unique IDs at random from all_ids
            selected_ids = np.random.choice(all_ids, size=5, replace=False)

        candidates = []
        for id in selected_ids:
            indiv = self._pop[id]
            fitness = self.get_fitness(indiv)
            candidates.append((id, indiv, fitness))

        # print unsorted candidates
        # for id, indiv, fitness in candidates:
        #     print(f"{id:>4}  {fitness:8.4f}  {indiv}")

        # Sort by fitness (descending - less negative is better)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # print sorted candidates
        # for id, indiv, fitness in candidates:
        #     print(f"{id:>4}  {fitness:8.4f}  {indiv}")

        # Select best 2
        self._parent1 = candidates[0][1]
        self._parent2 = candidates[1][1]

        return self._parent1, self._parent2
    # ############################### Termination ##################################

    def check_termination(generation, max_evaluations, best_fitness, n=8):
        """
        Check if termination condition is met.
        Stop if: solution found OR max evaluations reached.

        Args:
            generation: current generation number
            max_evaluations: maximum allowed evaluations
            best_fitness: best fitness in current population
            n: board size (to calculate max possible fitness)
        Returns:
            boolean (True if should terminate)
        """
        pass


# ############################### Representation ###############################


class Individual:
    # Class variable to track the next available ID
    _next_id = 0

    def __init__(self, chromosome, generationBirth=0):
        self._chromosome = chromosome
        self._age = 0
        self._generationBirth = generationBirth
        self._fitness_calculated = False
        self._fitness = None

        # Assign unique ID and increment the class counter
        self._id = Individual._next_id
        Individual._next_id += 1

    def get_chromosome(self):
        return self._chromosome

    def set_chromosome(self, value):
        self._chromosome = value
        self._fitness_calculated = False  # For chromosome changes

    def get_fitness(self):
        return self._fitness

    def set_fitness(self, fitness):
        self._fitness = fitness
        self._fitness_calculated = True

    def is_fitness_calculated(self):
        return self._fitness_calculated

    def get_age(self):
        return self._age

    def get_generation(self):
        return self._generationBirth

    def set_generation(self, value):
        self._generationBirth = value

    def increment_generation(self):
        self._generationBirth += 1

    def increment_age(self):
        self._age += 1

    def get_id(self):
        return self._id


# ############################### Crossover ####################################


def crossover_cut_and_fill(parent1, parent2):
    parent1_chromose = parent1.get_chromosome()
    parent2_chromose = parent2.get_chromosome()
    length = len(parent1_chromose)
    # visualize_chromosome(parent1, parent2)

    cut_point = np.random.randint(0, length)

    child1 = parent1_chromose[:cut_point]
    child2 = parent2_chromose[:cut_point]

    # --- Fill child1 ---
    used = set(child1)
    for gene in parent2_chromose:
        if gene not in used:
            child1 += gene
            if len(child1) == length:
                break

    # --- Fill child2 ---
    used = set(child2)
    for gene in parent1_chromose:
        if gene not in used:
            child2 += gene
            if len(child2) == length:
                break

    return child1, child2


def crossover_pmx(parent1, parent2):
    """
    Partially Mapped Crossover (PMX).
    Select two crossover points, map the segment, fill remaining positions.

    Args:
        parent1, parent2: parent chromosomes
    Returns:
        two offspring
    """
    pass


def crossover_n_cut(parent1, parent2, n_cuts=2):
    """
    N-cut crossover (generalization for 2-cut, 3-cut, etc.)

    Args:
        parent1, parent2: parent chromosomes
        n_cuts: number of cut points
    Returns:
        two offspring
    """
    pass

# ############################### Mutation #####################################


def mutate_swap(individual, mutation_prob):
    """
    Swap mutation: randomly swap two positions with given probability.

    Args:
        individual: chromosome to mutate
        mutation_prob: probability of mutation occurring
    Returns:
        mutated individual
    """
    pass


def mutate_bitwise(individual, mutation_prob):
    """
    Bitwise mutation: for each gene, with mutation_prob, change to random value.
    Must maintain permutation property.

    Args:
        individual: chromosome to mutate
        mutation_prob: probability per gene
    Returns:
        mutated individual
    """
    pass

# ############################### Survival Strategy ############################


def generational_replacement(population, offspring, fitnesses, offspring_fitnesses):
    """
    Generational replacement: entire population is replaced by offspring.

    Args:
        population: current population
        offspring: new offspring
        fitnesses: fitness of current population
        offspring_fitnesses: fitness of offspring
    Returns:
        new population, new fitnesses
    """
    pass


def elitism_replacement(population, offspring, fitnesses, offspring_fitnesses, n_elite=2):
    """
    Elitism: keep n_elite best individuals, rest are replaced.

    Args:
        population: current population
        offspring: new offspring
        fitnesses: fitness of current population
        offspring_fitnesses: fitness of offspring
        n_elite: number of elite individuals to preserve
    Returns:
        new population, new fitnesses
    """
    pass


# ############################### Main GA Loop #################################


def genetic_algorithm(pop_size=100, offspring_per_gen=2, mutation_prob=0.5,
                      crossover_prob=1.0, max_evaluations=10000, n=8,
                      crossover_type='cut_and_fill', mutation_type='swap',
                      survival_strategy='generational', n_elite=2):
    """
    Main genetic algorithm loop.

    Args:
        pop_size: population size
        offspring_per_gen: number of offspring to create per generation
        mutation_prob: probability of mutation
        crossover_prob: probability of crossover (recombination rate)
        max_evaluations: maximum fitness evaluations before stopping
        n: board size (8, 10, 12, 20...)
        crossover_type: 'cut_and_fill', 'pmx', '2_cut', '3_cut'
        mutation_type: 'swap' or 'bitwise'
        survival_strategy: 'generational' or 'elitism'
        n_elite: number of elite individuals (if using elitism)
    Returns:
        best_solution, best_fitness, generation_count, fitness_history
    """
    pass

# ############################### Utility Functions ############################


def is_valid_chromosome(chromosome, n=8):
    if len(chromosome) != n:
        raise ValueError(
            f"Invalid length: expected {n}, got {len(chromosome)}")
    if not chromosome.isdigit():
        raise ValueError("Chromosome contains non-digit characters")

    try:
        positions = [int(c) for c in chromosome]
    except ValueError:
        raise ValueError("Could not convert chromosome to integers")

    for pos in positions:
        if pos < 0 or pos >= n:
            raise ValueError(
                f"Invalid position {pos}: must be between 0 and {n-1}")
    if len(set(positions)) != n:
        raise ValueError("Chromosome has duplicate values")

    expected_set = set(range(n))
    actual_set = set(positions)
    if expected_set != actual_set:
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        raise ValueError(f"Missing values: {missing}, Extra values: {extra}")

    return True


def create_individual(chromosome, generationBirth):
    return Individual(chromosome, generationBirth)


def visualize_chessboard(individual):
    try:
        chromosome = individual.get_chromosome()
        n = len(chromosome)
        result_str = ""
        print("\nChessboard:")
        for i in range(n):
            index = chromosome.index(str(i))
            for j in range(n):
                if j != index:
                    result_str += ".\t"
                else:
                    result_str += "Q\t"
            result_str += "\n"
    except:
        print("Something went wrong when visualizing the chessboard.")
    else:
        print(result_str)


def visualize_chromosome(*individuals):
    for i, indiv in enumerate(individuals):
        chromosome = indiv.get_chromosome()
        fitness = indiv.get_fitness() if indiv.is_fitness_calculated() else "Not calculated"
        print(
            f'Individual {i + 1} (ID={indiv.get_id()}): \t {chromosome} \t Fitness: {fitness}')


def plot_convergence(fitness_history):
    """
    Plot fitness over generations.

    Args:
        fitness_history: list of best fitness per generation
    """
    pass


def run_experiment(num_runs=3, **ga_params):
    """
    Run GA multiple times and collect statistics.

    Args:
        num_runs: number of independent runs
        ga_params: parameters to pass to genetic_algorithm()
    Returns:
        results dictionary with statistics
    """
    pass


# ############################### Main Execution ###############################
if __name__ == "__main__":
    print("="*90)
    print("                              N-Queens Genetic Algorithm")
    print("="*90)
    queens_ga = QueensGA()
    queens_ga.initialize_population()
    parent1, parent2 = queens_ga.select_parents()
    visualize_chromosome(parent1)
    visualize_chessboard(parent1)
    visualize_chromosome(parent2)
    visualize_chessboard(parent2)
    child1, child2 = crossover_cut_and_fill(parent1, parent2)
    if (is_valid_chromosome(child1) and is_valid_chromosome(child2)):
        queens_ga.increment_generation()
        create_individual(child1, queens_ga.get_generation())
        create_individual(child2, queens_ga.get_generation())


# Task 1: Baseline implementation
# TODO: Run with baseline settings

# Task 2: Parameter sensitivity
# TODO: Test different mutation probabilities and recombination rates

# Task 3: Crossover strategy exploration
# TODO: Compare different crossover methods

# Task 4: Survival strategy comparison
# TODO: Compare generational vs elitism

# Task 5: Scalability study
# TODO: Test with N=10, 12, 20

pass
