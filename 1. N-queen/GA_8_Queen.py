import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint


# ############################### Individual Class #############################

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


# ############################### QueensGA Class ###############################

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

    # ----------------------------- Getters/Setters ----------------------------

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

    def get_best_fitness_history(self):
        return self._best_fitness_history

    def get_best_fitness(self):
        if not self._best_fitness_history:
            return None
        return max(self._best_fitness_history)

    def append_to_fitness_history(self, value):
        self._best_fitness_history.append(value)

    def get_avg_fitness_history(self):
        return self._avg_fitness_history

    def append_to_avg_fitness_history(self, value):
        self._avg_fitness_history.append(value)

    # ----------------------------- Initialization -----------------------------

    def initialize_population(self):
        def scramble_array(arr):
            arr_copy = np.copy(arr)
            np.random.shuffle(arr_copy)
            return arr_copy

        print("=" * 50)
        print("          INITIALIZING POPULATION")
        print("=" * 50)

        for i in range(self._pop_size):
            chromosome = scramble_array(np.arange(self._n))
            indiv = Individual(chromosome)
            # Key: indiv's ID, Value: indiv object
            self._pop[indiv.get_id()] = indiv

    def print_pop(self):
        for id, indiv in self._pop.items():
            print(f"Individual ID [{id}]:")
            pprint.pp(vars(indiv), indent=5)

    # ----------------------------- Fitness Function ---------------------------

    def get_fitness(self, indiv):
        if indiv.is_fitness_calculated():
            return indiv.get_fitness()

        chromosome = indiv._chromosome
        fitness = 0

        chromosome = np.array(chromosome)

        for ith_queen in range(self._n):
            ith_queen_pos = int(chromosome[ith_queen])
            for j in range(ith_queen + 1, self._n):
                check_factor = j - ith_queen
                gene_to_compare = int(chromosome[j])
                if (
                    gene_to_compare == ith_queen_pos + check_factor
                    or gene_to_compare == ith_queen_pos - check_factor
                ):
                    fitness -= 1

        indiv.set_fitness(fitness)
        self._evaluations += 1
        return fitness

    def get_population_fitness(self):
        fitness_pairs = []
        for id, indiv in self._pop.items():
            fitness = self.get_fitness(indiv)
            fitness_pairs.append((id, fitness))
        return fitness_pairs

    # ----------------------------- Parent Selection ---------------------------

    def select_parents(self):
        all_ids = np.array(list(self._pop.keys()))

        # Randomly select 5 individuals
        if len(all_ids) < 5:
            selected_ids = all_ids
        else:
            # pick 5 unique IDs at random from all_ids
            selected_ids = np.random.choice(all_ids, size=5, replace=True)

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

    # ----------------------------- Recombination ------------------------------

    def recombine(self, parent1, parent2, combinatorName="default"):
        current_gen = self.get_generation()

        match combinatorName:
            case "default" | "cut_and_fill":
                chromosome_1, chromosome_2 = crossover_cut_and_fill(
                    parent1, parent2
                )
            case "pmx":
                chromosome_1, chromosome_2 = crossover_pmx(
                    parent1, parent2
                )
            case "2_cut":
                chromosome_1, chromosome_2 = crossover_n_cut(
                    parent1, parent2, n_cuts=2
                )
            case "3_cut":
                chromosome_1, chromosome_2 = crossover_n_cut(
                    parent1, parent2, n_cuts=3
                )
            case _:
                raise ValueError(
                    f"Unknown combinator name:   '{combinatorName}'. "
                    f"Available: default, cut_and_fill, pmx, 2_cut, 3_cut"
                )

        if is_valid_chromosome(chromosome_1, self._n) and is_valid_chromosome(chromosome_2, self._n):
            child1 = create_individual(
                chromosome_1, generationBirth=current_gen)
            child2 = create_individual(
                chromosome_2, generationBirth=current_gen)

            return child1, child2
        else:
            raise ValueError(
                "Children creation failed! Child(ren) is/are NOT valid")

    # ----------------------------- Mutation -----------------------------------

    def mutate(self, individual, mutatorName="default"):
        original_chromosome = individual.get_chromosome()

        match mutatorName:
            case "default" | "swap":
                mutated_chromosome = mutate_swap(
                    original_chromosome, self._mutation_prob
                )
            case "bitwise":
                mutated_chromosome = mutate_bitwise(
                    original_chromosome, self._mutation_prob
                )

            case _:
                raise ValueError(
                    f"Unknown mutator name: '{mutatorName}'. "
                    f"Available: default, swap, bitwise"
                )

        if is_valid_chromosome(mutated_chromosome, self._n):
            mutated_individual = individual
            mutated_individual.set_chromosome(mutated_chromosome)
            return mutated_individual
        else:
            raise ValueError(
                "Mutation failed! Mutated chromosome is NOT valid"
            )

    # ----------------------------- Survivor Selection -------------------------

    def select_survivors(self, offsprings, strategy="generational", n_elite=2):
        child1, child2 = offsprings[0], offsprings[1]
        population_fitness = self.get_population_fitness()
        child1_fitness = self.get_fitness(child1)
        child2_fitness = self.get_fitness(child2)

        match strategy:
            case "default" | "fitness_based":
                replaced_id1, replaced_id2 = fitness_based_replacement(
                    population_fitness,
                    child1_fitness,
                    child2_fitness
                )

                # Shallow copy of _pop
                new_pop = dict(self._pop)

                if replaced_id1 is not None:
                    del new_pop[replaced_id1]
                    new_pop[child1.get_id()] = child1

                if replaced_id2 is not None:
                    del new_pop[replaced_id2]
                    new_pop[child2.get_id()] = child2

            case "generational":
                new_pop = generational_replacement(self._pop)

            case "elitism":
                # Elitism: keep n_elite best
                new_pop = elitism_replacement(offsprings, n_elite)

            case _:
                raise ValueError(
                    f"Unknown survival strategy: '{strategy}'. "
                    f"Available: generational, elitism"
                )

        return new_pop

    def run_ga(self, crossover_type="default", mutation_type="default",
               survival_strategy="default"):
        while True:
            generation = self.get_generation()
            population_fitness = self.get_population_fitness()

            fitnesses = [fit for _, fit in population_fitness]
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)

            self.append_to_fitness_history(best_fitness)
            self.append_to_avg_fitness_history(
                {'generation': generation, 'Average-fitness': avg_fitness})

            if self.check_termination(best_fitness):
                best_fitness_history = self.get_best_fitness()
                avg_fitness_history = self.get_avg_fitness_history()

                best_id = max(population_fitness, key=lambda x: x[1])[0]
                best_individual = self._pop[best_id]
                print(f"\nTerminated at generation {generation}")
                print(f"Best fitness: {best_fitness}")
                print(f"Total evaluations: {self._evaluations}")
                return best_individual, best_fitness, generation, best_fitness_history, avg_fitness_history

            parent1, parent2 = self.select_parents()

            child1, child2 = self.recombine(
                parent1, parent2, combinatorName=crossover_type)

            child1 = self.mutate(child1, mutatorName=mutation_type)
            child2 = self.mutate(child2, mutatorName=mutation_type)

            offsprings = [child1, child2]
            new_pop = self.select_survivors(
                offsprings, strategy=survival_strategy)
            self.set_pop(new_pop)
            self.increment_generation()

    # ----------------------------- Termination --------------------------------

    def check_termination(self, best_fitness):
        if best_fitness == 0 or self._evaluations >= self._max_evaluations:
            return True
        return False


# ############################### Crossover Functions ##########################


def crossover_cut_and_fill(parent1, parent2):
    parent1_chromosome = np.asarray(parent1.get_chromosome())
    parent2_chromosome = np.asarray(parent2.get_chromosome())
    length = len(parent1_chromosome)
    # visualize_chromosome(parent1, parent2)

    cut_point = np.random.randint(0, length)

    child1 = parent1_chromosome[:cut_point]
    child2 = parent2_chromosome[:cut_point]

    # --- Fill child1 ---
    used = set(int(gene) for gene in child1)
    idx = cut_point
    while len(child1) < length:
        pos = idx % length
        gene = parent2_chromosome[pos]
        if gene not in used:
            child1 = np.append(child1, gene)
        idx += 1

    # --- Fill child2 ---
    used = set(int(gene) for gene in child2)
    idx = cut_point
    while len(child2) < length:
        pos = idx % length
        gene = parent1_chromosome[pos]
        if gene not in used:
            child2 = np.append(child2, gene)
        idx += 1

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


# ############################### Mutation Functions ###########################
# chromosome.copy() creates Deep Copy
def mutate_swap(chromosome, mutation_probability=0.1):
    if np.random.random() > mutation_probability:
        return chromosome.copy()

    chromosome_len = len(chromosome)
    if chromosome_len < 2:
        return chromosome.copy()

    i, j = np.random.choice(chromosome_len, size=2, replace=False)
    new_mutated_chromosome = chromosome.copy()
    new_mutated_chromosome[i], new_mutated_chromosome[j] = (
        new_mutated_chromosome[j], new_mutated_chromosome[i]
    )
    return new_mutated_chromosome


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


# ############################### Survival Strategy Functions ##################

def fitness_based_replacement(population_fitness, child1_fitness, child2_fitness):
    population_fitness.sort(key=lambda x: x[1], reverse=True)

    replaced_id1 = None
    replaced_id2 = None

    if child1_fitness >= child2_fitness:
        better_child_fitness = child1_fitness
        worse_child_fitness = child2_fitness
        first_is_child1 = True
    else:
        better_child_fitness = child2_fitness
        worse_child_fitness = child1_fitness
        first_is_child1 = False

    for id, fitness in population_fitness:
        if replaced_id1 is None and fitness < better_child_fitness:
            replaced_id1 = id
            continue

        if replaced_id2 is None and fitness < worse_child_fitness:
            replaced_id2 = id
            break

    if not first_is_child1:
        replaced_id1, replaced_id2 = replaced_id2, replaced_id1

    return replaced_id1, replaced_id2


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


# ############################### Utility Functions ############################
# Utility table print function


def print_summary_table(results):
    print("\n" + "="*90)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*90)

    # Header
    print(f"{'Run':<6} {'Success':<10} {'Generations':<15} {'Evaluations':<15} {'Best Fitness':<15} {'Solution':<20}")
    print("-"*90)

    # Data rows
    for r in results:
        success_str = "YES" if r['success'] else "NO"
        solution_str = r['best_individual'].get_chromosome(
        ) if r['success'] else "N/A"

        print(f"{r['run']:<6} {success_str:<10} {r['generations']:<15,} {r['evaluations']:<15,} {r['best_fitness']:<15,} {solution_str}")

    print("-"*90)

    # Statistics
    successes = sum(1 for r in results if r['success'])
    success_rate = (successes / len(results)) * 100
    avg_generations = sum(r['generations'] for r in results) / len(results)
    avg_evaluations = sum(r['evaluations'] for r in results) / len(results)

    print(f"\n{'STATISTICS':<20}")
    print(f"{'Success Rate:':<20} {successes:,}/{len(results):,} ({success_rate:.1f}%)")
    print(f"{'Avg Generations:':<20} {avg_generations:,.2f}")
    print(f"{'Avg Evaluations:':<20} {avg_evaluations:,.2f}")

    print("="*90)


def is_valid_chromosome(chromosome, n=8):
    if len(chromosome) != n:
        raise ValueError(
            f"Invalid length: expected {n}, got {len(chromosome)}")
    if not np.issubdtype(chromosome.dtype, np.integer):
        raise ValueError("chromosome must contain integers")

    positions = chromosome.astype(int)

    for pos in positions:
        if pos < 0 or pos >= n:
            raise ValueError(
                f"Invalid position {pos}: must be between 0 and {n-1}")
    if len(set(positions)) != n:
        raise ValueError("chromosome has duplicate values")

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
            index = int(np.where(chromosome == i)[0][0])
            for j in range(n):
                if j != index:
                    result_str += ".\t"
                else:
                    result_str += "Q\t"
            result_str += "\n"
    except Exception as e:
        print(f"Something went wrong when visualizing the chessboard: {e}")
    else:
        print(result_str)


def visualize_chromosome(*individuals):
    for i, indiv in enumerate(individuals):
        chromosome = indiv.get_chromosome()
        if isinstance(chromosome, np.ndarray):
            chromosome_str = " ".join(map(str, chromosome.tolist()))
        else:
            chromosome_str = str(chromosome)
        fitness = indiv.get_fitness() if indiv.is_fitness_calculated() else "Not calculated"
        print(
            f'Individual {i + 1} (ID={indiv.get_id()}): \t {chromosome_str} \t Fitness: {fitness}')


def plot_convergence(fitness_history):
    """
    Plot fitness over generations.

    Args:
        fitness_history: list of best fitness per generation
    """
    pass


def run_experiment(num_runs=3, **ga_params):
    results = []

    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{num_runs}")
        print(f"{'='*60}")

        queens_ga = QueensGA(**ga_params)
        queens_ga.initialize_population()
        best_individual, best_fitness, generations, fitness_history, avg_fitness_history = queens_ga.run_ga()

        results.append({
            'run': run + 1,
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'generations': generations,
            'evaluations': queens_ga.get_evaluations(),
            'fitness_history': fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'success': best_fitness == 0
        })

        print(f"Result: {'SUCCESS' if best_fitness == 0 else 'FAILED'}")
        if best_fitness == 0:
            print(f"Solution: {best_individual.get_chromosome()}")
        visualize_chessboard(best_individual)

    return results


# ############################### Main Execution ###############################
if __name__ == "__main__":
    print("="*90)
    print("                              N-Queens Genetic Algorithm")
    print("="*90)

    # Baseline implementation
    print("\n" + "="*60)
    print("Baseline implementation")
    print("="*60)
    print

    results = run_experiment(
        num_runs=100,
        n=20,
        pop_size=100,
        mutation_prob=0.5,
        recombination_rate=1.0,
        max_evaluations=10000
    )
    print_summary_table(results)


# Task 2: Parameter sensitivity
# TODO: Test different mutation probabilities and recombination rates

# Task 3: Crossover strategy exploration
# TODO: Compare different crossover methods

# Task 4: Survival strategy comparison
# TODO: Compare generational vs elitism

# Task 5: Scalability study
# TODO: Test with N=10, 12, 20

pass
