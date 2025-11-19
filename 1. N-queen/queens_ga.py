# queens_ga.py
import numpy as np
import pprint
from individual import Individual
from crossover_functions import crossover_cut_and_fill, crossover_pmx, crossover_n_cut
from mutation_functions import mutate_swap, mutate_bitwise
from survival_functions import fitness_based_replacement, generational_replacement, elitism_replacement
from utility_functions import is_valid_chromosome, create_individual, repair_chromosome

# ############################### QueensGA Class ###############################


class QueensGA:
    def __init__(self, n=8, pop_size=100, mutation_prob=0.5,
                 recombination_rate=1.0, max_evaluations=10000,
                 crossover_type='default', mutation_type="default", survival_strategy="default"):
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
        self._crossover_type = crossover_type
        self._mutation_type = mutation_type
        self._survival_strategy = survival_strategy

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

        if not is_valid_chromosome(chromosome_1, self._n) or not is_valid_chromosome(chromosome_2, self._n):
            try:
                chromosome_1 = repair_chromosome(chromosome_1)
                chromosome_2 = repair_chromosome(chromosome_2)
            except:
                raise ValueError(
                    "Children creation failed! Child(ren) is/are NOT valid")

        child1 = create_individual(
            chromosome_1, generationBirth=current_gen)
        child2 = create_individual(
            chromosome_2, generationBirth=current_gen)

        return child1, child2

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

        if not is_valid_chromosome(mutated_chromosome, self._n):
            try:
                mutated_individual = repair_chromosome(mutated_chromosome)
            except:
                raise ValueError(
                    "Mutation failed! Mutated chromosome is NOT valid and couldn't repair it."
                )

        mutated_individual = individual
        mutated_individual.set_chromosome(mutated_chromosome)
        return mutated_individual

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
                new_pop = generational_replacement(self, offsprings)

            case "elitism":
                # Elitism: keep n_elite best
                new_pop = elitism_replacement(
                    self, offsprings, population_fitness, n_elite)

            case _:
                raise ValueError(
                    f"Unknown survival strategy: '{strategy}'. "
                    f"Available: generational, elitism"
                )

        return new_pop

    def run_ga(self):
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
                best_fitness_history = self.get_best_fitness_history()
                avg_fitness_history = self.get_avg_fitness_history()

                best_id = max(population_fitness, key=lambda x: x[1])[0]
                best_individual = self._pop[best_id]
                print(f"\nTerminated at generation {generation}")
                print(f"Best fitness: {best_fitness}")
                print(f"Total evaluations: {self._evaluations}")
                return best_individual, best_fitness, generation, best_fitness_history, avg_fitness_history

            parent1, parent2 = self.select_parents()

            child1, child2 = self.recombine(
                parent1, parent2, combinatorName=self._crossover_type)

            child1 = self.mutate(child1, mutatorName=self._mutation_type)
            child2 = self.mutate(child2, mutatorName=self._mutation_type)

            offsprings = [child1, child2]
            new_pop = self.select_survivors(
                offsprings, strategy=self._survival_strategy)
            self.set_pop(new_pop)
            self.increment_generation()

    # ----------------------------- Termination --------------------------------

    def check_termination(self, best_fitness):
        if best_fitness == 0 or self._evaluations >= self._max_evaluations:
            return True
        return False
