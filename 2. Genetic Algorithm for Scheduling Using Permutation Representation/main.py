# main.py (or whatever you named the main file)
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from individual import Individual
import pprint


class JobSchedulingGA:
    def __init__(self, processing_times, priorities, setup_times,
                 pop_size=50, max_generations=100, elitism=2, tournament_t=3, crossover_type='order', mutation_type='swap', mutation_rate=0.1):
        self._n_jobs = len(processing_times)
        self._processing_times = processing_times
        self._priorities = priorities
        self._setup_times = setup_times
        self._pop_size = pop_size
        self._max_generations = max_generations
        self._elitism = elitism
        self._tournament_t = tournament_t
        self._pop = {}
        self._crossover_type = crossover_type
        self._mutation_type = mutation_type
        self._mutation_rate = mutation_rate
        self._generation = 0

        self._parent1 = None
        self._parent2 = None

    def get_generation(self):
        return self._generation

    def get_pop(self):
        return self._pop

    def get_parent1(self):
        return self._parent1

    def get_parent2(self):
        return self._parent2

    def set_pop(self, value):
        self._pop = value

    def get_processing_times(self):
        return self._processing_times

    def set_processing_times(self, value):
        self._processing_times = value
        self._n_jobs = len(value)

    def get_priorities(self):
        return self._priorities

    def set_priorities(self, value):
        self._priorities = value

    def get_setup_times(self):
        return self._setup_times

    def set_setup_times(self, value):
        self._setup_times = value

    def get_pop_size(self):
        return self._pop_size

    def set_pop_size(self, value):
        self._pop_size = value

    def get_max_generations(self):
        return self._max_generations

    def set_max_generations(self, value):
        self._max_generations = value

    def get_elitism(self):
        return self._elitism

    def set_elitism(self, value):
        self._elitism = value

    def get_tournament_t(self):
        return self._tournament_t

    def set_tournament_t(self, value):
        self._tournament_t = value

    def get_n_jobs(self):
        return self._n_jobs

    def calculate_total_weighted_completion_time(self, sequence):
        completion_time = {}

        for idx, job in enumerate(sequence):
            if idx == 0:
                completion_time[job] = self._processing_times[job]
            else:
                previous_job = sequence[idx - 1]
                completion_time[job] = (completion_time[previous_job] +
                                        self._setup_times[previous_job][job] +
                                        self._processing_times[job])

        total = sum(self._priorities[job] * completion_time[job]
                    for job in sequence)
        return total

    def fitness(self, individual):
        sequence = individual.get_chromosome()
        twct = self.calculate_total_weighted_completion_time(sequence)
        return 1.0 / twct

    def create_population(self):
        print("          INITIALIZING POPULATION")

        for i in range(self._pop_size):
            chromosome = scramble_array(np.arange(self._n_jobs))
            indiv = Individual(chromosome)

            self._pop[indiv.get_id()] = indiv

    def print_pop(self):
        for id, indiv in self._pop.items():
            print(f"Individual ID [{id}]:")
            pprint.pp(vars(indiv), indent=5)

    def tournament_selection(self):
        all_ids = np.array(list(self._pop.keys()))
        if len(all_ids) < self._tournament_t:
            selected_ids = all_ids
        else:
            selected_ids = np.random.choice(
                all_ids, size=self._tournament_t, replace=False)

        best_fitness = -1
        best_indiv = None
        best_id = None

        for indiv_id in selected_ids:
            indiv = self._pop[indiv_id]
            fitness = self.fitness(indiv)
            if fitness > best_fitness:
                best_fitness = fitness
                best_indiv = indiv
                best_id = indiv_id

        return best_id, best_indiv

    def crossover_order(self, parent1, parent2):
        length = len(parent1)
        start_cut_point, end_cut_point = sorted(np.random.choice(
            range(length), size=2, replace=False))

        child = np.full(length, -1)
        child[start_cut_point:end_cut_point] = parent1[start_cut_point:end_cut_point]

        child_idx = parent_idx = end_cut_point
        filled = 0
        needed = length - (end_cut_point - start_cut_point)
        for i in range(length):
            if filled == needed:
                break
            elif parent2[parent_idx] not in child:
                child[child_idx] = parent2[parent_idx]
                child_idx = (child_idx + 1) % length
                filled += 1
            parent_idx = (parent_idx + 1) % length
        return child

    def crossover_pmx(self, parent1, parent2):
        def _create_offspring_pmx(p1, p2, point1, point2):
            size = len(p1)

            child = np.full(size, -1)

            child[point1:point2+1] = p1[point1:point2+1]

            for i in range(point1, point2+1):
                val = p2[i]

                if val not in child[point1:point2+1]:
                    j = child[i]
                    pos_j_in_p2 = np.where(p2 == j)[0][0]

                    while child[pos_j_in_p2] != -1:
                        k = child[pos_j_in_p2]
                        pos_j_in_p2 = np.where(p2 == k)[0][0]

                    child[pos_j_in_p2] = val

            for i in range(size):
                if child[i] == -1:
                    child[i] = p2[i]
            return child
        parent1_chromosome = np.asarray(parent1.get_chromosome())
        parent2_chromosome = np.asarray(parent2.get_chromosome())
        length = len(parent1_chromosome)

        point1, point2 = np.random.choice(
            range(length), size=2, replace=False)

        if point1 > point2:
            point1, point2 = point2, point1

        child1 = _create_offspring_pmx(
            parent1_chromosome, parent2_chromosome, point1, point2)

        child2 = _create_offspring_pmx(
            parent2_chromosome, parent1_chromosome, point1, point2)

        return child1, child2

    def crossover_cycle(self, parent1, parent2):
        length = len(parent1)

        child1 = np.full(length, -1)
        child2 = np.full(length, -1)

        patterns = detect_cycle_patterns(parent1, parent2)

        for cycle_num, (genes, indices) in enumerate(patterns):
            for position, idx in enumerate(indices):
                if cycle_num % 2 == 0:
                    child1[idx] = parent1[idx]
                    child2[idx] = parent2[idx]
                else:
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]

        return child1, child2

    def recombine(self, parent1, parent2, combinatorName='order'):
        p1_chr = parent1.get_chromosome()
        p2_chr = parent2.get_chromosome()

        current_gen = self.get_generation()

        match combinatorName:
            case "order":
                child1_chr = self.crossover_order(p1_chr, p2_chr)
                child2_chr = self.crossover_order(p2_chr, p1_chr)
            case "pmx":
                child1_chr, child2_chr = self.crossover_pmx(parent1, parent2)
            case "cycle":
                child1_chr, child2_chr = self.crossover_cycle(p1_chr, p2_chr)
            case _:
                raise ValueError(
                    f"Unknown crossover type: '{combinatorName}'. "
                    f"Available: order, pmx, cycle"
                )

        child1 = Individual(child1_chr, generationBirth=current_gen)
        child2 = Individual(child2_chr, generationBirth=current_gen)

        return child1, child2

    def mutate_swap(self, chromosome):
        chromosome_len = len(chromosome)
        if chromosome_len < 2:
            return chromosome.copy()

        i, j = np.random.choice(chromosome_len, size=2, replace=False)
        new_mutated_chromosome = chromosome.copy()
        new_mutated_chromosome[i], new_mutated_chromosome[j] = (
            new_mutated_chromosome[j], new_mutated_chromosome[i]
        )
        return new_mutated_chromosome

    def mutate_inversion(self, chromosome):
        mutated_chromosome = chromosome.copy()
        chromosome_len = len(chromosome)
        if chromosome_len < 2:
            return mutated_chromosome
        start_cut_point, end_cut_point = sorted(np.random.choice(
            range(chromosome_len), size=2, replace=False))

        to_invert = mutated_chromosome[start_cut_point:end_cut_point]
        mutated_chromosome[start_cut_point:end_cut_point] = list(
            reversed(to_invert))
        return mutated_chromosome

    def mutate_scramble(self, chromosome):
        mutated_chromosome = chromosome.copy()
        chromosome_len = len(chromosome)
        if chromosome_len < 2:
            return mutated_chromosome
        start_cut_point, end_cut_point = sorted(np.random.choice(
            range(chromosome_len), size=2, replace=False))
        to_scramble = mutated_chromosome[start_cut_point:end_cut_point]
        mutated_chromosome[start_cut_point:end_cut_point] = scramble_array(
            to_scramble)
        return mutated_chromosome

    def mutate(self, individual, mutatorName='swap'):

        if np.random.random() > self._mutation_rate:
            return individual

        original_chromosome = individual.get_chromosome()

        match mutatorName:
            case "swap":
                mutated_chromosome = self.mutate_swap(original_chromosome)
            case "inversion":
                mutated_chromosome = self.mutate_inversion(original_chromosome)
            case "scramble":
                mutated_chromosome = self.mutate_scramble(original_chromosome)
            case _:
                raise ValueError(
                    f"Unknown mutation type: '{mutatorName}'. "
                    f"Available: swap, inversion, scramble"
                )

        mutated_individual = individual
        mutated_individual.set_chromosome(mutated_chromosome)
        return mutated_individual

    def select_parents(self):
        _, parent1 = self.tournament_selection()
        _, parent2 = self.tournament_selection()
        self._parent1 = parent1
        self._parent2 = parent2
        return parent1, parent2

    def run_ga(self):
        self.create_population()
        self._generation = 0

        best_fitness_history = []
        avg_fitness_history = []

        for generation in range(self._max_generations):
            population_fitness = []
            for indiv_id, indiv in self._pop.items():
                fit = self.fitness(indiv)
                indiv.set_fitness(fit)
                population_fitness.append((indiv_id, fit))

            fitnesses = [fit for _, fit in population_fitness]
            best_fitness_history.append(max(fitnesses))
            avg_fitness_history.append(np.mean(fitnesses))

            parent1, parent2 = self.select_parents()
            child1, child2 = self.recombine(
                parent1, parent2, self._crossover_type)
            child1 = self.mutate(child1, self._mutation_type)
            child2 = self.mutate(child2, self._mutation_type)

            self._pop = elitism_replacement(
                self, (child1, child2), population_fitness, self._elitism
            )

            self._generation += 1

        final_fitness = [(indiv_id, self.fitness(indiv))
                         for indiv_id, indiv in self._pop.items()]
        best_id = max(final_fitness, key=lambda x: x[1])[0]
        best_individual = self._pop[best_id]

        return {
            'best_individual': best_individual,
            'best_fitness': self.fitness(best_individual),
            'best_chromosome': best_individual.get_chromosome(),
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history
        }


def scramble_array(arr):
    arr_copy = np.copy(arr)
    np.random.shuffle(arr_copy)
    return arr_copy


def detect_cycle_patterns(chromosome_1, chromosome_2):

    patterns = []
    visited_genes_chromosome_1 = set()

    for i in range(len(chromosome_1)):
        if chromosome_1[i] in visited_genes_chromosome_1:
            continue
        visited_genes = []
        visited_indices = []
        index = i
        while True:
            gene = chromosome_1[index]
            if visited_genes and gene == visited_genes[0]:
                patterns.append((visited_genes, visited_indices))
                visited_genes_chromosome_1.update(visited_genes)
                break
            visited_genes.append(gene)
            visited_indices.append(index)
            chromosome_2_gene = chromosome_2[index]
            next_index = np.where(chromosome_1 == chromosome_2_gene)[0][0]
            index = next_index

    return patterns


def elitism_replacement(ga_instance, first_two_child, population_fitness, n_elite=2):
    current_population = ga_instance.get_pop()
    pop_size = ga_instance.get_pop_size()

    population_fitness.sort(key=lambda x: x[1], reverse=True)

    elite_ids = []
    for i in range(n_elite):
        id = population_fitness[i][0]
        elite_ids.append(id)

    new_population = {}
    for elite_id in elite_ids:
        new_population[elite_id] = current_population[elite_id]

    all_child = list(first_two_child)

    num_child_needed = pop_size - n_elite

    pairs_to_create = (num_child_needed - 2) // 2

    for _ in range(pairs_to_create):
        parent1, parent2 = ga_instance.select_parents()

        child1, child2 = ga_instance.recombine(
            parent1, parent2, combinatorName=ga_instance._crossover_type
        )

        child1 = ga_instance.mutate(
            child1, mutatorName=ga_instance._mutation_type)
        child2 = ga_instance.mutate(
            child2, mutatorName=ga_instance._mutation_type)

        all_child.append(child1)
        all_child.append(child2)

    for i in range(num_child_needed):
        child = all_child[i]
        new_population[child.get_id()] = child

    return new_population


def run_experiments(processing_times, priorities, setup_times,
                    pop_size=50, max_generations=100, runs_per_config=5):
    crossover_ops = ['order', 'pmx', 'cycle']
    mutation_ops = ['swap', 'inversion', 'scramble']

    results = {}

    print("\n" + "="*70)
    print("  GENETIC ALGORITHM EXPERIMENT: TESTING 9 CONFIGURATIONS")
    print("="*70)
    print(f"Population Size: {pop_size}")
    print(f"Max Generations: {max_generations}")
    print(f"Runs per Configuration: {runs_per_config}")
    print(
        f"Total Runs: {len(crossover_ops) * len(mutation_ops) * runs_per_config}")
    print("="*70 + "\n")

    config_num = 0
    total_configs = len(crossover_ops) * len(mutation_ops)

    for crossover in crossover_ops:
        for mutation in mutation_ops:
            config_num += 1
            config_name = f"{crossover}_{mutation}"
            config_results = []

            print(
                f"[{config_num}/{total_configs}] Configuration: {config_name.upper()}")
            print(
                f"      Crossover: {crossover.capitalize()} | Mutation: {mutation.capitalize()}")

            for run in range(runs_per_config):
                print(f"      Run {run+1}/{runs_per_config}...", end="")
                print()
                ga = JobSchedulingGA(
                    processing_times, priorities, setup_times,
                    pop_size=pop_size, max_generations=max_generations,
                    crossover_type=crossover, mutation_type=mutation
                )
                result = ga.run_ga()
                config_results.append(result)
                print(
                    f" Best Fitness: {result['best_fitness']:.6f} (TWCT: {1/result['best_fitness']:.2f})")

            avg_fitness = np.mean([r['best_fitness'] for r in config_results])
            print(
                f"      -> Average Best Fitness: {avg_fitness:.6f} (Avg TWCT: {1/avg_fitness:.2f})\n")

            results[config_name] = config_results

    best_combo = max(results.items(),
                     key=lambda x: np.mean([r['best_fitness'] for r in x[1]]))

    return results, best_combo


def plot_results(results, best_combo):
    config_names = list(results.keys())
    best_config_name = best_combo[0]
    best_run = max(results[best_config_name], key=lambda x: x['best_fitness'])

    # ============================================================
    # Plot 1: Box Plot - Best Fitness Distribution by Configuration
    # ============================================================
    plt.figure(figsize=(14, 6))
    best_fitnesses = [[r['best_fitness']
                       for r in results[config]] for config in config_names]
    bp = plt.boxplot(best_fitnesses, labels=config_names, patch_artist=True)

    colors = ['#FF6B6B', '#FF6B6B', '#FF6B6B',
              '#4ECDC4', '#4ECDC4', '#4ECDC4',
              '#45B7D1', '#45B7D1', '#45B7D1']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Best Fitness', fontsize=12)
    plt.xlabel('Configuration (Crossover_Mutation)', fontsize=12)
    plt.title(
        'Best Fitness Distribution Across All 9 Configurations\n(Red=Order, Teal=PMX, Blue=Cycle)',
        fontsize=14, fontweight='bold'
    )
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('01_fitness_distribution_boxplot.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 01_fitness_distribution_boxplot.png")

    # ============================================================
    # Plot 2: Maximum Fitness - All 9 Configurations
    # ============================================================
    plt.figure(figsize=(14, 8))
    colors_map = {'order': '#FF6B6B', 'pmx': '#4ECDC4', 'cycle': '#45B7D1'}
    linestyles = {'swap': '-', 'inversion': '--', 'scramble': ':'}

    for config in config_names:
        co, mu = config.split('_')
        # Use best run from each config
        best_config_run = max(results[config], key=lambda x: x['best_fitness'])
        plt.plot(
            best_config_run['best_fitness_history'],
            color=colors_map[co],
            linestyle=linestyles[mu],
            linewidth=2,
            alpha=0.8,
            label=config
        )

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Maximum Fitness', fontsize=12)
    plt.title(
        'Maximum Fitness Per Generation: All 9 Configurations\n'
        '(Color=Crossover | LineStyle=Mutation: Solid=Swap, Dashed=Inversion, Dotted=Scramble)',
        fontsize=14,
        fontweight='bold'
    )
    plt.legend(loc='lower right', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('02_maximum_fitness_all_configs.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 02_maximum_fitness_all_configs.png")

    # ============================================================
    # Plot 3: Average Fitness - All 9 Configurations
    # ============================================================
    plt.figure(figsize=(14, 8))

    for config in config_names:
        co, mu = config.split('_')
        best_config_run = max(results[config], key=lambda x: x['best_fitness'])
        plt.plot(
            best_config_run['avg_fitness_history'],
            color=colors_map[co],
            linestyle=linestyles[mu],
            linewidth=2,
            alpha=0.8,
            label=config
        )

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Average Fitness', fontsize=12)
    plt.title(
        'Average Fitness Per Generation: All 9 Configurations\n'
        '(Color=Crossover | LineStyle=Mutation: Solid=Swap, Dashed=Inversion, Dotted=Scramble)',
        fontsize=14,
        fontweight='bold'
    )
    plt.legend(loc='lower right', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('03_average_fitness_all_configs.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 03_average_fitness_all_configs.png")

    # ============================================================
    # Plot 4: Max vs Avg Fitness - Best Configuration Only
    # ============================================================
    plt.figure(figsize=(10, 6))
    generations = range(1, len(best_run['best_fitness_history']) + 1)

    plt.plot(
        generations,
        best_run['best_fitness_history'],
        label='Maximum Fitness',
        linewidth=2.5,
        color='#2E86AB',
        marker='o',
        markersize=4,
        markevery=5
    )

    plt.plot(
        generations,
        best_run['avg_fitness_history'],
        label='Average Fitness',
        linewidth=2.5,
        color='#F77F00',
        marker='s',
        markersize=4,
        markevery=5
    )

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(
        f'Maximum and Average Fitness Per Generation\nBest Configuration: {best_config_name}',
        fontsize=14,
        fontweight='bold'
    )
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('04_max_avg_fitness_best_config.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 04_max_avg_fitness_best_config.png")

    # ============================================================
    # Plot 5: Bar Chart - Average Performance Comparison
    # ============================================================
    plt.figure(figsize=(12, 6))
    avg_best = [np.mean([r['best_fitness'] for r in results[config]])
                for config in config_names]

    bar_colors = [colors_map[config.split('_')[0]] for config in config_names]

    bars = plt.bar(range(len(config_names)), avg_best,
                   color=bar_colors, alpha=0.7, edgecolor='black')

    # Highlight the best configuration
    best_idx = config_names.index(best_config_name)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    plt.xticks(range(len(config_names)), config_names, rotation=45, ha='right')
    plt.ylabel('Average Best Fitness', fontsize=12)
    plt.xlabel('Configuration (Crossover_Mutation)', fontsize=12)
    plt.title(
        'Average Best Fitness Across 5 Runs per Configuration\n(Gold border = Best overall)',
        fontsize=14,
        fontweight='bold'
    )
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('05_average_performance_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 05_average_performance_comparison.png")

    # ============================================================
    # Plot 6: Combined Max & Avg for All 9 Configs (Subplots)
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top: Maximum Fitness
    for config in config_names:
        co, mu = config.split('_')
        best_config_run = max(results[config], key=lambda x: x['best_fitness'])
        ax1.plot(
            best_config_run['best_fitness_history'],
            color=colors_map[co],
            linestyle=linestyles[mu],
            linewidth=2,
            alpha=0.8,
            label=config
        )

    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Maximum Fitness', fontsize=11)
    ax1.set_title('Maximum Fitness Per Generation: All 9 Configurations',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Bottom: Average Fitness
    for config in config_names:
        co, mu = config.split('_')
        best_config_run = max(results[config], key=lambda x: x['best_fitness'])
        ax2.plot(
            best_config_run['avg_fitness_history'],
            color=colors_map[co],
            linestyle=linestyles[mu],
            linewidth=2,
            alpha=0.8,
            label=config
        )

    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Average Fitness', fontsize=11)
    ax2.set_title('Average Fitness Per Generation: All 9 Configurations',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('06_combined_max_avg_all_configs.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: 06_combined_max_avg_all_configs.png")

    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":

    processing_times = np.array([12, 7, 15, 5, 9, 11, 8])
    priorities = np.array([4, 9, 3, 7, 5, 6, 8])
    setup_times = np.array([
        [0, 4, 6, 5, 7, 3, 4],
        [4, 0, 5, 6, 4, 5, 6],
        [6, 5, 0, 4, 5, 7, 6],
        [5, 6, 4, 0, 3, 4, 5],
        [7, 4, 5, 3, 0, 6, 4],
        [3, 5, 7, 4, 6, 0, 5],
        [4, 6, 6, 5, 4, 5, 0]
    ])

    # Run experiments
    results, best_combo = run_experiments(
        processing_times, priorities, setup_times,
        pop_size=50, max_generations=100, runs_per_config=5
    )

    best_config_name = best_combo[0]
    best_results = best_combo[1]

    best_run = max(best_results, key=lambda x: x['best_fitness'])

    # Print detailed results
    print("\n" + "="*70)
    print("  FINAL RESULTS - ALL CONFIGURATIONS")
    print("="*70)

    # Sort configurations by average fitness
    sorted_configs = sorted(results.items(),
                            key=lambda x: np.mean(
                                [r['best_fitness'] for r in x[1]]),
                            reverse=True)

    print(f"\n{'Rank':<6} {'Configuration':<20} {'Avg Fitness':<15} {'Avg TWCT':<12} {'Best TWCT':<12}")
    print("-"*70)

    for rank, (config_name, config_results) in enumerate(sorted_configs, 1):
        avg_fit = np.mean([r['best_fitness'] for r in config_results])
        avg_twct = 1/avg_fit
        best_twct = min([1/r['best_fitness'] for r in config_results])

        marker = " *" if rank == 1 else ""
        print(
            f"{rank:<6} {config_name:<20} {avg_fit:<15.6f} {avg_twct:<12.2f} {best_twct:<12.2f}{marker}")

    print("\n" + "="*70)
    print("  BEST CONFIGURATION DETAILS")
    print("="*70)
    print(f"Configuration: {best_config_name.upper()}")
    print(f"Best Job Sequence: {best_run['best_chromosome']}")
    print(f"Best Fitness: {best_run['best_fitness']:.6f}")
    print(
        f"Minimum Total Weighted Completion Time: {1/best_run['best_fitness']:.2f}")
    print(f"Generations Executed: {len(best_run['best_fitness_history'])}")

    # Show job sequence in human-readable format (1-indexed)
    print(
        f"\nJob Execution Order: {' -> '.join([str(j+1) for j in best_run['best_chromosome']])}")
    print("="*70 + "\n")

    # Generate plots
    print("Generating performance plots...")
    plot_results(results, best_combo)
