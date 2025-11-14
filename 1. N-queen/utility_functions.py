import numpy as np
from individual import Individual

# ############################### Utility Functions ############################


def repair_chromosome(broken_chromosome):
    n = len(broken_chromosome)

    all_valid_values = set(range(n))
    values_in_brokone_chromose = set(broken_chromosome)
    missing_values = np.array(
        sorted(all_valid_values - values_in_brokone_chromose))

    repaired_chromosome = broken_chromosome.copy()
    used = set()

    for idx, gene in enumerate(repaired_chromosome):
        if gene in used:
            repaired_chromosome[idx] = missing_values[0]
            used.add(missing_values[0])
            missing_values = np.delete(missing_values, [0])
        else:
            used.add(gene)
    return repaired_chromosome

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
        return False
    #     print("chromosome has duplicate values")

    # expected_set = set(range(n))
    # actual_set = set(positions)
    # if expected_set != actual_set:
    #     missing = expected_set - actual_set
    #     extra = actual_set - expected_set
    #     print(f"Missing values: {missing}, Extra values: {extra}")

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
