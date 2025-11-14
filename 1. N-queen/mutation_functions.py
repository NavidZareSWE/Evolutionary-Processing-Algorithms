import numpy as np

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


def mutate_bitwise(chromosome, mutation_probability):
    # print("Running mutate_bitwise")
    new_mutated_chromosome = chromosome.copy()
    chromosome_len = len(new_mutated_chromosome)
    for idx, gene in np.ndenumerate(new_mutated_chromosome):
        if np.random.random() > mutation_probability:
            continue
        else:
            i = np.random.randint(chromosome_len)
            new_mutated_chromosome[idx] = i
    return new_mutated_chromosome
