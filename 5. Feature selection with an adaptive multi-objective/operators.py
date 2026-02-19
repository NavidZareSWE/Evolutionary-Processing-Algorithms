import numpy as np

# ============================================================================
# CROSSOVER OPERATORS
# ============================================================================

def one_point_crossover(parent1, parent2, point=None):
    if point is None:
        point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def shuffle_crossover(parent1, parent2):
    length = len(parent1)
    shuffle_idx = np.random.permutation(length)

    shuffled_p1 = parent1[shuffle_idx]
    shuffled_p2 = parent2[shuffle_idx]
    shuffled_c1, shuffled_c2 = one_point_crossover(shuffled_p1, shuffled_p2)

    # Unshuffle
    inverse_idx = np.argsort(shuffle_idx)
    child1 = shuffled_c1[inverse_idx]
    child2 = shuffled_c2[inverse_idx]

    return child1, child2


def reduced_surrogate_crossover(parent1, parent2):
    diff_positions = np.where(parent1 != parent2)[0]

    if len(diff_positions) == 0:
        return parent1.copy(), parent2.copy()

    point_idx = np.random.randint(0, len(diff_positions))
    point = diff_positions[point_idx]

    child1, child2 = one_point_crossover(parent1, parent2, point)

    return child1, child2


CROSSOVER_OPERATORS = [
    shuffle_crossover,
    reduced_surrogate_crossover
]

OPERATOR_NAMES = [
    "Shuffle",
    "Reduced Surrogate"
]


# ============================================================================
# MUTATION OPERATOR
# ============================================================================

def uniform_mutation(individual, mutation_rate):
    mutant = individual.copy()
    for i in range(len(mutant)):
        if np.random.random() < mutation_rate:
            mutant[i] = 1 - mutant[i]
    return mutant
