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
