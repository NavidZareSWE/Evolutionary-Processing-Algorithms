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


def generational_replacement(ga_instance, first_two_child):
    all_child = list(
        first_two_child)

    for _ in range(49):
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

    new_population = {}
    for child in all_child:
        new_population[child.get_id()] = child

    return new_population


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

    # Calculate how many more we need
    # Total needed: pop_size - n_elite
    # Need to create: (pop_size - n_elite - 2(already created children)) / 2 pairs
    num_child_needed = pop_size - n_elite
    # // = to get integet instead of accidental float
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
