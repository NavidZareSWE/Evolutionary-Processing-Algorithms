import numpy as np

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
    def _create_offspring_pmx(p1, p2, point1, point2):
        size = len(p1)
        # mark unfilled positions
        child = np.full(size, -1)

        child[point1:point2+1] = p1[point1:point2+1]

        for i in range(point1, point2+1):
            val = p2[i]

            if val not in child[point1:point2+1]:
                j = child[i]
                pos_j_in_p2 = np.where(p2 == j)

                while child[pos_j_in_p2] != -1:
                    k = child[pos_j_in_p2]
                    pos_j_in_p2 = np.where(p2 == k)

                child[pos_j_in_p2] = val

        # Fill the rest
        for i in range(size):
            if child[i] == -1:
                child[i] = p2[i]
        return child
    parent1_chromosome = np.asarray(parent1.get_chromosome())
    parent2_chromosome = np.asarray(parent2.get_chromosome())
    length = len(parent1_chromosome)

    point1, point2 = np.random.choice(
        range(length), size=2, replace=False)

    if point1 > point2:  # Swap if point1 is larger
        point1, point2 = point2, point1

    child1 = _create_offspring_pmx(
        parent1_chromosome, parent2_chromosome, point1, point2)

    child2 = _create_offspring_pmx(
        parent2_chromosome, parent1_chromosome, point1, point2)

    return child1, child2


def crossover_n_cut(parent1, parent2, n_cuts=2):
    parent1_chromosome = np.asarray(parent1.get_chromosome())
    parent2_chromosome = np.asarray(parent2.get_chromosome())
    length = len(parent1_chromosome)
    cut_points = sorted(np.random.choice(
        range(1, length), size=n_cuts, replace=False))
    boundaries = [0] + cut_points + [length]

    child1 = np.array([], dtype=parent1_chromosome.dtype)
    child2 = np.array([], dtype=parent2_chromosome.dtype)

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        if i % 2 == 0:  # Even segments: already copied
            child1 = np.append(child1, parent1_chromosome[start:end])
            child2 = np.append(child2, parent2_chromosome[start:end])
        else:  # Odd segments: fill from other parent
            used1 = set(int(gene) for gene in child1)
            curr_idx = start
            # cut_point_len =  target_size = current_child_size + length_of_ith_cut_point
            # cut_point_len =  target_size = current_child_size + (end_of_ith_cut_point - start_of_ith_cut_point)
            target_size = len(child1) + (end - start)
            while len(child1) < target_size:
                p2_idx = curr_idx % length
                gene = parent2_chromosome[p2_idx]
                if gene not in used1:
                    child1 = np.append(child1, gene)
                    used1.add(int(gene))
                curr_idx += 1

            used2 = set(int(gene) for gene in child2)
            curr_idx = start
            target_size = len(child2) + (end - start)
            while len(child2) < target_size:
                p1_idx = curr_idx % length
                gene = parent1_chromosome[p1_idx]
                if gene not in used2:
                    child2 = np.append(child2, gene)
                    used2.add(int(gene))
                curr_idx += 1

    return child1, child2
