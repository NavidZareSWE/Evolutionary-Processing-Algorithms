# ============================================================================
# ============================================================================


def dominates(obj1, obj2):
    """Check if obj1 dominates obj2 (minimization).

    Checks if the first objective vector is Pareto dominant over the second objective vector.
    containing two elements: [error, n_features].

    Args:
        obj1,obj2 (tuple/list): The first objective vector, e.g., [15.5, 42] where 15.5%
                           error is achieved using 42 features.
    Returns:
        bool: True if obj1 is Pareto dominant over obj2, False otherwise.

    Example:
        dominance_check([15.5, 42], [20.0, 50])  # Returns True
        dominance_check([20.0, 50], [20.0, 40])  # Returns False
    """
    at_least_as_good = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
    strictly_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    return at_least_as_good and strictly_better


def fast_non_dominated_sort(population, objectives):
    n = len(population)
    # Initialize a list to count dominations for n [0,0,0,0,...]
    domination_count = [0] * n
    # Create a list of empty lists to store dominated solutions for n
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_solutions[j].append(i)
                domination_count[i] += 1

    # First front
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    # Other fronts
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for idx in fronts[i]:
            for dominated_idx in dominated_solutions[idx]:
                domination_count[dominated_idx] -= 1
                if domination_count[dominated_idx] == 0:
                    next_front.append(dominated_idx)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts


def crowding_distance(objectives, front_indices):
    # Assigns infinite crowding distance to solutions if there are 2 or fewer in the front.
    # Reason: This preserves boundary points, ensuring they remain in the selection process.
    if len(front_indices) <= 2:
        return {idx: float('inf') for idx in front_indices}

    n_objectives = len(objectives[0])
    distances = {idx: 0.0 for idx in front_indices}

    for m in range(n_objectives):
        sorted_indices = sorted(front_indices, key=lambda x: objectives[x][m])

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # Calculate range
        obj_min = objectives[sorted_indices[0]][m]
        obj_max = objectives[sorted_indices[-1]][m]
        obj_range = obj_max - obj_min

        if obj_range == 0:
            continue

        # Calculate distances for intermediate points (non-boundary solutions)
        # Larger gap -> larger crowding distance
        # Smaller gap -> more crowded -> smaller distance
        for i in range(1, len(sorted_indices) - 1):
            distances[sorted_indices[i]] += (
                (objectives[sorted_indices[i + 1]][m] -
                 objectives[sorted_indices[i - 1]][m])
                / obj_range
            )

    return distances


# ============================================================================
# ============================================================================


def merge_pareto_fronts(fronts_list):
    """Merge multiple Pareto fronts and return the combined non-dominated front."""
    all_points = []
    for front in fronts_list:
        all_points.extend(front)

    if not all_points:
        return []

    # Find non-dominated points
    non_dominated = []
    for point in all_points:
        is_dominated = False
        for other in all_points:
            if point != other and dominates(other, point):
                is_dominated = True
                break
        if not is_dominated:
            # Avoid duplicates - convert to native types for comparison
            point_native = [float(point[0]), int(point[1])]
            if point_native not in non_dominated:
                non_dominated.append(point_native)

    return sorted(non_dominated, key=lambda x: x[0])
