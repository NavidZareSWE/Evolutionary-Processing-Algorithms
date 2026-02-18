"""
MOBGA-AOS: Multi-Objective Binary Genetic Algorithm with Adaptive Operator Selection
Implementation based on: Yu Xue et al. (2021) - Knowledge-Based Systems 227 (2021) 107218
"Adaptive crossover operator based multi-objective binary genetic algorithm for feature selection"

Complete implementation for Evolutionary Computing Assignment 5
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def load_dataset(filepath):
    """Load dataset from CSV file. Last column is the target variable."""
    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=0)
        if np.isnan(data).any():
            data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        return X, y
    except Exception as e:
        raise ValueError("Error loading dataset: {}".format(e))


def normalize_data(X):
    """Normalize features to [0, 1] range."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1  # Avoid division by zero
    return (X - X_min) / denom


def train_test_split(X, y, train_ratio=0.7, seed=42):
    """Split data into training and test sets."""
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def format_pareto_front(pf):
    """
    Format Pareto front for clean console output without numpy type wrappers.
    
    Parameters:
    -----------
    pf : list - List of [error, n_features] pairs
    
    Returns:
    --------
    str - Formatted string representation
    """
    if not pf:
        return "[]"
    formatted = []
    for point in pf:
        error = float(point[0])
        n_feat = int(point[1])
        formatted.append("[{:.2f}, {}]".format(error, n_feat))
    return "[" + ", ".join(formatted) + "]"


def convert_pareto_front(pf):
    """
    Convert Pareto front to native Python types.
    
    Parameters:
    -----------
    pf : list - List of [error, n_features] pairs
    
    Returns:
    --------
    list - List with native Python types
    """
    if not pf:
        return []
    return [[float(point[0]), int(point[1])] for point in pf]


# ============================================================================
# k-NN CLASSIFIER (k=3) - VECTORIZED FOR PERFORMANCE
# ============================================================================

def knn_predict(X_train, y_train, X_test, k=3):
    """
    Predict labels for test samples using k-NN.
    OPTIMIZED: Uses vectorized numpy operations and squared distances.
    """
    # Compute squared distances (skip sqrt since we only need relative ordering)
    # Using the formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    train_sq = np.sum(X_train ** 2, axis=1)
    test_sq = np.sum(X_test ** 2, axis=1)
    cross_term = np.dot(X_test, X_train.T)
    distances_sq = test_sq[:, np.newaxis] + \
        train_sq[np.newaxis, :] - 2 * cross_term

    # Get indices of k nearest neighbors for all test samples at once
    k_nearest_indices = np.argpartition(distances_sq, k, axis=1)[:, :k]

    # Get the labels of k nearest neighbors
    k_nearest_labels = y_train[k_nearest_indices]

    # Majority voting (vectorized where possible)
    predictions = np.zeros(X_test.shape[0], dtype=y_train.dtype)
    for i in range(X_test.shape[0]):
        unique, counts = np.unique(k_nearest_labels[i], return_counts=True)
        predictions[i] = unique[np.argmax(counts)]

    return predictions


def cross_validation_error(X, y, selected_features, n_folds=3, k=3):
    """
    Compute classification error using n-fold cross-validation with k-NN.

    Parameters:
    -----------
    X : np.ndarray - Feature matrix
    y : np.ndarray - Labels
    selected_features : np.ndarray - Binary array indicating selected features
    n_folds : int - Number of folds for cross-validation
    k : int - Number of neighbors for k-NN

    Returns:
    --------
    float - Classification error rate (0-100%)
    """
    if np.sum(selected_features) == 0:
        return 100.0  # No features selected

    feature_indices = np.where(selected_features == 1)[0]
    X_selected = X[:, feature_indices]

    n_samples = X.shape[0]
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)

    total_errors = 0
    total_samples = 0

    for fold in range(n_folds):
        # Define test indices for this fold
        test_start = fold * fold_size
        if fold == n_folds - 1:
            test_end = n_samples
        else:
            test_end = (fold + 1) * fold_size

        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate(
            [indices[:test_start], indices[test_end:]])

        X_train, y_train = X_selected[train_indices], y[train_indices]
        X_test, y_test = X_selected[test_indices], y[test_indices]

        # Predict and count errors
        predictions = knn_predict(X_train, y_train, X_test, k=k)
        errors = np.sum(predictions != y_test)

        total_errors += errors
        total_samples += len(y_test)

    error_rate = (total_errors / total_samples) * 100.0
    return error_rate


def compute_all_features_error(X, y, n_folds=3, k=3):
    """Compute classification error using all features (baseline)."""
    all_features = np.ones(X.shape[1], dtype=int)
    return cross_validation_error(X, y, all_features, n_folds=n_folds, k=k)


# ============================================================================
# CROSSOVER OPERATORS (5 operators as per the paper)
# ============================================================================

def single_point_crossover(parent1, parent2):
    """Single-point crossover operator."""
    length = len(parent1)
    point = np.random.randint(1, length)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def two_point_crossover(parent1, parent2):
    """Two-point crossover operator."""
    length = len(parent1)
    points = sorted(np.random.choice(range(1, length), 2, replace=False))
    p1, p2 = points
    child1 = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]])
    child2 = np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]])
    return child1, child2


def uniform_crossover(parent1, parent2):
    """Uniform crossover operator."""
    mask = np.random.randint(0, 2, len(parent1))
    child1 = np.where(mask == 1, parent1, parent2)
    child2 = np.where(mask == 1, parent2, parent1)
    return child1, child2


def shuffle_crossover(parent1, parent2):
    """Shuffle crossover operator."""
    length = len(parent1)
    shuffle_idx = np.random.permutation(length)

    # Shuffle both parents
    shuffled_p1 = parent1[shuffle_idx]
    shuffled_p2 = parent2[shuffle_idx]

    # Apply single-point crossover
    point = np.random.randint(1, length)
    shuffled_c1 = np.concatenate([shuffled_p1[:point], shuffled_p2[point:]])
    shuffled_c2 = np.concatenate([shuffled_p2[:point], shuffled_p1[point:]])

    # Unshuffle
    inverse_idx = np.argsort(shuffle_idx)
    child1 = shuffled_c1[inverse_idx]
    child2 = shuffled_c2[inverse_idx]

    return child1, child2


def reduced_surrogate_crossover(parent1, parent2):
    """Reduced surrogate crossover operator - only crosses at differing positions."""
    diff_positions = np.where(parent1 != parent2)[0]

    if len(diff_positions) == 0:
        # Parents are identical, return copies
        return parent1.copy(), parent2.copy()

    # Select a random crossover point from differing positions
    point_idx = np.random.randint(0, len(diff_positions))
    point = diff_positions[point_idx]

    # Apply single-point crossover at this position
    child1 = np.concatenate([parent1[:point+1], parent2[point+1:]])
    child2 = np.concatenate([parent2[:point+1], parent1[point+1:]])

    return child1, child2


# List of crossover operators
CROSSOVER_OPERATORS = [
    single_point_crossover,
    two_point_crossover,
    uniform_crossover,
    shuffle_crossover,
    reduced_surrogate_crossover
]

OPERATOR_NAMES = [
    "Single-Point",
    "Two-Point",
    "Uniform",
    "Shuffle",
    "Reduced Surrogate"
]


# ============================================================================
# MUTATION OPERATOR
# ============================================================================

def uniform_mutation(individual, mutation_rate):
    """Uniform mutation operator - flip each bit with probability mutation_rate.
    OPTIMIZED: Vectorized using numpy.
    """
    mutant = individual.copy()
    # Generate all random numbers at once
    flip_mask = np.random.random(len(mutant)) < mutation_rate
    mutant[flip_mask] = 1 - mutant[flip_mask]
    return mutant


# ============================================================================
# PARETO DOMINANCE AND NON-DOMINATED SORTING
# ============================================================================

def dominates(obj1, obj2):
    """
    Check if obj1 dominates obj2 (minimization).
    obj1 dominates obj2 if obj1 is at least as good in all objectives
    and strictly better in at least one objective.
    """
    at_least_as_good = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
    strictly_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    return at_least_as_good and strictly_better


def fast_non_dominated_sort(population, objectives):
    """
    NSGA-II fast non-dominated sorting.

    Parameters:
    -----------
    population : list - List of individuals
    objectives : list - List of objective values [error, n_features] for each individual

    Returns:
    --------
    list of lists - Fronts, where fronts[i] contains indices of individuals in front i
    """
    n = len(population)
    domination_count = [0] * n  # Number of solutions dominating this solution
    # Solutions dominated by this solution
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

    # Subsequent fronts
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
    """
    Calculate crowding distance for individuals in a front.

    Parameters:
    -----------
    objectives : list - List of objective values for all individuals
    front_indices : list - Indices of individuals in the front

    Returns:
    --------
    dict - Mapping from individual index to crowding distance
    """
    if len(front_indices) <= 2:
        return {idx: float('inf') for idx in front_indices}

    n_objectives = len(objectives[0])
    distances = {idx: 0.0 for idx in front_indices}

    for m in range(n_objectives):
        # Sort by objective m
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

        # Calculate distances for intermediate points
        for i in range(1, len(sorted_indices) - 1):
            distances[sorted_indices[i]] += (
                (objectives[sorted_indices[i + 1]][m] -
                 objectives[sorted_indices[i - 1]][m])
                / obj_range
            )

    return distances


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def compute_hypervolume_2d(pareto_front, reference_point):
    """
    Compute hypervolume for 2D bi-objective minimization problem.

    Parameters:
    -----------
    pareto_front : list - List of [error, n_features] objective pairs
    reference_point : list - Reference point [max_error, max_features]

    Returns:
    --------
    float - Hypervolume value
    """
    if not pareto_front:
        return 0.0

    # Filter points that are dominated by reference point
    valid_points = [p for p in pareto_front if p[0] <
                    reference_point[0] and p[1] < reference_point[1]]

    if not valid_points:
        return 0.0

    # Sort by first objective (error)
    sorted_points = sorted(valid_points, key=lambda x: x[0])

    # Calculate hypervolume
    hv = 0.0
    prev_obj2 = reference_point[1]

    for point in sorted_points:
        if point[1] < prev_obj2:
            width = reference_point[0] - point[0]
            height = prev_obj2 - point[1]
            hv += width * height
            prev_obj2 = point[1]

    return hv


def compute_igd(obtained_front, true_front):
    """
    Compute Inverted Generational Distance (IGD).

    Parameters:
    -----------
    obtained_front : list - Obtained Pareto front
    true_front : list - True/reference Pareto front

    Returns:
    --------
    float - IGD value (lower is better)
    """
    if not obtained_front or not true_front:
        return float('inf')

    obtained_arr = np.array(obtained_front)
    true_arr = np.array(true_front)

    total_dist = 0.0
    for true_point in true_arr:
        min_dist = min(np.sqrt(np.sum((true_point - obt_point) ** 2))
                       for obt_point in obtained_arr)
        total_dist += min_dist

    return total_dist / len(true_arr)


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


# ============================================================================
# MOBGA-AOS ALGORITHM
# ============================================================================

class MOBGA_AOS:
    """
    Multi-Objective Binary Genetic Algorithm with Adaptive Operator Selection.

    Based on: Yu Xue et al. (2021) - Knowledge-Based Systems 227 (2021) 107218
    """

    def __init__(self, n_features, max_fes=30000, pop_size=100,
                 crossover_rate=0.9, lp=5, verbose=True):
        """
        Initialize MOBGA-AOS.

        Parameters:
        -----------
        n_features : int - Number of features in the dataset
        max_fes : int - Maximum number of fitness evaluations
        pop_size : int - Population size
        crossover_rate : float - Crossover probability
        lp : int - Learning period for OSP update
        verbose : bool - Print progress information
        """
        self.n_features = n_features
        self.max_fes = max_fes
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = 1.0 / n_features
        self.lp = lp
        self.verbose = verbose

        # Number of crossover operators
        self.n_operators = len(CROSSOVER_OPERATORS)

        # Operator Selection Probabilities (OSP)
        self.osp = np.ones(self.n_operators) / self.n_operators

        # Reward/Penalty matrices for LP generations
        self.RD = np.zeros((lp, self.n_operators))  # Rewards
        self.PN = np.zeros((lp, self.n_operators))  # Penalties

        # Data
        self.X = None
        self.y = None

        # Tracking
        self.n_fes = 0
        self.generation = 0
        self.hv_history = []
        self.osp_history = []

        # Fitness cache to avoid re-evaluating identical individuals
        self.fitness_cache = {}

    def load_data(self, X, y):
        """Load and normalize training data."""
        self.X = normalize_data(X)
        self.y = y

    def initialize_population(self):
        """Initialize random binary population."""
        population = []
        for _ in range(self.pop_size):
            # Random binary individual
            individual = np.random.randint(0, 2, self.n_features)
            population.append(individual)
        return population

    def evaluate(self, individual):
        """
        Evaluate an individual.
        Uses caching to avoid re-evaluating identical individuals.

        Returns:
        --------
        tuple - (classification_error, n_selected_features)
        """
        # Convert to hashable tuple for cache lookup
        cache_key = tuple(individual)

        if cache_key in self.fitness_cache:
            # Return cached result (doesn't count as new FE)
            return self.fitness_cache[cache_key]

        error = cross_validation_error(
            self.X, self.y, individual, n_folds=3, k=3)
        n_features = int(np.sum(individual))
        self.n_fes += 1

        result = (error, n_features)
        self.fitness_cache[cache_key] = result
        return result

    def roulette_wheel_selection(self, probabilities):
        """Select an operator index using roulette wheel selection."""
        cumsum = np.cumsum(probabilities)
        r = np.random.random()
        for i, cs in enumerate(cumsum):
            if r <= cs:
                return i
        return len(probabilities) - 1

    def binary_tournament_selection(self, population, objectives):
        """Select a parent using binary tournament selection."""
        i, j = np.random.choice(len(population), 2, replace=False)

        # Compare by dominance
        if dominates(objectives[i], objectives[j]):
            return i
        elif dominates(objectives[j], objectives[i]):
            return j
        else:
            # Non-dominated: random choice
            return np.random.choice([i, j])

    def credit_assignment(self, parents, children, parent_objs, child_objs, operator_idx):
        """
        Assign credit (reward/penalty) to the operator based on offspring quality.

        Based on Algorithm 2 from the paper.
        """
        reward = 0
        penalty = 0

        # Check dominance between parents
        if dominates(parent_objs[0], parent_objs[1]):
            # Parent 0 dominates parent 1
            dominating_parent = parent_objs[0]
            for c_obj in child_objs:
                if dominates(dominating_parent, c_obj):
                    penalty += 1
                else:
                    reward += 1
        elif dominates(parent_objs[1], parent_objs[0]):
            # Parent 1 dominates parent 0
            dominating_parent = parent_objs[1]
            for c_obj in child_objs:
                if dominates(dominating_parent, c_obj):
                    penalty += 1
                else:
                    reward += 1
        else:
            # Parents are non-dominated to each other
            for c_obj in child_objs:
                # Child is rewarded if not dominated by both parents
                if dominates(parent_objs[0], c_obj) or dominates(parent_objs[1], c_obj):
                    penalty += 1
                else:
                    reward += 1

        return reward, penalty

    def update_osp(self, gen_in_lp):
        """
        Update Operator Selection Probabilities based on accumulated rewards/penalties.

        Based on Equations (6)-(10) in the paper.
        """
        delta = 0.0001  # Small value to prevent division by zero

        # Sum rewards and penalties for each operator over LP generations
        S1 = np.sum(self.RD[:gen_in_lp], axis=0)  # Total rewards
        S2 = np.sum(self.PN[:gen_in_lp], axis=0)  # Total penalties

        # Calculate S3 (avoid division by zero)
        S3 = np.where(S1 == 0, delta, S1)

        # Calculate S4 (probability for each operator)
        S4 = S1 / (S3 + S2)

        # Normalize to get final OSP
        total = np.sum(S4)
        if total > 0:
            self.osp = S4 / total
        else:
            # Reset to uniform if all zero
            self.osp = np.ones(self.n_operators) / self.n_operators

        # Reset RD and PN matrices
        self.RD = np.zeros((self.lp, self.n_operators))
        self.PN = np.zeros((self.lp, self.n_operators))

    def environmental_selection(self, combined_pop, combined_objs):
        """
        Select next generation using NSGA-II environmental selection.
        """
        fronts = fast_non_dominated_sort(combined_pop, combined_objs)

        new_population = []
        new_objectives = []

        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                for idx in front:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_objs[idx])
            else:
                # Need to select some from this front based on crowding distance
                remaining = self.pop_size - len(new_population)
                distances = crowding_distance(combined_objs, front)

                # Sort by crowding distance (descending)
                sorted_front = sorted(
                    front, key=lambda x: distances[x], reverse=True)

                for idx in sorted_front[:remaining]:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_objs[idx])
                break

        return new_population, new_objectives

    def get_pareto_front(self, population, objectives):
        """Extract the first Pareto front with native Python types."""
        fronts = fast_non_dominated_sort(population, objectives)
        pareto_front = [[float(objectives[idx][0]), int(objectives[idx][1])] 
                        for idx in fronts[0]]
        return pareto_front

    def run(self, seed=42):
        """
        Run the MOBGA-AOS algorithm.

        Returns:
        --------
        tuple - (pareto_front, final_population)
        """
        np.random.seed(seed)

        # Initialize population
        population = self.initialize_population()
        objectives = [self.evaluate(ind) for ind in population]

        generation_in_lp = 0
        reference_point = [100.0, float(self.n_features + 1)]

        if self.verbose:
            print("Initial FEs: {}, Target: {}".format(self.n_fes, self.max_fes))

        while self.n_fes < self.max_fes:
            offspring_population = []
            offspring_objectives = []

            # Rewards/Penalties for this generation
            n_reward = np.zeros(self.n_operators)
            n_penalty = np.zeros(self.n_operators)

            # Generate N/2 pairs of offspring
            for _ in range(self.pop_size // 2):
                if self.n_fes >= self.max_fes:
                    break

                # Select crossover operator using roulette wheel
                operator_idx = self.roulette_wheel_selection(self.osp)
                crossover_op = CROSSOVER_OPERATORS[operator_idx]

                # Select two parents using binary tournament
                p1_idx = self.binary_tournament_selection(
                    population, objectives)
                p2_idx = self.binary_tournament_selection(
                    population, objectives)

                parent1 = population[p1_idx].copy()
                parent2 = population[p2_idx].copy()
                parent_objs = [objectives[p1_idx], objectives[p2_idx]]

                # Apply crossover with probability
                if np.random.random() < self.crossover_rate:
                    child1, child2 = crossover_op(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Apply mutation
                child1 = uniform_mutation(child1, self.mutation_rate)
                child2 = uniform_mutation(child2, self.mutation_rate)

                # Evaluate children
                child1_obj = self.evaluate(child1)
                child2_obj = self.evaluate(child2)
                child_objs = [child1_obj, child2_obj]

                # Credit assignment
                reward, penalty = self.credit_assignment(
                    [parent1, parent2], [child1, child2],
                    parent_objs, child_objs, operator_idx
                )
                n_reward[operator_idx] += reward
                n_penalty[operator_idx] += penalty

                # Add children to offspring
                offspring_population.extend([child1, child2])
                offspring_objectives.extend(child_objs)

            # Store rewards/penalties for this generation
            self.RD[generation_in_lp] = n_reward
            self.PN[generation_in_lp] = n_penalty
            generation_in_lp += 1

            # Update OSP every LP generations
            if generation_in_lp >= self.lp:
                self.update_osp(generation_in_lp)
                generation_in_lp = 0

            # Environmental selection (NSGA-II)
            combined_pop = population + offspring_population
            combined_objs = objectives + offspring_objectives
            population, objectives = self.environmental_selection(
                combined_pop, combined_objs)

            # Record history
            pareto_front = self.get_pareto_front(population, objectives)
            hv = compute_hypervolume_2d(pareto_front, reference_point)
            self.hv_history.append(hv)
            self.osp_history.append(self.osp.copy())

            self.generation += 1

            if self.verbose and self.generation % 10 == 0:
                print("Gen {}, FEs: {}, PF size: {}, HV: {:.4f}".format(
                    self.generation, self.n_fes, len(pareto_front), hv))

        # Extract final Pareto front
        final_pareto_front = self.get_pareto_front(population, objectives)

        return final_pareto_front, population


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_pareto_front(all_fronts, true_pf, dataset_name, n_features, baseline_error, save_path=None):
    """Plot Pareto fronts from multiple runs."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Plot individual run fronts
        for i, front in enumerate(all_fronts):
            if front:
                errors = [float(p[0]) for p in front]
                features = [int(p[1]) for p in front]
                ax.scatter(features, errors, c=colors[i % len(colors)], alpha=0.6,
                           s=50, label='Run {}'.format(i+1))

        # Plot true Pareto front
        if true_pf:
            true_errors = [float(p[0]) for p in true_pf]
            true_features = [int(p[1]) for p in true_pf]
            ax.plot(true_features, true_errors, 'k-', linewidth=2, marker='s',
                    markersize=8, label='Combined PF')

        # Plot baseline
        ax.axhline(y=baseline_error, color='r', linestyle='--',
                   label='All Features ({}): {:.2f}%'.format(n_features, baseline_error))

        ax.set_xlabel('Number of Selected Features', fontsize=12)
        ax.set_ylabel('Classification Error (%)', fontsize=12)
        ax.set_title('MOBGA-AOS Pareto Front: {}'.format(dataset_name), fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("Matplotlib not available for plotting")


def plot_hv_convergence(hv_histories, dataset_name, save_path=None):
    """Plot hypervolume convergence over generations."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, hv_history in enumerate(hv_histories):
            generations = range(1, len(hv_history) + 1)
            ax.plot(generations, hv_history, c=colors[i % len(colors)],
                    alpha=0.7, label='Run {}'.format(i+1))

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Hypervolume', fontsize=12)
        ax.set_title('Hypervolume Convergence: {}'.format(dataset_name), fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("Matplotlib not available for plotting")


def plot_osp_evolution(osp_histories, dataset_name, save_path=None):
    """Plot evolution of Operator Selection Probabilities."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Use first run's OSP history
        if osp_histories and osp_histories[0]:
            osp_array = np.array(osp_histories[0])
            generations = range(1, len(osp_array) + 1)

            for i, name in enumerate(OPERATOR_NAMES):
                ax.plot(generations, osp_array[:, i], label=name, linewidth=2)

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Selection Probability', fontsize=12)
        ax.set_title('Operator Selection Probability Evolution: {}'.format(dataset_name), fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("Matplotlib not available for plotting")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import os
    import json
    import time

    # Dataset configuration (as per assignment)
    DATASETS = {
        'DS02': ('DS02.csv', 'LungCancer', 10000),
        'DS04': ('DS04.csv', 'OpticalRecognition', 15000),
        'DS05': ('DS05.csv', 'MadelonValid', 30000),
        'DS07': ('DS07.csv', 'Har', 30000),
        'DS08': ('DS08.csv', 'HAPT', 30000),
        'DS10': ('DS10.csv', 'MultipleFeaturesDigit', 30000),
    }

    # Create results directory
    os.makedirs('results', exist_ok=True)

    all_results = {}

    for ds_id, (ds_file, ds_name, max_fes) in DATASETS.items():
        full_name = "{}_{}".format(ds_id, ds_name)
        print("\n" + "=" * 60)
        print("Processing: {}".format(full_name))
        print("=" * 60)

        try:
            # Load dataset
            X, y = load_dataset(ds_file)
            n_features = X.shape[1]
            n_samples = X.shape[0]
            n_classes = len(np.unique(y))

            print("Samples: {}, Features: {}, Classes: {}".format(
                n_samples, n_features, n_classes))

            # Run experiments (3 independent runs as per paper)
            all_fronts = []
            all_hv_histories = []
            all_osp_histories = []
            baselines = []
            run_times = []

            for run in range(3):
                seed = 42 + run * 100

                # Train-test split (70-30)
                X_train, y_train, X_test, y_test = train_test_split(
                    X, y, seed=seed)

                # Baseline error with all features
                baseline_error = compute_all_features_error(X_train, y_train)
                baselines.append(baseline_error)

                print("\nRun {}/3 (seed={})".format(run + 1, seed))
                print("  Baseline error (all features): {:.2f}%".format(baseline_error))

                # Initialize and run MOBGA-AOS
                mobga = MOBGA_AOS(
                    n_features=n_features,
                    max_fes=max_fes,
                    pop_size=100,
                    crossover_rate=0.9,
                    lp=5,
                    verbose=True
                )
                mobga.load_data(X_train, y_train)

                start_time = time.time()
                pareto_front, final_pop = mobga.run(seed=seed)
                run_time = time.time() - start_time

                all_fronts.append(pareto_front)
                all_hv_histories.append(mobga.hv_history)
                all_osp_histories.append(mobga.osp_history)
                run_times.append(run_time)

                print("  Pareto front size: {}".format(len(pareto_front)))
                print("  Run time: {:.2f}s".format(run_time))

            # Compute combined true Pareto front
            true_pf = merge_pareto_fronts(all_fronts)

            # Compute metrics
            reference_point = [100.0, float(n_features + 1)]

            igd_values = [compute_igd(front, true_pf) for front in all_fronts]
            hv_values = [compute_hypervolume_2d(
                front, reference_point) for front in all_fronts]

            mean_baseline = float(np.mean(baselines))
            igd_mean = float(np.mean(igd_values))
            igd_std = float(np.std(igd_values))
            hv_mean = float(np.mean(hv_values))
            hv_std = float(np.std(hv_values))

            print("\nResults for {}:".format(full_name))
            print("  IGD: {:.6f} +/- {:.6f}".format(igd_mean, igd_std))
            print("  HV:  {:.4f} +/- {:.4f}".format(hv_mean, hv_std))
            print("  Mean baseline error: {:.2f}%".format(mean_baseline))
            print("  Combined PF: {}".format(format_pareto_front(true_pf)))

            # Generate plots
            plot_pareto_front(all_fronts, true_pf, full_name, n_features,
                              mean_baseline, 'results/pareto_{}.png'.format(ds_id))
            plot_hv_convergence(all_hv_histories, full_name,
                                'results/hv_{}.png'.format(ds_id))
            plot_osp_evolution(all_osp_histories, full_name,
                               'results/osp_{}.png'.format(ds_id))

            # Store results with native Python types
            all_results[ds_id] = {
                'dataset': full_name,
                'n_features': int(n_features),
                'n_samples': int(n_samples),
                'n_classes': int(n_classes),
                'max_fes': int(max_fes),
                'igd_mean': igd_mean,
                'igd_std': igd_std,
                'hv_mean': hv_mean,
                'hv_std': hv_std,
                'baseline_error': mean_baseline,
                'true_pareto_front': convert_pareto_front(true_pf),
                'run_times': [float(t) for t in run_times],
            }

        except Exception as e:
            print("Error processing {}: {}".format(full_name, e))
            continue

    # Save all results
    with open('results/all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("Results saved to 'results/' directory")
    print("=" * 60)
