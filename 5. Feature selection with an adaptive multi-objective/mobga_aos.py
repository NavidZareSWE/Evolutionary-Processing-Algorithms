import os
import json
import time
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from plots import plot_opSelectProb_evolution, plot_hv_convergence, plot_pareto_front
import warnings
warnings.filterwarnings('ignore')  # stop showing warning messages

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def load_dataset(filepath):
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
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1  # Avoid division by zero
    return (X - X_min) / denom


def train_test_split(X, y, train_ratio=0.7, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def format_pareto_front(pf):
    if not pf:
        return "[]"
    formatted = []
    for point in pf:
        error = float(point[0])
        n_feat = int(point[1])
        formatted.append("[{:.2f}, {}]".format(error, n_feat))
    return "[" + ", ".join(formatted) + "]"


def convert_pareto_front(pf):
    if not pf:
        return []
    return [[float(point[0]), int(point[1])] for point in pf]


def knn_predict(X_train, y_train, X_test, k=3):
    # Compute squared distances (skip sqrt of Euclidean distance since we only need relative ordering)
    # Using the formula: ||a-b||² = ||a||² + ||b||² - 2·(a·b)
    train_sq = np.sum(X_train ** 2, axis=1)
    test_sq = np.sum(X_test ** 2, axis=1)
    cross_term = np.dot(X_test, X_train.T)
    # Simply put, numpy.newaxis is used to increase the dimension of the existing array by one more dimension, when used once.
    # Read More: https://stackoverflow.com/questions/29241056/how-do-i-use-np-newaxis#comment114074218_41267079
    distances_sq = test_sq[:, np.newaxis] + \
        train_sq[np.newaxis, :] - 2 * cross_term

    # argpartition is faster than full sorting when we only need the k smallest values
    # It rearranges indices so that the k smallest elements come first (but not necessarily in sorted order)
    # Read More: https://stackoverflow.com/a/52465229/27639316
    k_nearest_indices = np.argpartition(distances_sq, k, axis=1)[:, :k]
    k_nearest_labels = y_train[k_nearest_indices]

    # Majority voting
    predictions = np.zeros(X_test.shape[0], dtype=y_train.dtype)
    for i in range(X_test.shape[0]):
        unique, counts = np.unique(k_nearest_labels[i], return_counts=True)
        predictions[i] = unique[np.argmax(counts)]

    return predictions


def cross_validation_error(X, y, selected_features, n_folds=3, k=3):
    total_features = X.shape[1]
    num_feats_selected = np.sum(selected_features)

    if num_feats_selected == 0:
        return 100.0
    elif num_feats_selected == total_features:
        X_selected = X
    else:
        feature_indices = np.where(selected_features == 1)[0]
        X_selected = X[:, feature_indices]

    n_samples = X.shape[0]
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)

    total_errors = 0
    total_samples = 0

    for fold in range(n_folds):
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

        predictions = knn_predict(X_train, y_train, X_test, k)
        errors = np.sum(predictions != y_test)

        total_errors += errors
        total_samples += len(y_test)

    error_rate = (total_errors / total_samples) * 100.0
    return error_rate


def compute_all_features_error(X, y, n_folds=3, k=3):
    all_features = np.ones(X.shape[1], dtype=int)
    return cross_validation_error(X, y, all_features, n_folds, k)


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


class MOBGA_AOS:
    def __init__(self, n_features, max_fes=30000, pop_size=100,
                 crossover_rate=0.9, lp=5):
        self.n_features = n_features
        self.max_fes = max_fes
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = 1.0 / n_features
        self.lp = lp
        self.num_x_over_operators = len(CROSSOVER_OPERATORS)
        self.X = None
        self.y = None
        self.n_fes = 0
        self.generation = 0
        self.hv_history = []
        self.opSelectProb_history = []
        self.fitness_cache = {}

        # Operator Selection Probabilities (opSelectProb)
        self.opSelectProb = np.ones(
            self.num_x_over_operators) / self.num_x_over_operators

        # Reward/Penalty matrices for LP(Learning Period for AOS) generations
        self.RD = np.zeros((lp, self.num_x_over_operators))  # Rewards
        self.PN = np.zeros((lp, self.num_x_over_operators))  # Penalties

    def load_data(self, X, y):
        self.X = normalize_data(X)
        self.y = y

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = np.random.randint(0, 2, self.n_features)
            population.append(individual)
        return population

    def evaluate(self, individual):
        # Convert to hashable tuple for cache lookup
        cache_key = tuple(individual)

        if cache_key in self.fitness_cache:
            # Return cached result (doesn't count as new Fitness Evaluation)
            return self.fitness_cache[cache_key]

        error = cross_validation_error(
            self.X, self.y, individual, n_folds=3, k=3)
        n_features = int(np.sum(individual))
        self.n_fes += 1

        result = (error, n_features)
        self.fitness_cache[cache_key] = result
        return result

    def roulette_wheel_selection(self, probabilities):
        # Return the cumulative sum of the elements along a given axis.
        cumsum = np.cumsum(probabilities)
        rand = np.random.random()
        for i, cumulative_sum in enumerate(cumsum):
            if rand <= cumulative_sum:
                return i
        # Fallback: If none of the cumulative sums match (unlikely),
        # it defaults to returning the last index.
        return len(probabilities) - 1

    def binary_tournament_selection(self, population, objectives):
        i, j = np.random.choice(len(population), 2, replace=False)

        if dominates(objectives[i], objectives[j]):
            return i
        elif dominates(objectives[j], objectives[i]):
            return j
        else:
            return np.random.choice([i, j])

    def credit_assignment(self, parent_objs, child_objs):
        reward = 0
        penalty = 0

        if dominates(parent_objs[0], parent_objs[1]):
            # Parent 0 dominates parent 1
            dominating_parent = parent_objs[0]
            for child_obj in child_objs:
                if dominates(dominating_parent, child_obj):
                    penalty += 1
                else:
                    reward += 1
        elif dominates(parent_objs[1], parent_objs[0]):
            # Parent 1 dominates parent 0
            dominating_parent = parent_objs[1]
            for child_obj in child_objs:
                if dominates(dominating_parent, child_obj):
                    penalty += 1
                else:
                    reward += 1
        else:
            # Parents are non-dominated to each other
            for child_obj in child_objs:
                # Child is rewarded if not dominated by both parents
                if dominates(parent_objs[0], child_obj) or dominates(parent_objs[1], child_obj):
                    penalty += 1
                else:
                    reward += 1

        return reward, penalty

    def update_opSelectProb(self, gen_in_lp):
        delta = 0.0001  # Avoid division by zero

        # Sum rewards and penalties for each operator over LP (Learning Period for AOS) generations
        total_rewards = np.sum(self.RD[:gen_in_lp], axis=0)
        total_penalties = np.sum(self.PN[:gen_in_lp], axis=0)

        # Check if the reward is zero and replace it with delta if true; otherwise, keep the original value
        safe_rewards = np.where(total_rewards == 0, delta, total_rewards)

        # Calculate operator performance score (selection probability basis)
        # High reward, low penalty -> value near 1
        # Low reward, high penalty -> value near 0
        operator_scores = total_rewards / (safe_rewards + total_penalties)

        total = np.sum(operator_scores)
        if total > 0:
            self.opSelectProb = operator_scores / total
        else:
            self.opSelectProb = np.ones(
                self.num_x_over_operators) / self.num_x_over_operators

        # Reset RD and PN matrices
        self.RD = np.zeros((self.lp, self.num_x_over_operators))
        self.PN = np.zeros((self.lp, self.num_x_over_operators))

    def environmental_selection(self, combined_pop, combined_objs):
        fronts = fast_non_dominated_sort(combined_pop, combined_objs)

        new_population = []
        new_objectives = []

        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                for idx in front:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_objs[idx])
            else:
                remaining = self.pop_size - len(new_population)
                distances = crowding_distance(combined_objs, front)

                sorted_front = sorted(
                    front, key=lambda x: distances[x], reverse=True)

                for idx in sorted_front[:remaining]:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_objs[idx])
                break

        return new_population, new_objectives

    def get_pareto_front(self, population, objectives):
        fronts = fast_non_dominated_sort(population, objectives)
        # Extract Only the First Front
        pareto_front = [[float(objectives[idx][0]), int(objectives[idx][1])]
                        for idx in fronts[0]]
        return pareto_front

    def run(self, seed=42):
        np.random.seed(seed)

        # ############### POPULATION INITIALIZATION ###############
        population = self.initialize_population()
        objectives = [self.evaluate(ind) for ind in population]
        generation_in_lp = 0
        reference_point = [100.0, float(self.n_features + 1)]
        print("Initial FEs: {}, Target: {}".format(self.n_fes, self.max_fes))

        while self.n_fes < self.max_fes:
            offspring_population = []
            offspring_objectives = []

            # Rewards/Penalties for this generation
            n_reward = np.zeros(self.num_x_over_operators)
            n_penalty = np.zeros(self.num_x_over_operators)

            # Generate N/2 pairs of offspring
            for _ in range(self.pop_size // 2):
                if self.n_fes >= self.max_fes:
                    break

                # Select crossover operator using roulette wheel
                operator_idx = self.roulette_wheel_selection(self.opSelectProb)
                crossover_op = CROSSOVER_OPERATORS[operator_idx]

                # ############### PARENT SELECTION ###############
                p1_idx = self.binary_tournament_selection(
                    population, objectives)
                p2_idx = self.binary_tournament_selection(
                    population, objectives)

                parent1 = population[p1_idx].copy()
                parent2 = population[p2_idx].copy()
                parent_objs = [objectives[p1_idx], objectives[p2_idx]]

                # ############### CROSSOVER ###############
                if np.random.random() < self.crossover_rate:
                    child1, child2 = crossover_op(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # ############### MUTATION ###############
                child1 = uniform_mutation(child1, self.mutation_rate)
                child2 = uniform_mutation(child2, self.mutation_rate)

                # Evaluate children
                child1_obj = self.evaluate(child1)
                child2_obj = self.evaluate(child2)
                child_objs = [child1_obj, child2_obj]

                # Credit assignment
                reward, penalty = self.credit_assignment(
                    parent_objs, child_objs
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

            # Update opSelectProb every LP generations
            if generation_in_lp >= self.lp:
                self.update_opSelectProb(generation_in_lp)
                generation_in_lp = 0

            # ############### SURVIVOR SELECTION ###############
            # Environmental selection (NSGA-II)
            combined_pop = population + offspring_population
            combined_objs = objectives + offspring_objectives
            population, objectives = self.environmental_selection(
                combined_pop, combined_objs)

            # Record history
            pareto_front = self.get_pareto_front(population, objectives)
            hv_ind = HV(ref_point=np.array(reference_point))
            hv = hv_ind(np.array(pareto_front)) if pareto_front else 0.0
            self.hv_history.append(hv)
            self.opSelectProb_history.append(self.opSelectProb.copy())

            self.generation += 1

            if self.generation % 10 == 0:
                print("Gen {}, FEs: {}, PF size: {}, HV: {:.4f}".format(
                    self.generation, self.n_fes, len(pareto_front), hv))

        # Extract final Pareto front
        final_pareto_front = self.get_pareto_front(population, objectives)

        return final_pareto_front, population


if __name__ == "__main__":

    DATASETS = {
        'DS02': ('DS02.csv', 'LungCancer', 10000),
        'DS04': ('DS04.csv', 'OpticalRecognition', 15000),
        'DS05': ('DS05.csv', 'MadelonValid', 30000),
        'DS07': ('DS07.csv', 'Har', 30000),
        'DS08': ('DS08.csv', 'HAPT', 30000),
        'DS10': ('DS10.csv', 'MultipleFeaturesDigit', 30000),
    }

    os.makedirs('results', exist_ok=True)

    all_results = {}

    for ds_id, (ds_file, ds_name, max_fitness_evaluations) in DATASETS.items():
        full_name = "{}_{}".format(ds_id, ds_name)
        print("\n" + "=" * 60)
        print("Processing: {}".format(full_name))
        print("=" * 60)

        try:
            X, y = load_dataset(ds_file)
            n_samples = X.shape[0]
            n_features = X.shape[1]
            n_classes = len(np.unique(y))

            print("Samples: {}, Features: {}, Classes: {}".format(
                n_samples, n_features, n_classes))

            all_fronts = []
            all_HV_histories = []
            all_opSelectProb_histories = []
            baselines = []
            run_times = []

            # Run experiments -
            # (Paper:   run 30 times per dataset -
            # This Code: run 3 times per dataset  (cause of time consumption))
            # Read More [Paper]: 4.2. Benchmark algorithms and parameter settings
            for run in range(3):
                seed = 42 + run * 100

                # Train-test split: (70-30)
                X_train, y_train, X_test, y_test = train_test_split(
                    X, y, seed=seed)

                baseline_error = compute_all_features_error(X_train, y_train)
                baselines.append(baseline_error)

                print("\nRun {}/3 (seed={})".format(run + 1, seed))
                print("  Baseline error (all features): {:.2f}%".format(
                    baseline_error))

                mobga = MOBGA_AOS(
                    n_features=n_features,
                    max_fes=max_fitness_evaluations,
                    pop_size=100,
                    crossover_rate=0.9,
                    lp=5
                )
                mobga.load_data(X_train, y_train)

                start_time = time.time()
                pareto_front, final_pop = mobga.run(seed=seed)
                run_time = time.time() - start_time

                all_fronts.append(pareto_front)
                all_HV_histories.append(mobga.hv_history)
                all_opSelectProb_histories.append(mobga.opSelectProb_history)
                run_times.append(run_time)

                print("  Pareto front size: {}".format(len(pareto_front)))
                print("  Run time: {:.2f}s".format(run_time))

            # Compute combined true Pareto front
            true_pf = merge_pareto_fronts(all_fronts)

            # Compute metrics
            reference_point = [100.0, float(n_features + 1)]

            # Using pymoo's built-in HV and IGD indicators instead of custom implementations.
            # HV measures the volume of objective space dominated by the Pareto front (higher = better).
            # IGD measures the average distance from the true front to the obtained front (lower = better).
            # pymoo handles edge cases, is numerically stable, and supports N-dimensional fronts.
            igd_ind = IGD(pf=np.array(true_pf))
            igd_values = [igd_ind(np.array(front)) if front else float(
                'inf') for front in all_fronts]
            hv_ind = HV(ref_point=np.array(reference_point))
            hv_values = [hv_ind(np.array(front))
                         if front else 0.0 for front in all_fronts]

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
            plot_hv_convergence(all_HV_histories, full_name,
                                'results/hv_{}.png'.format(ds_id))
            plot_opSelectProb_evolution(all_opSelectProb_histories, full_name,
                                        'results/opSelectProb_{}.png'.format(ds_id))

            # Store results with native Python types
            all_results[ds_id] = {
                'dataset': full_name,
                'n_features': int(n_features),
                'n_samples': int(n_samples),
                'n_classes': int(n_classes),
                'max_fes': int(max_fitness_evaluations),
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
