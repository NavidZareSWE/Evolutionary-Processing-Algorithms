import numpy as np
from pymoo.indicators.hv import HV
from operators import CROSSOVER_OPERATORS, uniform_mutation
from nsga2 import dominates, fast_non_dominated_sort, crowding_distance
from utils import normalize_data, cross_validation_error


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
