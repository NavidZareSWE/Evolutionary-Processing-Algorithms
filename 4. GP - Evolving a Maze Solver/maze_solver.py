import numpy as np
import random
from constants import (
    POPULATION_SIZE, MAX_GENERATIONS, MAX_TREE_DEPTH_INIT, MAX_TREE_DEPTH,
    CROSSOVER_RATE, MUTATION_RATE, MAX_STEPS_PER_EPISODE, MIN_FITNESS_THRESHOLD,
    MAZE, START_POS, GOAL_POS, MAZE_HEIGHT, MAZE_WIDTH,
    MOVES, TERMINALS, FUNCTIONS
)
from tree import (
    TreeNode, generate_full_tree, generate_grow_tree, get_all_nodes
)
from visualizer import Visualizer


def ramped_half_and_half(pop_size, max_depth):
    """
    Initialize population: ramped half-and-half method.
    - Half: full/complete trees
    - Half: incomplete trees
    - Reject any solution that is ***too good*** (fitness < 25)
    """
    population = []
    depths = list(range(2, max_depth + 1))
    individuals_per_depth = pop_size // len(depths)

    def generate_valid_tree(generator_func, depth):
        max_attempts = 50
        for _ in range(max_attempts):
            tree = generator_func(depth)
            fitness = simulate_agent(tree)
            # You want trees that are not TOO good,
            # My only solution to not find the goal in generetion 0
            if fitness >= MIN_FITNESS_THRESHOLD:
                return tree
        # If we can't find a bad enough solution
        # return the last one anyway
        return tree

    for depth in depths:
        for _ in range(individuals_per_depth // 2):
            population.append(generate_valid_tree(generate_full_tree, depth))
        for _ in range(individuals_per_depth // 2):
            population.append(generate_valid_tree(generate_grow_tree, depth))

    while len(population) < pop_size:
        if random.random() < 0.5:
            population.append(generate_valid_tree(
                generate_full_tree, random.choice(depths)))
        else:
            population.append(generate_valid_tree(
                generate_grow_tree, random.choice(depths)))

    return population


# ##############################################################################
# Utility Functions
# ##############################################################################
def is_within_bounds(row, col):
    return 0 <= row < MAZE_HEIGHT and 0 <= col < MAZE_WIDTH


def is_wall(row, col):
    if is_within_bounds(row, col):
        return MAZE[row][col] == 1
    return True  # Out of bounds -> treated as wall


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# ##############################################################################
# TREE EVALUATION
# ##############################################################################

def evaluate_condition(condition, agent_row, agent_col):
    goal_row, goal_col = GOAL_POS

    # Wall conditions
    if condition == 'IF_WALL_UP':
        return is_wall(agent_row - 1, agent_col)
    elif condition == 'IF_WALL_DOWN':
        return is_wall(agent_row + 1, agent_col)
    elif condition == 'IF_WALL_LEFT':
        return is_wall(agent_row, agent_col - 1)
    elif condition == 'IF_WALL_RIGHT':
        return is_wall(agent_row, agent_col + 1)

    # Goal conditions
    elif condition == 'IF_GOAL_UP':
        return goal_row < agent_row
    elif condition == 'IF_GOAL_DOWN':
        return goal_row > agent_row
    elif condition == 'IF_GOAL_LEFT':
        return goal_col < agent_col
    elif condition == 'IF_GOAL_RIGHT':
        return goal_col > agent_col

    return False


def execute_tree(tree, agent_row, agent_col):
    if tree.is_terminal():
        return tree.value

    condition_result = evaluate_condition(
        tree.value, agent_row, agent_col)

    if condition_result:
        return execute_tree(tree.children[0], agent_row, agent_col)
    else:
        return execute_tree(tree.children[1], agent_row, agent_col)


# ##############################################################################
# SIMULATION
# ##############################################################################

def simulate_agent(tree, return_path=False):
    """
    IMPORTANT (assignment corrections):
    - Agent CAN move through walls (receives penalty)
    - Agent CAN move in all 4 directions at any time
    - Hitting boundary keeps agent in place but costs a step
    - Fitness is ALWAYS calculated with the formula, even when goal is reached
    - Optimal (F=0) means: goal reached + minimum steps + 0 walls + 0 loops

    Returns:
    - fitness value (LOWER is better)
    - optionally returns the path taken
    """
    agent_row, agent_col = START_POS
    # visited = []
    path = [(agent_row, agent_col)]

    steps = 0
    wall_hits = 0
    revisits = 0
    goal_reached = False

    for _ in range(MAX_STEPS_PER_EPISODE):
        if (agent_row, agent_col) == GOAL_POS:
            goal_reached = True
            break

        action = execute_tree(tree, agent_row, agent_col)

        # Calculate new position
        dr, dc = MOVES[action]
        new_row = agent_row + dr
        new_col = agent_col + dc

        steps += 1

        # Check if move is within bounds
        if is_within_bounds(new_row, new_col):
            if is_wall(new_row, new_col):
                wall_hits += 1

            # Check for revisit (loop)
            # if (new_row, new_col) in visited:
            #     revisits += 1

            agent_row, agent_col = new_row, new_col
            # visited.add((agent_row, agent_col))
            path.append((agent_row, agent_col))
        else:
            # Out of bounds - agent stays in place, but still counts as a step
            # This implicitly penalizes boundary hits through wasted steps
            wall_hits += 1
            pass

    # Calculate fitness: F = s + 2d + 10w + 5l (ALWAYS applied)
    # Even when goal is reached, steps/walls/loops still count!
    # Fitness 0 only when: goal reached + optimal steps + 0 walls + 0 loops
    distance = manhattan_distance((agent_row, agent_col), GOAL_POS)
    fitness = steps + 2 * distance + 10 * wall_hits + 5 * revisits

    if return_path:
        return fitness, path, goal_reached
    return fitness


# ##############################################################################
# SELECTION OPERATORS
# ##############################################################################

def fitness_proportional_selection(population, fitnesses):
    # Convert to maximization problem (invert fitnesses)
    max_fitness = max(fitnesses) + 1  # +1 to avoid division by zero
    inverted = [max_fitness - f for f in fitnesses]

    total = sum(inverted)
    # All have the same fitness
    if total == 0:
        return random.choice(population)

    # Roulette wheel selection
    pick = random.uniform(0, total)
    current = 0
    for i, indiv_fit in enumerate(inverted):
        current += indiv_fit
        if current >= pick:
            return population[i].copy()

    return population[-1].copy()


def tournament_selection(population, fitnesses, tournament_size=3):
    all_indices = np.arange(len(population))

    if len(all_indices) < tournament_size:
        selected_indices = all_indices
    else:
        selected_indices = np.random.choice(
            all_indices, size=tournament_size, replace=False)

    best_fitness = float('inf')  # Start high since lower is better
    best_indiv = None

    for idx in selected_indices:
        indiv = population[idx]
        fitness = fitnesses[idx]
        if fitness < best_fitness:  # Lower is better in maze solver
            best_fitness = fitness
            best_indiv = indiv

    return best_indiv.copy()


# ##############################################################################
# GENETIC OPERATORS
# ##############################################################################

def subtree_crossover(parent1, parent2):
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Select crossover points
    nodes1 = get_all_nodes(offspring1, include_root=False)
    nodes2 = get_all_nodes(offspring2, include_root=False)

    # Prevents crossover on empty trees
    if not nodes1 or not nodes2:
        return offspring1, offspring2

    node1, parent1_node, idx1 = random.choice(nodes1)
    node2, parent2_node, idx2 = random.choice(nodes2)

    # Compute new tree depth:
    # inserted subtree depth + (original tree depth - replaced subtree depth)
    depth1 = node2.subtree_depth() + (offspring1.subtree_depth() - node1.subtree_depth())
    depth2 = node1.subtree_depth() + (offspring2.subtree_depth() - node2.subtree_depth())

    if depth1 <= MAX_TREE_DEPTH and depth2 <= MAX_TREE_DEPTH:
        if parent1_node and parent2_node:
            temp = parent1_node.children[idx1]
            parent1_node.children[idx1] = parent2_node.children[idx2]
            parent2_node.children[idx2] = temp

    return offspring1, offspring2


def subtree_mutation(tree):

    to_mutate = tree.copy()
    nodes = get_all_nodes(to_mutate, include_root=False)

    if not nodes:
        return to_mutate

    node, parent, idx = random.choice(nodes)

    # Node being replaced is not Root
    if parent is not None:
        max_new_depth = MAX_TREE_DEPTH - \
            (to_mutate.subtree_depth() - node.subtree_depth())
        max_new_depth = max(2, min(max_new_depth, 4))

        if random.random() < 0.5:
            new_subtree = generate_grow_tree(max_new_depth)
        else:
            new_subtree = generate_full_tree(max_new_depth)

        parent.children[idx] = new_subtree

    return to_mutate


def point_mutation(tree):
    to_mutate = tree.copy()
    nodes = get_all_nodes(to_mutate)

    if not nodes:
        return to_mutate

    node, __, ___ = random.choice(nodes)

    if node.is_terminal():
        other_terminals = [t for t in TERMINALS if t != node.value]
        if other_terminals:
            node.value = random.choice(other_terminals)
    else:
        other_functions = [f for f in FUNCTIONS if f != node.value]
        if other_functions:
            node.value = random.choice(other_functions)

    return to_mutate


# ##############################################################################
# MAIN GP ALGORITHM
# ##############################################################################

def run_gp():
    """
    - Ramped half-and-half initialization
    - Fitness proportional parent selection
    - Subtree crossover
    - Subtree and point mutation
    - Generational survivor selection
    """
    print("=" * 60)
    print("GENETIC PROGRAMMING MAZE SOLVER")
    print("=" * 60)
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Max Generations: {MAX_GENERATIONS}")
    print(f"Crossover Rate: {CROSSOVER_RATE}")
    print(f"Mutation Rate: {MUTATION_RATE}")
    print(f"Max Steps per Episode: {MAX_STEPS_PER_EPISODE}")
    print("=" * 60)

    # Initialize population using ramped half-and-half
    print("\nInitializing population (ramped half-and-half)...")
    population = ramped_half_and_half(POPULATION_SIZE, MAX_TREE_DEPTH_INIT)

    # Statistics tracking
    best_fitness_history = []
    avg_fitness_history = []
    best_individual_ever = None
    best_fitness_ever = float('inf')

    # Track suboptimal solutions (solutions that reach goal but aren't optimal)
    suboptimal_solutions = []

    # Main evolutionary loop
    for generation in range(MAX_GENERATIONS):
        # Evaluate fitness for all individuals
        fitnesses = [simulate_agent(ind) for ind in population]

        # Track statistics
        best_fitness = min(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_idx = fitnesses.index(best_fitness)

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        # Update best ever
        if best_fitness < best_fitness_ever:
            best_fitness_ever = best_fitness
            best_individual_ever = population[best_idx].copy()

        # Collect suboptimal solutions (reach goal but not optimal)
        if generation % 20 == 0:
            for i, ind in enumerate(population):
                fit, path, reached = simulate_agent(ind, return_path=True)
                if reached and fit > best_fitness_ever and fit < 100:
                    already_have = any(
                        s[0] == fit for s in suboptimal_solutions)
                    if not already_have and len(suboptimal_solutions) < 10:
                        suboptimal_solutions.append((fit, ind.copy(), path))

        # Print progress every 10 generations
        if generation % 10 == 0 or generation == MAX_GENERATIONS - 1:
            print(f"Gen {generation:3d}: Best={best_fitness:6.1f}, Avg={avg_fitness:6.1f}, "
                  f"Best Ever={best_fitness_ever:6.1f}")

        # Check termination - optimal solution found
        if best_fitness_ever == 0:
            print(
                f"\n*** OPTIMAL SOLUTION FOUND at generation {generation}! ***")
            break

        # Create next generation
        new_population = []

        # Elitism: keep best individual
        new_population.append(population[best_idx].copy())

        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            # Select parents using fitness proportional selection
            parent1 = fitness_proportional_selection(population, fitnesses)
            parent2 = fitness_proportional_selection(population, fitnesses)

            # Crossover
            if random.random() < CROSSOVER_RATE:
                offspring1, offspring2 = subtree_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()

            # Mutation
            if random.random() < MUTATION_RATE:
                if random.random() < 0.5:
                    offspring1 = subtree_mutation(offspring1)
                else:
                    offspring1 = point_mutation(offspring1)

            if random.random() < MUTATION_RATE:
                if random.random() < 0.5:
                    offspring2 = subtree_mutation(offspring2)
                else:
                    offspring2 = point_mutation(offspring2)

            new_population.append(offspring1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(offspring2)

        population = new_population

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)

    # Sort suboptimal by fitness (higher = worse)
    suboptimal_solutions.sort(key=lambda x: x[0], reverse=True)

    return best_individual_ever, best_fitness_ever, best_fitness_history, avg_fitness_history, suboptimal_solutions


# ##############################################################################
# OUTPUT HELPERS
# ##############################################################################

def print_solution_details(tree, fitness, path, goal_reached):

    print("\n" + "=" * 60)
    print("BEST SOLUTION DETAILS")
    print("=" * 60)

    print(f"\nFitness: {fitness}")
    print(f"Goal Reached: {'Yes' if goal_reached else 'No'}")
    print(f"Path Length: {len(path)} steps")

    # Count statistics
    wall_hits = sum(1 for r, c in path if MAZE[r][c] == 1)
    unique_cells = len(set(path))
    revisits = len(path) - unique_cells

    print(f"Wall Hits: {wall_hits}")
    print(f"Unique Cells Visited: {unique_cells}")
    print(f"Revisits (Loops): {revisits}")

    if goal_reached:
        final_pos = path[-1]
    else:
        final_pos = path[-1] if path else START_POS

    distance = manhattan_distance(final_pos, GOAL_POS)
    print(f"Final Position: {final_pos}")
    print(f"Distance to Goal: {distance}")

    # Print path
    print("\nMove Sequence:")
    moves = []
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        dr = curr[0] - prev[0]
        dc = curr[1] - prev[1]

        for move_name, (mr, mc) in MOVES.items():
            if dr == mr and dc == mc:
                moves.append(move_name)
                break

    for i in range(0, len(moves), 10):
        print("  " + " -> ".join(moves[i:i+10]))

    print("\nTree Structure:")
    tree.print_tree()


# ##############################################################################
# MAIN
# ##############################################################################

def main():
    random.seed(42)
    np.random.seed(42)

    # Display maze
    print("\nMaze Layout (0=open, 1=wall/penalty):")
    print("Start: (0,0) top-left")
    print("Goal: (9,9) bottom-right\n")
    for i, row in enumerate(MAZE):
        print(f"Row {i}: {row}")

    best_tree, best_fitness, best_history, avg_history, suboptimal = run_gp()

    fitness, path, goal_reached = simulate_agent(best_tree, return_path=True)

    print_solution_details(best_tree, fitness, path, goal_reached)

    print("\nGenerating visualizations...")

    # Plot fitness history
    fig1 = Visualizer.plot_fitness_history(
        best_history, avg_history, 'fitness_progression.png')

    # Visualize best solution path
    fig2 = Visualizer.visualize_maze_with_path(
        path, f"GP Solution - OPTIMAL (Fitness: {fitness})", 'solution_path.png')

    # Show 2 suboptimal solutions
    if len(suboptimal) >= 2:
        print("\n" + "=" * 60)
        print("SUBOPTIMAL SOLUTION #1")
        print("=" * 60)
        sub1_fit, sub1_tree, sub1_path = suboptimal[0]
        _, _, sub1_reached = simulate_agent(sub1_tree, return_path=True)
        print_solution_details(sub1_tree, sub1_fit, sub1_path, sub1_reached)

        fig3 = Visualizer.visualize_maze_with_path(
            sub1_path, f"Suboptimal #1 (Fitness: {sub1_fit})", 'suboptimal_path_1.png')

        print("\n" + "=" * 60)
        print("SUBOPTIMAL SOLUTION #2")
        print("=" * 60)
        sub2_fit, sub2_tree, sub2_path = suboptimal[1]
        _, _, sub2_reached = simulate_agent(sub2_tree, return_path=True)
        print_solution_details(sub2_tree, sub2_fit, sub2_path, sub2_reached)

        fig4 = Visualizer.visualize_maze_with_path(
            sub2_path, f"Suboptimal #2 (Fitness: {sub2_fit})", 'suboptimal_path_2.png')
    else:
        print("\nNot enough suboptimal solutions found to display.")

    Visualizer.close_all_figures()

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)

    return best_tree, best_fitness, path


if __name__ == "__main__":
    main()
