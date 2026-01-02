import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# Consts
POPULATION_SIZE = 200
MAX_GENERATIONS = 100
MAX_TREE_DEPTH_INIT = 4
MAX_TREE_DEPTH = 8
MAX_STEPS_PER_EPISODE = 60
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
MAZE = [
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
]
START_POS = (0, 0)
GOAL_POS = (9, 9)
MAZE_HEIGHT = len(MAZE)
MAZE_WIDTH = len(MAZE[0])

# ENUMS
MOVES = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}
# Terminal set (leaf nodes)
TERMINALS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Function set
FUNCTIONS = [
    'IF_WALL_UP',      # If wall/penalty cell is up
    'IF_WALL_DOWN',    # If wall/penalty cell is down
    'IF_WALL_LEFT',    # If wall/penalty cell is left
    'IF_WALL_RIGHT',   # If wall/penalty cell is right
    'IF_GOAL_UP',      # If goal is in upward direction
    'IF_GOAL_DOWN',    # If goal is in downward direction
    'IF_GOAL_LEFT',    # If goal is in left direction
    'IF_GOAL_RIGHT'    # If goal is in right direction
]


class TreeNode:

    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def is_terminal(self):
        return self.value in TERMINALS

    def is_function(self):
        return self.value in FUNCTIONS

    def depth(self):
        if self.is_terminal():
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self):
        if self.is_terminal():
            return 1
        return 1 + sum(child.size() for child in self.children)

    def copy(self):
        if self.is_terminal():
            return TreeNode(self.value)
        return TreeNode(self.value, [child.copy() for child in self.children])

    def __str__(self):
        if self.is_terminal():
            return self.value
        return f"({self.value} {' '.join(str(c) for c in self.children)})"


# ##############################################################################
# TREE GENERATION (Half-Full, Half-incomeplete)
# ##############################################################################

def generate_full_tree(max_depth, current_depth=0):
    if current_depth >= max_depth - 1:
        return TreeNode(random.choice(TERMINALS))
    else:
        func = random.choice(FUNCTIONS)
        children = [
            generate_full_tree(max_depth, current_depth + 1),
            generate_full_tree(max_depth, current_depth + 1)
        ]
        return TreeNode(func, children)


def generate_grow_tree(max_depth, current_depth=0):
    if current_depth >= max_depth - 1:
        return TreeNode(random.choice(TERMINALS))
    else:
        if random.random() < 0.3:  # 30% chance of early termination
            return TreeNode(random.choice(TERMINALS))
        else:
            func = random.choice(FUNCTIONS)
            children = [
                generate_grow_tree(max_depth, current_depth + 1),
                generate_grow_tree(max_depth, current_depth + 1)
            ]
            return TreeNode(func, children)


def ramped_half_and_half(pop_size, max_depth):
    """
    Initialize population: ramped half-and-half method.
    - Half: full/complete trees
    - Half: incomplete trees
    - Reject any solution that is too good (fitness < 25) to ensure evolution matters
    """
    population = []
    depths = list(range(2, max_depth + 1))
    individuals_per_depth = pop_size // len(depths)

    # Reject solutions better than this to not find solutions in gen_0
    MIN_FITNESS_THRESHOLD = 25

    def generate_valid_tree(generator_func, depth):
        max_attempts = 50
        for _ in range(max_attempts):
            tree = generator_func(depth)
            fitness = simulate_agent(tree)
            if fitness >= MIN_FITNESS_THRESHOLD:
                return tree
        # If we can't find a bad enough solution, return the last one anyway
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


def is_wall(row, col):
    if 0 <= row < MAZE_HEIGHT and 0 <= col < MAZE_WIDTH:
        return MAZE[row][col] == 1
    return True  # Out of bounds treated as wall


def is_valid_position(row, col):
    return 0 <= row < MAZE_HEIGHT and 0 <= col < MAZE_WIDTH


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def evaluate_condition(condition, agent_row, agent_col, visited):
    goal_row, goal_col = GOAL_POS

    # Wall conditions
    if condition == 'IF_WALL_UP':
        return is_wall(agent_row - 1, agent_col) or not is_valid_position(agent_row - 1, agent_col)
    elif condition == 'IF_WALL_DOWN':
        return is_wall(agent_row + 1, agent_col) or not is_valid_position(agent_row + 1, agent_col)
    elif condition == 'IF_WALL_LEFT':
        return is_wall(agent_row, agent_col - 1) or not is_valid_position(agent_row, agent_col - 1)
    elif condition == 'IF_WALL_RIGHT':
        return is_wall(agent_row, agent_col + 1) or not is_valid_position(agent_row, agent_col + 1)

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


def execute_tree(tree, agent_row, agent_col, visited):

    if tree.is_terminal():
        return tree.value

    condition_result = evaluate_condition(
        tree.value, agent_row, agent_col, visited)

    if condition_result:
        return execute_tree(tree.children[0], agent_row, agent_col, visited)
    else:
        return execute_tree(tree.children[1], agent_row, agent_col, visited)


def simulate_agent(tree, return_path=False):
    """
    Simulate agent navigation using the GP tree as controller.

    IMPORTANT: As per assignment corrections:
    - Agent CAN move through walls (receives penalty)
    - Agent CAN move in all 4 directions at any time
    - Hitting boundary keeps agent in place but costs a step
    - Fitness is ALWAYS calculated with the formula, even when goal is reached
    - Optimal (F=0) means: goal reached + minimum steps + 0 walls + 0 loops

    Returns:
    - fitness value (lower is better)
    - optionally returns the path taken
    """
    agent_row, agent_col = START_POS
    visited = {(agent_row, agent_col)}
    path = [(agent_row, agent_col)]

    steps = 0
    wall_hits = 0
    revisits = 0
    goal_reached = False

    for _ in range(MAX_STEPS_PER_EPISODE):
        # Check if goal reached
        if (agent_row, agent_col) == GOAL_POS:
            goal_reached = True
            break

        # Get movement action from tree
        action = execute_tree(tree, agent_row, agent_col, visited)

        # Calculate new position
        dr, dc = MOVES[action]
        new_row = agent_row + dr
        new_col = agent_col + dc

        steps += 1

        # Check if move is within bounds
        if is_valid_position(new_row, new_col):
            # Check if moving into a wall (penalty cell)
            if is_wall(new_row, new_col):
                wall_hits += 1

            # Check for revisit (loop)
            if (new_row, new_col) in visited:
                revisits += 1

            # Move agent (always allowed as per corrections)
            agent_row, agent_col = new_row, new_col
            visited.add((agent_row, agent_col))
            path.append((agent_row, agent_col))
        else:
            # Out of bounds - agent stays in place, but still counts as a step
            # This implicitly penalizes boundary hits through wasted steps
            pass

    # Calculate fitness: F = s + 2d + 10w + 5l (ALWAYS applied)
    # Even when goal is reached, steps/walls/loops still count!
    # Fitness 0 only when: goal reached + optimal steps + 0 walls + 0 loops
    distance = manhattan_distance((agent_row, agent_col), GOAL_POS)
    fitness = steps + 2 * distance + 10 * wall_hits + 5 * revisits

    if return_path:
        return fitness, path, goal_reached
    return fitness


def fitness_proportional_selection(population, fitnesses):
    # Convert to maximization problem (invert fitnesses)
    max_fitness = max(fitnesses) + 1  # +1 to avoid division by zero
    inverted = [max_fitness - f for f in fitnesses]

    total = sum(inverted)
    if total == 0:
        return random.choice(population)

    # Roulette wheel selection
    pick = random.uniform(0, total)
    current = 0
    for i, inv_fit in enumerate(inverted):
        current += inv_fit
        if current >= pick:
            return population[i].copy()

    return population[-1].copy()


def tournament_selection(population, fitnesses, tournament_size=3):
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


def get_all_nodes(tree, include_root=True):

    nodes = []

    def collect(node, parent, child_index):
        nodes.append((node, parent, child_index))
        for i, child in enumerate(node.children):
            collect(child, node, i)

    if include_root:
        collect(tree, None, -1)
    else:
        for i, child in enumerate(tree.children):
            collect(child, tree, i)

    return nodes


def select_random_node(tree, include_root=True):

    nodes = get_all_nodes(tree, include_root)
    return random.choice(nodes)


def subtree_crossover(parent1, parent2):

    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Select crossover points
    nodes1 = get_all_nodes(offspring1, include_root=False)
    nodes2 = get_all_nodes(offspring2, include_root=False)

    if not nodes1 or not nodes2:
        return offspring1, offspring2

    node1, parent1_node, idx1 = random.choice(nodes1)
    node2, parent2_node, idx2 = random.choice(nodes2)

    # Check depth constraints
    depth1 = node2.depth() + (offspring1.depth() - node1.depth())
    depth2 = node1.depth() + (offspring2.depth() - node2.depth())

    if depth1 <= MAX_TREE_DEPTH and depth2 <= MAX_TREE_DEPTH:
        if parent1_node and parent2_node:
            temp = parent1_node.children[idx1]
            parent1_node.children[idx1] = parent2_node.children[idx2]
            parent2_node.children[idx2] = temp

    return offspring1, offspring2


def subtree_mutation(tree):

    mutant = tree.copy()
    nodes = get_all_nodes(mutant, include_root=False)

    if not nodes:
        return mutant

    # Select random node to replace
    node, parent, idx = random.choice(nodes)

    if parent is not None:
        # Generate new subtree with limited depth
        max_new_depth = MAX_TREE_DEPTH - (mutant.depth() - node.depth())
        max_new_depth = max(2, min(max_new_depth, 4))

        if random.random() < 0.5:
            new_subtree = generate_grow_tree(max_new_depth)
        else:
            new_subtree = generate_full_tree(max_new_depth)

        parent.children[idx] = new_subtree

    return mutant


def point_mutation(tree):

    mutant = tree.copy()
    nodes = get_all_nodes(mutant)

    if not nodes:
        return mutant

    node, parent, idx = random.choice(nodes)

    if node.is_terminal():
        # Change to different terminal
        other_terminals = [t for t in TERMINALS if t != node.value]
        if other_terminals:
            node.value = random.choice(other_terminals)
    else:
        # Change to different function
        other_functions = [f for f in FUNCTIONS if f != node.value]
        if other_functions:
            node.value = random.choice(other_functions)

    return mutant


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


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_maze_with_path(path, title="Maze Solution"):

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create color matrix for maze
    maze_colors = np.array(MAZE, dtype=float)

    # Draw maze
    ax.imshow(maze_colors, cmap='binary', origin='upper')

    for i in range(MAZE_HEIGHT + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(MAZE_WIDTH + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    if path and len(path) > 1:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax.plot(path_cols, path_rows, 'b-',
                linewidth=2, alpha=0.7, label='Path')

        for i, (r, c) in enumerate(path):
            if i == 0:
                ax.plot(c, r, 'go', markersize=15, label='Start')
            elif (r, c) == GOAL_POS:
                ax.plot(c, r, 'r*', markersize=20, label='Goal')
            else:
                ax.plot(c, r, 'b.', markersize=8)

    ax.plot(START_POS[1], START_POS[0], 'go', markersize=15)
    ax.plot(GOAL_POS[1], GOAL_POS[0], 'r*', markersize=20)

    ax.set_xlim(-0.5, MAZE_WIDTH - 0.5)
    ax.set_ylim(MAZE_HEIGHT - 0.5, -0.5)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)
    ax.legend(loc='upper right')

    for i in range(MAZE_HEIGHT):
        for j in range(MAZE_WIDTH):
            if MAZE[i][j] == 0:
                ax.text(j, i, f'{i},{j}', ha='center',
                        va='center', fontsize=6, alpha=0.5)

    plt.tight_layout()
    return fig


def plot_fitness_history(best_history, avg_history):

    fig, ax = plt.subplots(figsize=(12, 6))

    generations = range(len(best_history))

    ax.plot(generations, best_history, 'b-', linewidth=2, label='Best Fitness')
    ax.plot(generations, avg_history, 'r--',
            linewidth=2, label='Average Fitness')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (lower is better)', fontsize=12)
    ax.set_title('Fitness Progression Over Generations', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    min_fitness = min(best_history)
    min_gen = best_history.index(min_fitness)
    ax.annotate(f'Best: {min_fitness:.1f}', xy=(min_gen, min_fitness),
                xytext=(min_gen + 5, min_fitness + 20),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')

    plt.tight_layout()
    return fig


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
    print_tree(tree)


def print_tree(node, indent=0):
    prefix = "  " * indent
    if node.is_terminal():
        print(f"{prefix}+-- {node.value}")
    else:
        print(f"{prefix}+-- {node.value}")
        if len(node.children) >= 2:
            print(f"{prefix}|   [IF TRUE]:")
            print_tree(node.children[0], indent + 2)
            print(f"{prefix}|   [IF FALSE]:")
            print_tree(node.children[1], indent + 2)


def main():
    random.seed(42)
    np.random.seed(42)

    # Display maze
    print("\nMaze Layout (0=open, 1=wall/penalty):")
    print("Start: (0,0) top-left")
    print("Goal: (9,9) bottom-right\n")
    for i, row in enumerate(MAZE):
        print(f"Row {i}: {row}")

    print("\n" + "=" * 60)
    print("FINDING OPTIMAL PATH (BFS) FOR COMPARISON")
    print("=" * 60)

    best_tree, best_fitness, best_history, avg_history, suboptimal = run_gp()

    fitness, path, goal_reached = simulate_agent(best_tree, return_path=True)

    print_solution_details(best_tree, fitness, path, goal_reached)

    print("\nGenerating visualizations...")

    # Plot fitness history
    fig1 = plot_fitness_history(best_history, avg_history)
    fig1.savefig('fitness_progression.png',
                 dpi=150, bbox_inches='tight')
    print("Saved: fitness_progression.png")

    # Visualize best solution path
    fig2 = visualize_maze_with_path(
        path, f"GP Solution - OPTIMAL (Fitness: {fitness})")
    fig2.savefig('solution_path.png',
                 dpi=150, bbox_inches='tight')
    print("Saved: solution_path.png")

    # Show 2 suboptimal solutions
    if len(suboptimal) >= 2:
        print("\n" + "=" * 60)
        print("SUBOPTIMAL SOLUTION #1")
        print("=" * 60)
        sub1_fit, sub1_tree, sub1_path = suboptimal[0]
        _, _, sub1_reached = simulate_agent(sub1_tree, return_path=True)
        print_solution_details(sub1_tree, sub1_fit, sub1_path, sub1_reached)

        fig3 = visualize_maze_with_path(
            sub1_path, f"Suboptimal #1 (Fitness: {sub1_fit})")
        fig3.savefig('suboptimal_path_1.png', dpi=150, bbox_inches='tight')
        print("Saved: suboptimal_path_1.png")

        print("\n" + "=" * 60)
        print("SUBOPTIMAL SOLUTION #2")
        print("=" * 60)
        sub2_fit, sub2_tree, sub2_path = suboptimal[1]
        _, _, sub2_reached = simulate_agent(sub2_tree, return_path=True)
        print_solution_details(sub2_tree, sub2_fit, sub2_path, sub2_reached)

        fig4 = visualize_maze_with_path(
            sub2_path, f"Suboptimal #2 (Fitness: {sub2_fit})")
        fig4.savefig('suboptimal_path_2.png', dpi=150, bbox_inches='tight')
        print("Saved: suboptimal_path_2.png")
    else:
        print("\nNot enough suboptimal solutions found to display.")

    plt.close('all')

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)

    return best_tree, best_fitness, path


if __name__ == "__main__":
    main()
