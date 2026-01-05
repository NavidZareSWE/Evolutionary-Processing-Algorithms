# Genetic Programming Parameters
POPULATION_SIZE = 200
MAX_GENERATIONS = 100
MAX_TREE_DEPTH_INIT = 4
MAX_TREE_DEPTH = 8
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1

# Simulation Parameters
MAX_STEPS_PER_EPISODE = 60
MIN_FITNESS_THRESHOLD = 25  # Reject solutions better than this in initial population

# Maze Configuration
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

# Movement Definitions
MOVES = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# GP Terminal Set (leaf nodes)
TERMINALS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# GP Function Set (internal nodes)
FUNCTIONS = [
    'IF_WALL_UP',
    'IF_WALL_DOWN',
    'IF_WALL_LEFT',
    'IF_WALL_RIGHT',
    'IF_GOAL_UP',
    'IF_GOAL_DOWN',
    'IF_GOAL_LEFT',
    'IF_GOAL_RIGHT'
]
