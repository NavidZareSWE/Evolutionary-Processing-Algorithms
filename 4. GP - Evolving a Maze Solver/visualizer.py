import numpy as np
import matplotlib.pyplot as plt
from constants import MAZE, MAZE_HEIGHT, MAZE_WIDTH, START_POS, GOAL_POS


class Visualizer:

    def visualize_maze_with_path(path, title="Maze Solution", filename=None):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create color matrix for maze
        maze_colors = np.array(MAZE, dtype=float)

        # Draw maze
        ax.imshow(maze_colors, cmap='binary', origin='upper')

        # Grid lines
        for i in range(MAZE_HEIGHT + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(MAZE_WIDTH + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)

        # Draw path
        if path and len(path) > 1:
            path_rows = [p[0] for p in path]
            path_cols = [p[1] for p in path]
            ax.plot(path_cols, path_rows, 'b-',
                    linewidth=2, alpha=0.7, label='Path')

            # Mark special positions
            for i, (r, c) in enumerate(path):
                if i == 0:
                    ax.plot(c, r, 'go', markersize=15, label='Start')
                elif (r, c) == GOAL_POS:
                    ax.plot(c, r, 'r*', markersize=20, label='Goal')
                else:
                    ax.plot(c, r, 'b.', markersize=8)

        # Always show start and goal
        ax.plot(START_POS[1], START_POS[0], 'go', markersize=15)
        ax.plot(GOAL_POS[1], GOAL_POS[0], 'r*', markersize=20)

        # Configure plot
        ax.set_xlim(-0.5, MAZE_WIDTH - 0.5)
        ax.set_ylim(MAZE_HEIGHT - 0.5, -0.5)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title)
        ax.legend(loc='upper right')

        # Add cell coordinates
        for i in range(MAZE_HEIGHT):
            for j in range(MAZE_WIDTH):
                if MAZE[i][j] == 0:
                    ax.text(j, i, f'{i},{j}', ha='center', va='center',
                            fontsize=6, alpha=0.5)

        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")

        return fig

    def plot_fitness_history(best_history, avg_history, filename=None):

        fig, ax = plt.subplots(figsize=(12, 6))

        generations = range(len(best_history))

        ax.plot(generations, best_history, 'b-',
                linewidth=2, label='Best Fitness')
        ax.plot(generations, avg_history, 'r--',
                linewidth=2, label='Average Fitness')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness (lower is better)', fontsize=12)
        ax.set_title('Fitness Progression Over Generations', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate best fitness
        min_fitness = min(best_history)
        min_gen = best_history.index(min_fitness)
        ax.annotate(f'Best: {min_fitness:.1f}',
                    xy=(min_gen, min_fitness),
                    xytext=(min_gen + 5, min_fitness + 20),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=10, color='blue')

        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")

        return fig

    def close_all_figures():
        plt.close('all')
