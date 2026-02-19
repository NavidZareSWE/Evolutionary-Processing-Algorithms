import numpy as np


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
        ax.set_title(
            'MOBGA-AOS Pareto Front: {}'.format(dataset_name), fontsize=14)
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
        ax.set_title('Hypervolume Convergence: {}'.format(
            dataset_name), fontsize=14)
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


def plot_opSelectProb_evolution(opSelectProb_histories, dataset_name, save_path=None):
    """Plot evolution of Operator Selection Probabilities."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Use first run's opSelectProb history
        if opSelectProb_histories and opSelectProb_histories[0]:
            opSelectProb_array = np.array(opSelectProb_histories[0])
            generations = range(1, len(opSelectProb_array) + 1)

            for i, name in enumerate(OPERATOR_NAMES):
                ax.plot(generations,
                        opSelectProb_array[:, i], label=name, linewidth=2)

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Selection Probability', fontsize=12)
        ax.set_title('Operator Selection Probability Evolution: {}'.format(
            dataset_name), fontsize=14)
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
