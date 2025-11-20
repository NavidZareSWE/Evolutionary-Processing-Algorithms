import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Close all plots to save files instead of showing them
plt.ion()  # Turn on interactive mode


def _save_and_close(filename):
    """Helper function to save and close plots"""
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filename}")
    plt.show(block=True)  # Display the plot
    plt.pause(0.1)


def show_all_plots_and_wait():
    """
    Display all generated plots and wait for user to close them.
    Call this at the end of your experiment suite.
    """
    print("\n" + "="*80)
    print("📊 ALL PLOTS ARE NOW DISPLAYED")
    print("="*80)
    print("💡 TIP: Review all plot windows.")
    print("⚠️  Close all plot windows to exit the program.")
    print("="*80 + "\n")

    plt.show(block=True)  # Block until all plot windows are closed
    print("✅ All plots closed. Exiting...")

# ========================= TASK 2: PARAMETER SENSITIVITY =========================


def plot_parameter_sensitivity(results_dict, param_name, param_values):
    """
    Plot convergence curves and statistics for parameter sensitivity analysis.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Average Convergence Curves
    ax1 = fig.add_subplot(gs[0, :])
    for param_val in param_values:
        runs = results_dict[param_val]
        all_histories = [r['fitness_history'] for r in runs]
        max_len = max(len(h) for h in all_histories)

        padded_histories = []
        for hist in all_histories:
            padded = list(hist) + [hist[-1]] * (max_len - len(hist))
            padded_histories.append(padded)

        avg_history = np.mean(padded_histories, axis=0)
        std_history = np.std(padded_histories, axis=0)

        generations = range(len(avg_history))
        ax1.plot(generations, avg_history,
                 label=f'{param_name}={param_val}', linewidth=2)
        ax1.fill_between(generations,
                         avg_history - std_history,
                         avg_history + std_history,
                         alpha=0.2)

    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title(
        f'Convergence Curves: {param_name} Sensitivity', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--',
                linewidth=1, label='Optimal Fitness')

    # 2. Success Rate Bar Chart
    ax2 = fig.add_subplot(gs[1, 0])
    success_rates = []
    for param_val in param_values:
        runs = results_dict[param_val]
        success_rate = sum(1 for r in runs if r['success']) / len(runs) * 100
        success_rates.append(success_rate)

    bars = ax2.bar(range(len(param_values)), success_rates,
                   color='steelblue', alpha=0.7)
    ax2.set_xlabel(param_name, fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(param_values)))
    ax2.set_xticklabels(param_values)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    # 3. Average Generations to Solution
    ax3 = fig.add_subplot(gs[1, 1])
    avg_generations = []
    std_generations = []
    for param_val in param_values:
        runs = results_dict[param_val]
        successful_runs = [r for r in runs if r['success']]
        if successful_runs:
            gens = [r['generations'] for r in successful_runs]
            avg_generations.append(np.mean(gens))
            std_generations.append(np.std(gens))
        else:
            avg_generations.append(0)
            std_generations.append(0)

    bars = ax3.bar(range(len(param_values)), avg_generations,
                   yerr=std_generations, capsize=5, color='coral', alpha=0.7)
    ax3.set_xlabel(param_name, fontsize=12)
    ax3.set_ylabel('Avg Generations (successful runs)', fontsize=12)
    ax3.set_title('Convergence Speed', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(param_values)))
    ax3.set_xticklabels(param_values)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Average Evaluations Box Plot
    ax4 = fig.add_subplot(gs[2, 0])
    eval_data = []
    for param_val in param_values:
        runs = results_dict[param_val]
        evals = [r['evaluations'] for r in runs]
        eval_data.append(evals)

    bp = ax4.boxplot(eval_data, labels=param_values, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax4.set_xlabel(param_name, fontsize=12)
    ax4.set_ylabel('Evaluations', fontsize=12)
    ax4.set_title('Evaluations Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Final Fitness Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    fitness_data = []
    for param_val in param_values:
        runs = results_dict[param_val]
        fitnesses = [r['best_fitness'] for r in runs]
        fitness_data.append(fitnesses)

    bp = ax5.boxplot(fitness_data, labels=param_values, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('plum')
        patch.set_alpha(0.7)
    ax5.set_xlabel(param_name, fontsize=12)
    ax5.set_ylabel('Final Best Fitness', fontsize=12)
    ax5.set_title('Final Fitness Distribution', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=1)

    plt.suptitle(f'Parameter Sensitivity Analysis: {param_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    filename = f'parameter_sensitivity_{param_name.replace(" ", "_").lower()}.png'
    _save_and_close(filename)


# ========================= TASK 3: CROSSOVER COMPARISON =========================

def plot_crossover_comparison(results_dict, crossover_names):
    """Compare different crossover strategies."""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Convergence Curves
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for idx, crossover_name in enumerate(crossover_names):
        runs = results_dict[crossover_name]
        all_histories = [r['fitness_history'] for r in runs]
        max_len = max(len(h) for h in all_histories)

        padded_histories = []
        for hist in all_histories:
            padded = list(hist) + [hist[-1]] * (max_len - len(hist))
            padded_histories.append(padded)

        avg_history = np.mean(padded_histories, axis=0)
        std_history = np.std(padded_histories, axis=0)

        generations = range(len(avg_history))
        ax1.plot(generations, avg_history, label=crossover_name,
                 linewidth=2.5, color=colors[idx % len(colors)])
        ax1.fill_between(generations,
                         avg_history - std_history,
                         avg_history + std_history,
                         alpha=0.15, color=colors[idx % len(colors)])

    ax1.set_xlabel('Generation', fontsize=13)
    ax1.set_ylabel('Best Fitness', fontsize=13)
    ax1.set_title('Crossover Strategy Convergence Comparison',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Optimal')

    # 2. Success Rate Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    success_rates = []
    for crossover_name in crossover_names:
        runs = results_dict[crossover_name]
        success_rate = sum(1 for r in runs if r['success']) / len(runs) * 100
        success_rates.append(success_rate)

    bars = ax2.bar(crossover_names, success_rates,
                   color=colors[:len(crossover_names)], alpha=0.7)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate by Crossover', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    # 3. Convergence Speed (Generations)
    ax3 = fig.add_subplot(gs[1, 1])
    avg_gens = []
    std_gens = []
    for crossover_name in crossover_names:
        runs = results_dict[crossover_name]
        successful = [r for r in runs if r['success']]
        if successful:
            gens = [r['generations'] for r in successful]
            avg_gens.append(np.mean(gens))
            std_gens.append(np.std(gens))
        else:
            avg_gens.append(0)
            std_gens.append(0)

    bars = ax3.bar(crossover_names, avg_gens, yerr=std_gens, capsize=5,
                   color=colors[:len(crossover_names)], alpha=0.7)
    ax3.set_ylabel('Avg Generations', fontsize=12)
    ax3.set_title('Convergence Speed', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Evaluations Comparison
    ax4 = fig.add_subplot(gs[1, 2])
    eval_data = [
        [r['evaluations'] for r in results_dict[name]]
        for name in crossover_names
    ]

    bp = ax4.boxplot(eval_data, labels=crossover_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(crossover_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_ylabel('Evaluations', fontsize=12)
    ax4.set_title('Evaluations Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Crossover Strategy Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    _save_and_close('crossover_comparison.png')


# ========================= TASK 4: SURVIVAL STRATEGY =========================

def plot_survival_strategies(results_dict, strategy_names):
    """Compare survival strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Survival Strategy Comparison',
                 fontsize=16, fontweight='bold')

    # 1. Convergence Curves
    ax1 = axes[0, 0]
    for strategy in strategy_names:
        runs = results_dict[strategy]
        all_histories = [r['fitness_history'] for r in runs]
        max_len = max(len(h) for h in all_histories)

        padded_histories = []
        for hist in all_histories:
            padded = list(hist) + [hist[-1]] * (max_len - len(hist))
            padded_histories.append(padded)

        avg_history = np.mean(padded_histories, axis=0)
        std_history = np.std(padded_histories, axis=0)

        generations = range(len(avg_history))
        ax1.plot(generations, avg_history, label=strategy, linewidth=2.5)
        ax1.fill_between(generations,
                         avg_history - std_history,
                         avg_history + std_history,
                         alpha=0.2)

    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Convergence Comparison', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)

    # 2. Success Rate
    ax2 = axes[0, 1]
    success_rates = []
    for strategy in strategy_names:
        runs = results_dict[strategy]
        success_rate = sum(1 for r in runs if r['success']) / len(runs) * 100
        success_rates.append(success_rate)

    colors = plt.cm.Set2(range(len(strategy_names)))
    bars = ax2.bar(strategy_names, success_rates, color=colors, alpha=0.7)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

    # 3. Generations to Solution
    ax3 = axes[1, 0]
    gen_data = []
    for strategy in strategy_names:
        runs = results_dict[strategy]
        successful = [r for r in runs if r['success']]
        gens = [r['generations'] for r in successful] if successful else [0]
        gen_data.append(gens)

    bp = ax3.boxplot(gen_data, labels=strategy_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Generations', fontsize=12)
    ax3.set_title('Generations to Solution (Successful Runs)',
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Solution Quality Consistency
    ax4 = axes[1, 1]
    fitness_data = []
    for strategy in strategy_names:
        runs = results_dict[strategy]
        fitnesses = [r['best_fitness'] for r in runs]
        fitness_data.append(fitnesses)

    bp = ax4.boxplot(fitness_data, labels=strategy_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel('Final Fitness', fontsize=12)
    ax4.set_title('Solution Quality Distribution',
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)

    plt.tight_layout()
    _save_and_close('survival_strategy_comparison.png')


# ========================= TASK 5: SCALABILITY STUDY =========================

def plot_scalability_study(results_dict, n_values):
    """Analyze scalability across different N values."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Convergence curves for all N values
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))

    for idx, n in enumerate(n_values):
        runs = results_dict[n]
        all_histories = [r['fitness_history'] for r in runs]
        max_len = max(len(h) for h in all_histories)

        padded_histories = []
        for hist in all_histories:
            padded = list(hist) + [hist[-1]] * (max_len - len(hist))
            padded_histories.append(padded)

        avg_history = np.mean(padded_histories, axis=0)

        generations = range(len(avg_history))
        ax1.plot(generations, avg_history, label=f'N={n}',
                 linewidth=2.5, color=colors[idx])

    ax1.set_xlabel('Generation', fontsize=13)
    ax1.set_ylabel('Best Fitness', fontsize=13)
    ax1.set_title('Convergence Curves: Scalability Study',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Success Rate vs N
    ax2 = fig.add_subplot(gs[1, 0])
    success_rates = []
    for n in n_values:
        runs = results_dict[n]
        success_rate = sum(1 for r in runs if r['success']) / len(runs) * 100
        success_rates.append(success_rate)

    ax2.plot(n_values, success_rates, marker='o', linewidth=2.5,
             markersize=10, color='steelblue')
    ax2.set_xlabel('N (Board Size)', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate vs Problem Size',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    for n, rate in zip(n_values, success_rates):
        ax2.annotate(f'{rate:.1f}%', (n, rate), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10)

    # 3. Average Generations vs N
    ax3 = fig.add_subplot(gs[1, 1])
    avg_gens = []
    std_gens = []
    for n in n_values:
        runs = results_dict[n]
        successful = [r for r in runs if r['success']]
        if successful:
            gens = [r['generations'] for r in successful]
            avg_gens.append(np.mean(gens))
            std_gens.append(np.std(gens))
        else:
            avg_gens.append(0)
            std_gens.append(0)

    ax3.errorbar(n_values, avg_gens, yerr=std_gens, marker='s',
                 linewidth=2.5, markersize=8, capsize=5, color='coral')
    ax3.set_xlabel('N (Board Size)', fontsize=12)
    ax3.set_ylabel('Avg Generations', fontsize=12)
    ax3.set_title('Convergence Speed vs Problem Size',
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Average Evaluations vs N (Log scale)
    ax4 = fig.add_subplot(gs[2, 0])
    avg_evals = []
    std_evals = []
    for n in n_values:
        runs = results_dict[n]
        evals = [r['evaluations'] for r in runs]
        avg_evals.append(np.mean(evals))
        std_evals.append(np.std(evals))

    ax4.errorbar(n_values, avg_evals, yerr=std_evals, marker='^',
                 linewidth=2.5, markersize=8, capsize=5, color='green')
    ax4.set_xlabel('N (Board Size)', fontsize=12)
    ax4.set_ylabel('Avg Evaluations', fontsize=12)
    ax4.set_title('Evaluations vs Problem Size',
                  fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, which='both')

    # 5. Complexity Analysis (Generations/N^2)
    ax5 = fig.add_subplot(gs[2, 1])
    normalized_gens = []
    for n, avg_gen in zip(n_values, avg_gens):
        if avg_gen > 0:
            normalized_gens.append(avg_gen / (n ** 2))
        else:
            normalized_gens.append(0)

    ax5.plot(n_values, normalized_gens, marker='D', linewidth=2.5,
             markersize=8, color='purple')
    ax5.set_xlabel('N (Board Size)', fontsize=12)
    ax5.set_ylabel('Generations / N^2', fontsize=12)
    ax5.set_title('Normalized Complexity', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Scalability Study: N-Queens Problem',
                 fontsize=16, fontweight='bold', y=0.998)

    _save_and_close('scalability_study.png')


# ========================= COMPREHENSIVE COMPARISON =========================

def create_comparison_table(results_dict, category_name):
    """Create a comparison table for any category of experiments."""
    data = []

    for category_value, runs in results_dict.items():
        successes = sum(1 for r in runs if r['success'])
        success_rate = (successes / len(runs)) * 100

        successful_runs = [r for r in runs if r['success']]

        if successful_runs:
            generations_list = [r['generations'] for r in successful_runs]
            evaluations_list = [r['evaluations'] for r in successful_runs]

            avg_generations = np.mean(generations_list)
            std_generations = np.std(generations_list)
            avg_evaluations = np.mean(evaluations_list)
            std_evaluations = np.std(evaluations_list)

            # Debug: print min/max to understand the spread
            min_gen = min(generations_list)
            max_gen = max(generations_list)
            print(f"  [{category_value}] Generations range: {min_gen} to {max_gen} (mean={avg_generations:.1f}, std={std_generations:.1f})")
        else:
            avg_generations = np.nan
            std_generations = np.nan
            avg_evaluations = np.nan
            std_evaluations = np.nan

        all_fitness = [r['best_fitness'] for r in runs]
        avg_fitness = np.mean(all_fitness)
        best_fitness = min(all_fitness)  # Lower is better in N-Queens

        data.append({
            category_name: category_value,
            'Total Runs': len(runs),
            'Successes': successes,
            'Success Rate (%)': f'{success_rate:.1f}',
            'Avg Generations': f'{avg_generations:.1f} [+-] {std_generations:.1f}' if not np.isnan(avg_generations) else 'N/A',
            'Avg Evaluations': f'{avg_evaluations:.0f} [+-] {std_evaluations:.0f}' if not np.isnan(avg_evaluations) else 'N/A',
            'Avg Final Fitness': f'{avg_fitness:.2f}',
            'Best Fitness': f'{best_fitness:.0f}'
        })

    df = pd.DataFrame(data)

    # Print with better formatting
    print(f"\n{'='*120}")
    print(f"{category_name} Comparison Table".center(120))
    print(f"{'='*120}")

    # Custom column widths for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 25)

    print(df.to_string(index=False))
    print(f"{'='*120}\n")

    # Print a summary section with better formatting
    print(f"\n{' SUMMARY '.center(120, '-')}")
    for idx, row in df.iterrows():
        print(f"\n{category_name}: {row[category_name]}")
        print(f"  {'Total Runs:':<25} {row['Total Runs']}")
        print(
            f"  {'Successes:':<25} {row['Successes']} ({row['Success Rate (%)']}%)")
        print(f"  {'Avg Generations:':<25} {row['Avg Generations']}")
        print(f"  {'Avg Evaluations:':<25} {row['Avg Evaluations']}")
        print(f"  {'Avg Final Fitness:':<25} {row['Avg Final Fitness']}")
        print(f"  {'Best Fitness:':<25} {row['Best Fitness']}")
    print(f"{'-'*120}\n")

    # Save to CSV
    csv_filename = f'{category_name.replace(" ", "_").lower()}_comparison.csv'
    df.to_csv(csv_filename, index=False)
    print(f"[OK] Saved table: {csv_filename}")

    return df


def plot_single_experiment(results, experiment_name="Experiment"):
    """Plot results from a single run_experiment call."""
    if len(results) < 2:
        print("Need at least 2 runs to generate meaningful plots")
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'{experiment_name} - {len(results)} Runs',
                 fontsize=16, fontweight='bold')

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Convergence Curves (All Runs)
    ax1 = fig.add_subplot(gs[0, :2])
    for result in results:
        fitness_history = result['fitness_history']
        generations = range(len(fitness_history))
        alpha = 0.3 if len(results) > 10 else 0.6
        color = 'green' if result['success'] else 'red'
        ax1.plot(generations, fitness_history,
                 alpha=alpha, color=color, linewidth=1)

    # Plot average convergence
    all_histories = [r['fitness_history'] for r in results]
    max_len = max(len(h) for h in all_histories)
    padded_histories = []
    for hist in all_histories:
        padded = list(hist) + [hist[-1]] * (max_len - len(hist))
        padded_histories.append(padded)

    avg_history = np.mean(padded_histories, axis=0)
    ax1.plot(range(len(avg_history)), avg_history,
             'b-', linewidth=3, label='Average')

    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Convergence Curves (All Runs)',
                  fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--',
                linewidth=1.5, label='Optimal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Success Rate Pie Chart
    ax2 = fig.add_subplot(gs[0, 2])
    successes = sum(1 for r in results if r['success'])
    failures = len(results) - successes
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0) if successes > 0 else (0, 0.1)

    ax2.pie([successes, failures], labels=['Success', 'Failure'], autopct='%1.1f%%',
            colors=colors, explode=explode, startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'Success Rate\n({successes}/{len(results)} runs)',
                  fontsize=13, fontweight='bold')

    # 3. Generations Distribution (Successful Runs)
    ax3 = fig.add_subplot(gs[1, 0])
    successful_runs = [r for r in results if r['success']]

    if successful_runs:
        generations_list = [r['generations'] for r in successful_runs]
        ax3.hist(generations_list, bins=min(15, len(successful_runs)),
                 color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(generations_list), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(generations_list):.1f}')
        ax3.set_xlabel('Generations', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Generations to Solution\n(Successful Runs)',
                      fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No Successful Runs', ha='center', va='center',
                 fontsize=14, transform=ax3.transAxes)
        ax3.set_title('Generations to Solution',
                      fontsize=13, fontweight='bold')

    # 4. Evaluations Box Plot
    ax4 = fig.add_subplot(gs[1, 1])
    evaluations_list = [r['evaluations'] for r in results]
    bp = ax4.boxplot([evaluations_list], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('coral')
    bp['boxes'][0].set_alpha(0.7)

    ax4.set_ylabel('Evaluations', fontsize=12)
    ax4.set_title(f'Evaluations Distribution\nMean: {np.mean(evaluations_list):.0f}',
                  fontsize=13, fontweight='bold')
    ax4.set_xticklabels(['All Runs'])
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Final Fitness Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    final_fitness = [r['best_fitness'] for r in results]

    if len(set(final_fitness)) > 1:
        ax5.hist(final_fitness, bins=min(15, len(set(final_fitness))),
                 color='plum', alpha=0.7, edgecolor='black')
        ax5.axvline(np.mean(final_fitness), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(final_fitness):.2f}')
    else:
        ax5.bar(['All Runs'], [final_fitness[0]], color='plum', alpha=0.7)

    ax5.axvline(0, color='green', linestyle='--', linewidth=2, label='Optimal')
    ax5.set_xlabel('Final Best Fitness', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Final Fitness Distribution', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    filename = f'{experiment_name.replace(" ", "_")}_results.png'
    _save_and_close(filename)
