"""
Plotting Utilities for MOBGA-AOS
All visualization functions for Pareto fronts, metrics, and analysis

Author: Assignment 5 - Evolutionary Computing 2025
"""

import numpy as np
import os

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will not be generated.")


def plot_combined_pareto_fronts(results, save_path='results/combined_pareto.png'):
    """
    Plot all Pareto fronts in a single figure with subplots.
    
    Parameters:
    -----------
    results : dict - Results dictionary with dataset results
    save_path : str - Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return

    n_datasets = len(results)
    if n_datasets == 0:
        print("No results to plot")
        return
        
    cols = 3
    rows = (n_datasets + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if n_datasets > 1 else [axes]

    for i, (ds_id, r) in enumerate(sorted(results.items())):
        ax = axes[i]

        if 'combined_pareto_front' in r and r['combined_pareto_front']:
            pf = np.array(r['combined_pareto_front'])
            ax.scatter(pf[:, 1], pf[:, 0], c='blue', s=100, marker='o',
                       label='Pareto Front', zorder=5)
            ax.plot(pf[:, 1], pf[:, 0], 'b-', alpha=0.5)

        # Baseline line
        baseline_val = r['baseline_error_mean']
        ax.axhline(y=baseline_val, color='red', linestyle='--',
                   label="Baseline: {:.1f}%".format(baseline_val))

        ax.set_xlabel('Number of Features', fontsize=11)
        ax.set_ylabel('Classification Error (%)', fontsize=11)
        ax.set_title("{}: {} features -> optimized".format(ds_id, r['n_features']), fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('MOBGA-AOS Pareto Fronts Across All Datasets',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_metrics_comparison(results, save_path='results/metrics_comparison.png'):
    """
    Plot IGD and HV metrics comparison across datasets.
    
    Parameters:
    -----------
    results : dict - Results dictionary
    save_path : str - Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return

    datasets = sorted(results.keys())
    igd_means = [results[ds]['igd_mean'] for ds in datasets]
    igd_stds = [results[ds]['igd_std'] for ds in datasets]
    hv_means = [results[ds]['hv_mean'] for ds in datasets]
    hv_stds = [results[ds]['hv_std'] for ds in datasets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(datasets))
    width = 0.6

    # IGD plot
    bars1 = ax1.bar(x, igd_means, width, yerr=igd_stds,
                    capsize=5, color='steelblue')
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('IGD (lower is better)', fontsize=12)
    ax1.set_title('Inverted Generational Distance', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, igd_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 '{:.3f}'.format(val), ha='center', va='bottom', fontsize=9)

    # HV plot
    bars2 = ax2.bar(x, hv_means, width, yerr=hv_stds,
                    capsize=5, color='forestgreen')
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('Hypervolume (higher is better)', fontsize=12)
    ax2.set_title('Hypervolume', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars2, hv_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 '{:.1f}'.format(val), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_feature_reduction(results, save_path='results/feature_reduction.png'):
    """
    Plot feature reduction achieved by MOBGA-AOS.
    
    Parameters:
    -----------
    results : dict - Results dictionary
    save_path : str - Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = sorted(results.keys())
    original_features = [results[ds]['n_features'] for ds in datasets]

    # Get minimum features in Pareto front
    min_features = []
    for ds in datasets:
        pf = results[ds].get('combined_pareto_front', [[100, 1]])
        if pf:
            min_features.append(min(p[1] for p in pf))
        else:
            min_features.append(results[ds]['n_features'])

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, original_features, width,
                   label='Original Features', color='coral')
    bars2 = ax.bar(x + width/2, min_features, width,
                   label='Minimum Selected', color='teal')

    # Add reduction percentage labels
    for i, (orig, mini) in enumerate(zip(original_features, min_features)):
        reduction = (1 - mini/orig) * 100
        ax.annotate('-{:.0f}%'.format(reduction), xy=(x[i], max(orig, mini) + 10),
                    ha='center', fontsize=10, color='darkgreen')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Reduction by MOBGA-AOS', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_error_comparison(results, save_path='results/error_comparison.png'):
    """
    Plot classification error: baseline vs best Pareto solution.
    
    Parameters:
    -----------
    results : dict - Results dictionary
    save_path : str - Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = sorted(results.keys())
    baselines = [results[ds]['baseline_error_mean'] for ds in datasets]

    # Get best (minimum) error from Pareto front
    best_errors = []
    for ds in datasets:
        pf = results[ds].get('combined_pareto_front', [[100, 1]])
        if pf:
            best_errors.append(min(p[0] for p in pf))
        else:
            best_errors.append(baselines[datasets.index(ds)])

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, baselines, width,
                   label='Baseline (All Features)', color='salmon')
    bars2 = ax.bar(x + width/2, best_errors, width,
                   label='Best MOBGA-AOS', color='lightgreen')

    # Add improvement labels
    for i, (base, best) in enumerate(zip(baselines, best_errors)):
        improvement = base - best
        if improvement > 0:
            ax.annotate('v{:.1f}%'.format(improvement), xy=(x[i], max(base, best) + 1),
                        ha='center', fontsize=10, color='green')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Classification Error (%)', fontsize=12)
    ax.set_title('Classification Error: Baseline vs MOBGA-AOS', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_pareto_front(all_fronts, true_pf, dataset_name, n_features, 
                      baseline_error, save_path):
    """
    Plot Pareto fronts from multiple runs with combined front.
    
    Parameters:
    -----------
    all_fronts : list - List of Pareto fronts from each run
    true_pf : list - Combined true Pareto front
    dataset_name : str - Name of the dataset
    n_features : int - Total number of features
    baseline_error : float - Baseline classification error
    save_path : str - Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_fronts)))
    
    # Plot individual run fronts
    for i, front in enumerate(all_fronts):
        if front:
            front_arr = np.array(front)
            ax.scatter(front_arr[:, 1], front_arr[:, 0], 
                      c=[colors[i]], s=50, alpha=0.6,
                      label='Run {}'.format(i+1), marker='o')
    
    # Plot combined Pareto front
    if true_pf:
        pf_arr = np.array(true_pf)
        ax.scatter(pf_arr[:, 1], pf_arr[:, 0], 
                  c='red', s=150, marker='*',
                  label='Combined PF', zorder=10)
        ax.plot(pf_arr[:, 1], pf_arr[:, 0], 'r--', alpha=0.5, linewidth=2)
    
    # Baseline line
    ax.axhline(y=baseline_error, color='gray', linestyle=':', 
               label='Baseline: {:.1f}%'.format(baseline_error))
    
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Classification Error (%)', fontsize=12)
    ax.set_title('Pareto Front: {} (n={})'.format(dataset_name, n_features), fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_hv_convergence(hv_histories, dataset_name, save_path):
    """
    Plot hypervolume convergence over generations.
    
    Parameters:
    -----------
    hv_histories : list - List of HV history arrays from each run
    dataset_name : str - Name of the dataset
    save_path : str - Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(hv_histories)))
    
    for i, hv_hist in enumerate(hv_histories):
        if hv_hist:
            ax.plot(hv_hist, color=colors[i], linewidth=1.5,
                   label='Run {}'.format(i+1), alpha=0.8)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Hypervolume', fontsize=12)
    ax.set_title('HV Convergence: {}'.format(dataset_name), fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_osp_evolution(osp_histories, dataset_name, save_path):
    """
    Plot Operator Selection Probability evolution over generations.
    
    Parameters:
    -----------
    osp_histories : list - List of OSP history arrays from each run
    dataset_name : str - Name of the dataset
    save_path : str - Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return

    fig, axes = plt.subplots(1, len(osp_histories), figsize=(5*len(osp_histories), 5))
    
    if len(osp_histories) == 1:
        axes = [axes]
    
    operator_names = ['1-Point', '2-Point', 'Uniform', 'AND', 'OR']
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for run_idx, (ax, osp_hist) in enumerate(zip(axes, osp_histories)):
        if osp_hist:
            osp_arr = np.array(osp_hist)
            for op_idx in range(osp_arr.shape[1]):
                ax.plot(osp_arr[:, op_idx], color=colors[op_idx],
                       label=operator_names[op_idx], linewidth=1.5)
        
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('Selection Probability', fontsize=11)
        ax.set_title('Run {} OSP'.format(run_idx+1), fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Operator Selection Probability: {}'.format(dataset_name), fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_operator_usage(osp_final, dataset_name, save_path='results/operator_usage.png'):
    """
    Plot final operator usage as a bar chart.
    
    Parameters:
    -----------
    osp_final : list - Final OSP values for each run
    dataset_name : str - Name of the dataset
    save_path : str - Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return
    
    operator_names = ['1-Point', '2-Point', 'Uniform', 'AND', 'OR']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(operator_names))
    width = 0.25
    
    for run_idx, osp in enumerate(osp_final):
        if osp:
            ax.bar(x + run_idx * width, osp, width, 
                   label='Run {}'.format(run_idx+1), alpha=0.8)
    
    ax.set_xlabel('Crossover Operator', fontsize=12)
    ax.set_ylabel('Final Selection Probability', fontsize=12)
    ax.set_title('Final Operator Usage: {}'.format(dataset_name), fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(operator_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def generate_all_analysis_plots(results, output_dir='results'):
    """
    Generate all analysis plots from experiment results.
    
    Parameters:
    -----------
    results : dict - Complete experiment results
    output_dir : str - Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping all plots")
        return
    
    print("\nGenerating analysis plots...")
    print("-" * 40)
    
    plot_combined_pareto_fronts(results, 
                                os.path.join(output_dir, 'combined_pareto.png'))
    plot_metrics_comparison(results, 
                           os.path.join(output_dir, 'metrics_comparison.png'))
    plot_feature_reduction(results, 
                          os.path.join(output_dir, 'feature_reduction.png'))
    plot_error_comparison(results, 
                         os.path.join(output_dir, 'error_comparison.png'))
    
    print("-" * 40)
    print("All plots generated successfully!")
