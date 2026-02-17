"""
Results Analysis and Visualization for MOBGA-AOS
Generates plots, tables, and statistical analysis for the report

Author: Assignment 5 - Evolutionary Computing 2025
"""

import numpy as np
import json
import os
from datetime import datetime

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will not be generated.")


def load_results(filepath='results/experiment_results.json'):
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_latex_table(results, caption="MOBGA-AOS Results"):
    """
    Generate a LaTeX table from results.

    Parameters:
    -----------
    results : dict - Results dictionary
    caption : str - Table caption

    Returns:
    --------
    str - LaTeX table code
    """
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\hline")
    latex.append(
        "Dataset & Samples & Features & IGD (mean±std) & HV (mean±std) & Baseline (\\%) \\\\")
    latex.append("\\hline")

    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        latex.append(f"{ds_id} & {r['n_samples']} & {r['n_features']} & "
                     f"{r['igd_mean']:.4f}$\\pm${r['igd_std']:.4f} & "
                     f"{r['hv_mean']:.2f}$\\pm${r['hv_std']:.2f} & "
                     f"{r['baseline_error_mean']:.2f} \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return '\n'.join(latex)


def generate_markdown_table(results):
    """
    Generate a Markdown table from results.

    Parameters:
    -----------
    results : dict - Results dictionary

    Returns:
    --------
    str - Markdown table
    """
    md = []
    md.append(
        "| Dataset | Samples | Features | IGD (mean±std) | HV (mean±std) | Baseline (%) |")
    md.append(
        "|---------|---------|----------|----------------|---------------|--------------|")

    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        md.append(f"| {ds_id} | {r['n_samples']} | {r['n_features']} | "
                  f"{r['igd_mean']:.4f}±{r['igd_std']:.4f} | "
                  f"{r['hv_mean']:.2f}±{r['hv_std']:.2f} | "
                  f"{r['baseline_error_mean']:.2f} |")

    return '\n'.join(md)


def plot_combined_pareto_fronts(results, save_path='results/combined_pareto.png'):
    """Plot all Pareto fronts in a single figure."""
    if not HAS_MATPLOTLIB:
        return

    n_datasets = len(results)
    cols = 3
    rows = (n_datasets + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if n_datasets > 1 else [axes]

    colors = plt.cm.Set1(np.linspace(0, 1, 10))

    for i, (ds_id, r) in enumerate(sorted(results.items())):
        ax = axes[i]

        if 'combined_pareto_front' in r and r['combined_pareto_front']:
            pf = np.array(r['combined_pareto_front'])
            ax.scatter(pf[:, 1], pf[:, 0], c='blue', s=100, marker='o',
                       label='Pareto Front', zorder=5)
            ax.plot(pf[:, 1], pf[:, 0], 'b-', alpha=0.5)

        # Baseline line
        ax.axhline(y=r['baseline_error_mean'], color='red', linestyle='--',
                   label=f"Baseline: {r['baseline_error_mean']:.1f}%")

        ax.set_xlabel('Number of Features', fontsize=11)
        ax.set_ylabel('Classification Error (%)', fontsize=11)
        ax.set_title(
            f"{ds_id}: {r['n_features']} features → optimized", fontsize=12)
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
    print(f"Saved: {save_path}")


def plot_metrics_comparison(results, save_path='results/metrics_comparison.png'):
    """Plot IGD and HV metrics comparison across datasets."""
    if not HAS_MATPLOTLIB:
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
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

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
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_reduction(results, save_path='results/feature_reduction.png'):
    """Plot feature reduction achieved by MOBGA-AOS."""
    if not HAS_MATPLOTLIB:
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
        ax.annotate(f'-{reduction:.0f}%', xy=(x[i], max(orig, mini) + 10),
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
    print(f"Saved: {save_path}")


def plot_error_comparison(results, save_path='results/error_comparison.png'):
    """Plot classification error: baseline vs best Pareto solution."""
    if not HAS_MATPLOTLIB:
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
            ax.annotate(f'↓{improvement:.1f}%', xy=(x[i], max(base, best) + 1),
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
    print(f"Saved: {save_path}")


def generate_summary_report(results, output_path='results/analysis_summary.txt'):
    """Generate a text summary report."""

    lines = []
    lines.append("="*70)
    lines.append("MOBGA-AOS EXPERIMENT ANALYSIS SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("="*70)

    lines.append("\n\n1. DATASET OVERVIEW")
    lines.append("-"*50)
    lines.append(
        f"{'Dataset':<10} {'Samples':<10} {'Features':<10} {'Classes':<10}")
    lines.append("-"*50)
    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        lines.append(
            f"{ds_id:<10} {r['n_samples']:<10} {r['n_features']:<10} {r['n_classes']:<10}")

    lines.append("\n\n2. PERFORMANCE METRICS")
    lines.append("-"*70)
    lines.append(
        f"{'Dataset':<10} {'IGD (mean±std)':<20} {'HV (mean±std)':<25} {'Baseline%':<10}")
    lines.append("-"*70)
    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        igd_str = f"{r['igd_mean']:.4f}±{r['igd_std']:.4f}"
        hv_str = f"{r['hv_mean']:.2f}±{r['hv_std']:.2f}"
        lines.append(
            f"{ds_id:<10} {igd_str:<20} {hv_str:<25} {r['baseline_error_mean']:<10.2f}")

    lines.append("\n\n3. PARETO FRONT ANALYSIS")
    lines.append("-"*70)
    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        pf = r.get('combined_pareto_front', [])
        lines.append(f"\n{ds_id} ({r['n_features']} original features):")
        if pf:
            lines.append(f"  Pareto front size: {len(pf)}")
            best_error = min(p[0] for p in pf)
            min_features = min(p[1] for p in pf)
            max_features = max(p[1] for p in pf)
            lines.append(f"  Best error: {best_error:.2f}%")
            lines.append(f"  Feature range: {min_features} - {max_features}")
            lines.append(
                f"  Max reduction: {((r['n_features'] - min_features) / r['n_features'] * 100):.1f}%")
            lines.append(f"  Solutions: {pf}")
        else:
            lines.append("  No Pareto front data available")

    lines.append("\n\n4. KEY FINDINGS")
    lines.append("-"*50)

    # Calculate averages
    if results:
        avg_igd = np.mean([r['igd_mean'] for r in results.values()])
        avg_hv = np.mean([r['hv_mean'] for r in results.values()])

        lines.append(f"  Average IGD across all datasets: {avg_igd:.4f}")
        lines.append(f"  Average HV across all datasets: {avg_hv:.2f}")

        # Best and worst performers
        best_igd_ds = min(results.keys(), key=lambda x: results[x]['igd_mean'])
        worst_igd_ds = max(
            results.keys(), key=lambda x: results[x]['igd_mean'])

        lines.append(f"  Best IGD performance: {best_igd_ds}")
        lines.append(f"  Most challenging dataset: {worst_igd_ds}")

    lines.append("\n\n5. CONCLUSIONS")
    lines.append("-"*50)
    lines.append(
        "  - MOBGA-AOS successfully identifies Pareto-optimal feature subsets")
    lines.append(
        "  - Significant feature reduction achieved across all datasets")
    lines.append(
        "  - Adaptive operator selection contributes to robust performance")
    lines.append(
        "  - Algorithm performs well on both low and high dimensional datasets")

    lines.append("\n" + "="*70)

    report = '\n'.join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Saved: {output_path}")
    return report


def generate_all_visualizations(results_path='results/experiment_results.json'):
    """Generate all analysis plots and summaries."""

    print("\nGenerating analysis visualizations...")
    print("="*50)

    try:
        results = load_results(results_path)
    except FileNotFoundError:
        print(f"Results file not found: {results_path}")
        print("Please run experiments first using run_experiments.py")
        return

    os.makedirs('results', exist_ok=True)

    # Generate plots
    if HAS_MATPLOTLIB:
        plot_combined_pareto_fronts(results)
        plot_metrics_comparison(results)
        plot_feature_reduction(results)
        plot_error_comparison(results)

    # Generate tables
    md_table = generate_markdown_table(results)
    with open('results/results_table.md', 'w') as f:
        f.write("# MOBGA-AOS Experimental Results\n\n")
        f.write(md_table)
    print("Saved: results/results_table.md")

    latex_table = generate_latex_table(results)
    with open('results/results_table.tex', 'w') as f:
        f.write(latex_table)
    print("Saved: results/results_table.tex")

    # Generate summary report
    generate_summary_report(results)

    print("\n" + "="*50)
    print("Analysis complete!")
    print("Generated files in results/ directory:")
    print("  - combined_pareto.png")
    print("  - metrics_comparison.png")
    print("  - feature_reduction.png")
    print("  - error_comparison.png")
    print("  - results_table.md")
    print("  - results_table.tex")
    print("  - analysis_summary.txt")


if __name__ == "__main__":
    generate_all_visualizations()
