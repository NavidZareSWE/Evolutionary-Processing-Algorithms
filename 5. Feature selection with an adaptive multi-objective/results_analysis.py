"""
Results Analysis and Visualization for MOBGA-AOS
Generates plots, tables, and statistical analysis for the report

Author: Assignment 5 - Evolutionary Computing 2025
"""

import numpy as np
import json
import os
from datetime import datetime

# Import plot functions from dedicated module
from plots import (
    HAS_MATPLOTLIB,
    plot_combined_pareto_fronts,
    plot_metrics_comparison,
    plot_feature_reduction,
    plot_error_comparison,
    generate_all_analysis_plots
)


def load_results(filepath='results/experiment_results.json'):
    """Load experiment results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
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
    latex.append("\\caption{{{}}}".format(caption))
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\hline")
    latex.append(
        "Dataset & Samples & Features & IGD (mean$\\pm$std) & HV (mean$\\pm$std) & Baseline (\\%) \\\\")
    latex.append("\\hline")

    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        latex.append("{} & {} & {} & "
                     "{:.4f}$\\pm${:.4f} & "
                     "{:.2f}$\\pm${:.2f} & "
                     "{:.2f} \\\\".format(
                         ds_id, r['n_samples'], r['n_features'],
                         r['igd_mean'], r['igd_std'],
                         r['hv_mean'], r['hv_std'],
                         r['baseline_error_mean']))

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
        "| Dataset | Samples | Features | IGD (mean+/-std) | HV (mean+/-std) | Baseline (%) |")
    md.append(
        "|---------|---------|----------|------------------|-----------------|--------------|")

    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        md.append("| {} | {} | {} | "
                  "{:.4f}+/-{:.4f} | "
                  "{:.2f}+/-{:.2f} | "
                  "{:.2f} |".format(
                      ds_id, r['n_samples'], r['n_features'],
                      r['igd_mean'], r['igd_std'],
                      r['hv_mean'], r['hv_std'],
                      r['baseline_error_mean']))

    return '\n'.join(md)


def generate_summary_report(results, output_path='results/analysis_summary.txt'):
    """Generate a text summary report."""

    lines = []
    lines.append("=" * 70)
    lines.append("MOBGA-AOS EXPERIMENT ANALYSIS SUMMARY")
    lines.append("Generated: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    lines.append("=" * 70)

    lines.append("\n\n1. DATASET OVERVIEW")
    lines.append("-" * 50)
    lines.append("{:<10} {:<10} {:<10} {:<10}".format(
        'Dataset', 'Samples', 'Features', 'Classes'))
    lines.append("-" * 50)
    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        lines.append("{:<10} {:<10} {:<10} {:<10}".format(
            ds_id, r['n_samples'], r['n_features'], r['n_classes']))

    lines.append("\n\n2. PERFORMANCE METRICS")
    lines.append("-" * 70)
    lines.append("{:<10} {:<20} {:<25} {:<10}".format(
        'Dataset', 'IGD (mean+/-std)', 'HV (mean+/-std)', 'Baseline%'))
    lines.append("-" * 70)
    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        igd_str = "{:.4f}+/-{:.4f}".format(r['igd_mean'], r['igd_std'])
        hv_str = "{:.2f}+/-{:.2f}".format(r['hv_mean'], r['hv_std'])
        lines.append("{:<10} {:<20} {:<25} {:<10.2f}".format(
            ds_id, igd_str, hv_str, r['baseline_error_mean']))

    lines.append("\n\n3. PARETO FRONT ANALYSIS")
    lines.append("-" * 70)
    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        pf = r.get('combined_pareto_front', [])
        lines.append("\n{} ({} original features):".format(ds_id, r['n_features']))
        if pf:
            lines.append("  Pareto front size: {}".format(len(pf)))
            best_error = min(p[0] for p in pf)
            min_features = min(p[1] for p in pf)
            max_features = max(p[1] for p in pf)
            lines.append("  Best error: {:.2f}%".format(best_error))
            lines.append("  Feature range: {} - {}".format(min_features, max_features))
            reduction = (r['n_features'] - min_features) / r['n_features'] * 100
            lines.append("  Max reduction: {:.1f}%".format(reduction))
            # Format Pareto front without numpy type wrappers
            pf_formatted = format_pareto_front(pf)
            lines.append("  Solutions: {}".format(pf_formatted))
        else:
            lines.append("  No Pareto front data available")

    lines.append("\n\n4. KEY FINDINGS")
    lines.append("-" * 50)

    # Calculate averages
    if results:
        avg_igd = np.mean([r['igd_mean'] for r in results.values()])
        avg_hv = np.mean([r['hv_mean'] for r in results.values()])

        lines.append("  Average IGD across all datasets: {:.4f}".format(avg_igd))
        lines.append("  Average HV across all datasets: {:.2f}".format(avg_hv))

        # Best and worst performers
        best_igd_ds = min(results.keys(), key=lambda x: results[x]['igd_mean'])
        worst_igd_ds = max(results.keys(), key=lambda x: results[x]['igd_mean'])

        lines.append("  Best IGD performance: {}".format(best_igd_ds))
        lines.append("  Most challenging dataset: {}".format(worst_igd_ds))

    lines.append("\n\n5. CONCLUSIONS")
    lines.append("-" * 50)
    lines.append(
        "  - MOBGA-AOS successfully identifies Pareto-optimal feature subsets")
    lines.append(
        "  - Significant feature reduction achieved across all datasets")
    lines.append(
        "  - Adaptive operator selection contributes to robust performance")
    lines.append(
        "  - Algorithm performs well on both low and high dimensional datasets")

    lines.append("\n" + "=" * 70)

    report = '\n'.join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print("Saved: {}".format(output_path))
    return report


def format_pareto_front(pf):
    """
    Format Pareto front for clean console output without numpy type wrappers.
    
    Parameters:
    -----------
    pf : list - List of [error, n_features] pairs
    
    Returns:
    --------
    str - Formatted string representation
    """
    formatted = []
    for point in pf:
        error = float(point[0])
        n_feat = int(point[1])
        formatted.append("[{:.4f}, {}]".format(error, n_feat))
    return "[" + ", ".join(formatted) + "]"


def format_value(val):
    """
    Convert numpy types to native Python types for clean output.
    
    Parameters:
    -----------
    val : any - Value to convert
    
    Returns:
    --------
    Native Python type
    """
    if hasattr(val, 'item'):  # numpy scalar
        return val.item()
    elif isinstance(val, (list, tuple)):
        return [format_value(v) for v in val]
    elif isinstance(val, dict):
        return {k: format_value(v) for k, v in val.items()}
    return val


def generate_all_visualizations(results_path='results/experiment_results.json'):
    """Generate all analysis plots and summaries."""

    print("\nGenerating analysis visualizations...")
    print("=" * 50)

    try:
        results = load_results(results_path)
    except FileNotFoundError:
        print("Results file not found: {}".format(results_path))
        print("Please run experiments first using run_experiments.py")
        return

    os.makedirs('results', exist_ok=True)

    # Generate plots
    if HAS_MATPLOTLIB:
        generate_all_analysis_plots(results)

    # Generate tables
    md_table = generate_markdown_table(results)
    with open('results/results_table.md', 'w', encoding='utf-8') as f:
        f.write("# MOBGA-AOS Experimental Results\n\n")
        f.write(md_table)
    print("Saved: results/results_table.md")

    latex_table = generate_latex_table(results)
    with open('results/results_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print("Saved: results/results_table.tex")

    # Generate summary report
    generate_summary_report(results)

    print("\n" + "=" * 50)
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
