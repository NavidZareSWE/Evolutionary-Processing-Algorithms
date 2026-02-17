"""
Experiment Runner for MOBGA-AOS
Runs experiments on all 6 datasets and generates results

Usage: python run_experiments.py
"""

import numpy as np
import os
import json
import time
from datetime import datetime

# Import MOBGA-AOS implementation
from mobga_aos import (
    MOBGA_AOS, load_dataset, train_test_split,
    compute_all_features_error, merge_pareto_fronts,
    compute_igd, compute_hypervolume_2d,
    plot_pareto_front, plot_hv_convergence, plot_osp_evolution
)


def run_single_dataset(ds_id, ds_file, ds_name, max_fes, n_runs=3):
    """
    Run MOBGA-AOS experiments on a single dataset.

    Parameters:
    -----------
    ds_id : str - Dataset identifier (e.g., 'DS02')
    ds_file : str - Path to dataset CSV file
    ds_name : str - Human-readable dataset name
    max_fes : int - Maximum fitness evaluations
    n_runs : int - Number of independent runs

    Returns:
    --------
    dict - Results dictionary with metrics and Pareto fronts
    """
    full_name = f"{ds_id}_{ds_name}"
    print(f"\n{'='*60}")
    print(f"Dataset: {full_name}")
    print(f"Max FEs: {max_fes}")
    print(f"{'='*60}")

    try:
        # Load dataset
        X, y = load_dataset(ds_file)
        n_features = X.shape[1]
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        print(
            f"Loaded: {n_samples} samples, {n_features} features, {n_classes} classes")

        # Storage for multiple runs
        all_fronts = []
        all_hv_histories = []
        all_osp_histories = []
        baselines = []
        run_times = []
        all_populations = []

        for run in range(n_runs):
            seed = 42 + run * 100

            print(f"\n--- Run {run + 1}/{n_runs} (seed={seed}) ---")

            # Train-test split
            X_train, y_train, X_test, y_test = train_test_split(
                X, y, seed=seed)

            # Baseline
            baseline_error = compute_all_features_error(X_train, y_train)
            baselines.append(baseline_error)
            print(
                f"Baseline error (all {n_features} features): {baseline_error:.2f}%")

            # Run MOBGA-AOS
            mobga = MOBGA_AOS(
                n_features=n_features,
                max_fes=max_fes,
                pop_size=100,
                crossover_rate=0.9,
                lp=5,
                verbose=False  # Reduce output
            )
            mobga.load_data(X_train, y_train)

            start_time = time.time()
            pareto_front, final_pop = mobga.run(seed=seed)
            run_time = time.time() - start_time

            # Store results
            all_fronts.append(pareto_front)
            all_hv_histories.append(mobga.hv_history)
            all_osp_histories.append(mobga.osp_history)
            run_times.append(run_time)
            all_populations.append(final_pop)

            # Print run summary
            if pareto_front:
                best_error = min(p[0] for p in pareto_front)
                min_features = min(p[1] for p in pareto_front)
                print(f"PF size: {len(pareto_front)}, Best error: {best_error:.2f}%, "
                      f"Min features: {min_features}, Time: {run_time:.1f}s")

        # Compute combined Pareto front
        true_pf = merge_pareto_fronts(all_fronts)

        # Compute metrics
        reference_point = [100.0, float(n_features + 1)]

        igd_values = [compute_igd(front, true_pf) for front in all_fronts]
        hv_values = [compute_hypervolume_2d(
            front, reference_point) for front in all_fronts]

        mean_baseline = np.mean(baselines)

        # Summary
        print(f"\n{'='*40}")
        print(f"SUMMARY: {full_name}")
        print(f"{'='*40}")
        print(f"IGD: {np.mean(igd_values):.6f} ± {np.std(igd_values):.6f}")
        print(f"HV:  {np.mean(hv_values):.4f} ± {np.std(hv_values):.4f}")
        print(f"Mean Baseline: {mean_baseline:.2f}%")
        print(f"Combined PF size: {len(true_pf)}")
        print(f"Combined PF: {true_pf}")

        # Generate plots
        os.makedirs('results', exist_ok=True)

        try:
            plot_pareto_front(all_fronts, true_pf, full_name, n_features,
                              mean_baseline, f'results/pareto_{ds_id}.png')
            plot_hv_convergence(all_hv_histories, full_name,
                                f'results/hv_{ds_id}.png')
            plot_osp_evolution(all_osp_histories, full_name,
                               f'results/osp_{ds_id}.png')
            print(f"Plots saved to results/")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

        return {
            'dataset_id': ds_id,
            'dataset_name': full_name,
            'n_features': n_features,
            'n_samples': n_samples,
            'n_classes': n_classes,
            'max_fes': max_fes,
            'n_runs': n_runs,
            'igd_mean': float(np.mean(igd_values)),
            'igd_std': float(np.std(igd_values)),
            'igd_values': [float(v) for v in igd_values],
            'hv_mean': float(np.mean(hv_values)),
            'hv_std': float(np.std(hv_values)),
            'hv_values': [float(v) for v in hv_values],
            'baseline_error_mean': float(mean_baseline),
            'baseline_errors': [float(b) for b in baselines],
            'run_times': [float(t) for t in run_times],
            'pareto_fronts': all_fronts,
            'combined_pareto_front': true_pf,
            'final_osp': [list(osp[-1]) if osp else None for osp in all_osp_histories],
        }

    except Exception as e:
        print(f"ERROR: Failed to process {full_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_experiments():
    """Run experiments on all 6 datasets as specified in the assignment."""

    # Dataset configuration
    # Based on the assignment: DS02, DS04, DS05, DS07, DS08, DS10
    DATASETS = {
        'DS02': ('DS02.csv', 'LungCancer', 10000),
        'DS04': ('DS04.csv', 'OpticalRecognition', 15000),
        'DS05': ('DS05.csv', 'MadelonValid', 50000),
        'DS07': ('DS07.csv', 'Har', 50000),
        'DS08': ('DS08.csv', 'HAPT', 50000),
        'DS10': ('DS10.csv', 'MultipleFeaturesDigit', 50000),
    }

    print("="*70)
    print("MOBGA-AOS EXPERIMENT RUNNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"\nDatasets to process: {list(DATASETS.keys())}")
    print("Running 3 independent runs per dataset\n")

    all_results = {}

    for ds_id, (ds_file, ds_name, max_fes) in DATASETS.items():
        result = run_single_dataset(ds_id, ds_file, ds_name, max_fes, n_runs=3)
        if result:
            all_results[ds_id] = result

    # Save complete results
    os.makedirs('results', exist_ok=True)

    # Save JSON results (without numpy arrays)
    json_results = {}
    for ds_id, result in all_results.items():
        json_results[ds_id] = {
            'dataset_id': result['dataset_id'],
            'dataset_name': result['dataset_name'],
            'n_features': result['n_features'],
            'n_samples': result['n_samples'],
            'n_classes': result['n_classes'],
            'max_fes': result['max_fes'],
            'n_runs': result['n_runs'],
            'igd_mean': result['igd_mean'],
            'igd_std': result['igd_std'],
            'igd_values': result['igd_values'],
            'hv_mean': result['hv_mean'],
            'hv_std': result['hv_std'],
            'hv_values': result['hv_values'],
            'baseline_error_mean': result['baseline_error_mean'],
            'baseline_errors': result['baseline_errors'],
            'run_times': result['run_times'],
            'combined_pareto_front': result['combined_pareto_front'],
        }

    with open('results/experiment_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # Generate summary table
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)

    print("\n{:<10} {:<8} {:<8} {:<15} {:<15} {:<10}".format(
        "Dataset", "Samples", "Features", "IGD (mean±std)", "HV (mean±std)", "Baseline%"
    ))
    print("-"*70)

    for ds_id, result in all_results.items():
        print("{:<10} {:<8} {:<8} {:.4f}±{:.4f}    {:.2f}±{:.2f}    {:.2f}%".format(
            result['dataset_id'],
            result['n_samples'],
            result['n_features'],
            result['igd_mean'],
            result['igd_std'],
            result['hv_mean'],
            result['hv_std'],
            result['baseline_error_mean']
        ))

    print("\n" + "="*70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Results saved to: results/experiment_results.json")
    print("Plots saved to: results/*.png")
    print("="*70)

    return all_results


def quick_test():
    """Quick test with reduced parameters for debugging."""
    print("QUICK TEST MODE")
    print("Running DS02 with reduced FEs for testing...\n")

    result = run_single_dataset(
        'DS02', 'DS02.csv', 'LungCancer', max_fes=3000, n_runs=2)

    if result:
        print("\nQuick test completed successfully!")
        return result
    else:
        print("\nQuick test failed!")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Quick test mode
        quick_test()
    else:
        # Full experiment run
        run_all_experiments()
