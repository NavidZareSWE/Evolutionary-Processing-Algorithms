import os
import json
import time
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from plots import plot_opSelectProb_evolution, plot_hv_convergence, plot_pareto_front
from mobga_aos import MOBGA_AOS
from nsga2 import merge_pareto_fronts
from utils import load_dataset, train_test_split, compute_all_features_error, format_pareto_front, convert_pareto_front
import warnings
warnings.filterwarnings('ignore')  # stop showing warning messages

if __name__ == "__main__":

    DATASETS = {
        'DS02': ('DS02.csv', 'LungCancer', 10000),
        'DS04': ('DS04.csv', 'OpticalRecognition', 15000),
        'DS05': ('DS05.csv', 'MadelonValid', 30000),
        'DS07': ('DS07.csv', 'Har', 30000),
        'DS08': ('DS08.csv', 'HAPT', 30000),
        'DS10': ('DS10.csv', 'MultipleFeaturesDigit', 30000),
    }

    os.makedirs('results', exist_ok=True)

    all_results = {}

    for ds_id, (ds_file, ds_name, max_fitness_evaluations) in DATASETS.items():
        full_name = "{}_{}".format(ds_id, ds_name)
        print("\n" + "=" * 60)
        print("Processing: {}".format(full_name))
        print("=" * 60)

        try:
            X, y = load_dataset(ds_file)
            n_samples = X.shape[0]
            n_features = X.shape[1]
            n_classes = len(np.unique(y))

            print("Samples: {}, Features: {}, Classes: {}".format(
                n_samples, n_features, n_classes))

            all_fronts = []
            all_HV_histories = []
            all_opSelectProb_histories = []
            baselines = []
            run_times = []

            # Run experiments -
            # (Paper:   run 30 times per dataset -
            # This Code: run 3 times per dataset  (cause of time consumption))
            # Read More [Paper]: 4.2. Benchmark algorithms and parameter settings
            for run in range(3):
                seed = 42 + run * 100

                # Train-test split: (70-30)
                X_train, y_train, X_test, y_test = train_test_split(
                    X, y, seed=seed)

                baseline_error = compute_all_features_error(X_train, y_train)
                baselines.append(baseline_error)

                print("\nRun {}/3 (seed={})".format(run + 1, seed))
                print("  Baseline error (all features): {:.2f}%".format(
                    baseline_error))

                mobga = MOBGA_AOS(
                    n_features=n_features,
                    max_fes=max_fitness_evaluations,
                    pop_size=100,
                    crossover_rate=0.9,
                    lp=5
                )
                mobga.load_data(X_train, y_train)

                start_time = time.time()
                pareto_front, final_pop = mobga.run(seed=seed)
                run_time = time.time() - start_time

                all_fronts.append(pareto_front)
                all_HV_histories.append(mobga.hv_history)
                all_opSelectProb_histories.append(mobga.opSelectProb_history)
                run_times.append(run_time)

                print("  Pareto front size: {}".format(len(pareto_front)))
                print("  Run time: {:.2f}s".format(run_time))

            # Compute combined true Pareto front
            true_pf = merge_pareto_fronts(all_fronts)

            # Compute metrics
            reference_point = [100.0, float(n_features + 1)]

            # Using pymoo's built-in HV and IGD indicators instead of custom implementations.
            # HV measures the volume of objective space dominated by the Pareto front (higher = better).
            # IGD measures the average distance from the true front to the obtained front (lower = better).
            # pymoo handles edge cases, is numerically stable, and supports N-dimensional fronts.
            igd_ind = IGD(pf=np.array(true_pf))
            igd_values = [igd_ind(np.array(front)) if front else float(
                'inf') for front in all_fronts]
            hv_ind = HV(ref_point=np.array(reference_point))
            hv_values = [hv_ind(np.array(front))
                         if front else 0.0 for front in all_fronts]

            mean_baseline = float(np.mean(baselines))
            igd_mean = float(np.mean(igd_values))
            igd_std = float(np.std(igd_values))
            hv_mean = float(np.mean(hv_values))
            hv_std = float(np.std(hv_values))

            print("\nResults for {}:".format(full_name))
            print("  IGD: {:.6f} +/- {:.6f}".format(igd_mean, igd_std))
            print("  HV:  {:.4f} +/- {:.4f}".format(hv_mean, hv_std))
            print("  Mean baseline error: {:.2f}%".format(mean_baseline))
            print("  Combined PF: {}".format(format_pareto_front(true_pf)))

            # Generate plots
            plot_pareto_front(all_fronts, true_pf, full_name, n_features,
                              mean_baseline, 'results/pareto_{}.png'.format(ds_id))
            plot_hv_convergence(all_HV_histories, full_name,
                                'results/hv_{}.png'.format(ds_id))
            plot_opSelectProb_evolution(all_opSelectProb_histories, full_name,
                                        'results/opSelectProb_{}.png'.format(ds_id))

            # Store results with native Python types
            all_results[ds_id] = {
                'dataset': full_name,
                'n_features': int(n_features),
                'n_samples': int(n_samples),
                'n_classes': int(n_classes),
                'max_fes': int(max_fitness_evaluations),
                'igd_mean': igd_mean,
                'igd_std': igd_std,
                'hv_mean': hv_mean,
                'hv_std': hv_std,
                'baseline_error': mean_baseline,
                'true_pareto_front': convert_pareto_front(true_pf),
                'run_times': [float(t) for t in run_times],
            }

        except Exception as e:
            print("Error processing {}: {}".format(full_name, e))
            continue

    # Save all results
    with open('results/all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("Results saved to 'results/' directory")
    print("=" * 60)
