# main.py
from queens_ga import QueensGA
from plotting_functions import plot_single_experiment
from all_experminets_Runner import (
    run_experiment,
    print_summary_table,
    run_task_2_only,
    run_task_3_only,
    run_task_4_only,
    run_task_5_only,
    run_all_experiments
)

if __name__ == "__main__":
    print("="*90)
    print("                              N-Queens Genetic Algorithm")
    print("="*90)

    # results = run_experiment(
    #     num_runs=30,
    #     n=8,
    #     pop_size=100,
    #     mutation_prob=0.5,
    #     recombination_rate=1.0,
    #     max_evaluations=10000
    # )

    # Print summary and plot
    # print_summary_table(results)
    # plot_single_experiment(results, "Baseline 8-Queens")

    # run_all_experiments()

    # Run only specific tasks:
    # run_task_2_only()  # Just parameter sensitivity
    run_task_3_only()  # Just crossover comparison
    # run_task_4_only()  # Just survival strategies
    # run_task_5_only()  # Just scalability
