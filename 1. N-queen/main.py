# main.py
from queens_ga import QueensGA
from all_experminets_Runner import run_all_experiments


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
    run_all_experiments()

    # Print summary and plot
    # print_summary_table(results)
    # plot_single_experiment(results, "Baseline 8-Queens")
