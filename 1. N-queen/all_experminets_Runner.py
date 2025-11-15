import os
from queens_ga import QueensGA
from utility_functions import visualize_chessboard, print_summary_table
from plotting_functions import (
    plot_parameter_sensitivity,
    plot_crossover_comparison,
    plot_survival_strategies,
    plot_scalability_study,
    create_comparison_table,
    plot_single_experiment
)

# My Single Runner


def run_experiment(num_runs=3, **ga_params):
    results = []

    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{num_runs}")
        print(f"{'='*60}")

        queens_ga = QueensGA(**ga_params)
        queens_ga.initialize_population()
        best_individual, best_fitness, generations, fitness_history, avg_fitness_history = queens_ga.run_ga()

        results.append({
            'run': run + 1,
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'generations': generations,
            'evaluations': queens_ga.get_evaluations(),
            'fitness_history': fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'success': best_fitness == 0
        })

        print(f"Result: {'SUCCESS' if best_fitness == 0 else 'FAILED'}")
        if best_fitness == 0:
            print(f"Solution: {best_individual.get_chromosome()}")
        visualize_chessboard(best_individual)
        # Print summary and plot
        # print_summary_table(results)

    return results


# ##############################################################################
# ##############################################################################
# ##############################################################################
# ##############################################################################
# ##############################################################################
# ##############################################################################
# ##############################################################################


def ensure_results_directory():
    """Create Results directory if it doesn't exist"""
    if not os.path.exists('Results'):
        os.makedirs('Results')
    return 'Results'


def run_all_experiments():
    results_dir = ensure_results_directory()
    original_dir = os.getcwd()

    # Change to Results directory for saving plots
    os.chdir(results_dir)

    try:
        print("\n" + "="*90)
        print("STARTING COMPREHENSIVE EXPERIMENTS FOR 8-QUEENS GENETIC ALGORITHM")
        print("="*90)

        # =====================================================================
        # TASK 2: PARAMETER SENSITIVITY ANALYSIS
        # Lines: 1-150 (approx)
        # =====================================================================
        print("\n" + "="*90)
        print("TASK 2: PARAMETER SENSITIVITY ANALYSIS")
        print("="*90)

        # -----------------------------------------------------------------
        # TASK 2.1: Mutation Probability Sensitivity
        # Test 3 mutation probabilities: 20%, 50%, 100%
        # -----------------------------------------------------------------
        print("\n>>> TASK 2.1: Mutation Probability Sensitivity...")
        mutation_probs = [0.2, 0.5, 1.0]
        mutation_results = {}

        for mut_prob in mutation_probs:
            print(f"\n--- Testing Mutation Probability: {mut_prob} ---")
            mutation_results[mut_prob] = run_experiment(
                num_runs=30,
                n=8,
                pop_size=100,
                mutation_prob=mut_prob,
                recombination_rate=1.0,
                max_evaluations=10000,
                crossover_type='cut_and_fill',
                mutation_type='swap',
                survival_strategy='fitness_based'
            )

        plot_parameter_sensitivity(
            mutation_results, "Mutation Probability", mutation_probs)
        create_comparison_table(mutation_results, "Mutation Probability")

        # -----------------------------------------------------------------
        # TASK 2.2: Recombination Rate Sensitivity
        # Test 2 recombination rates: 50%, 100%
        # -----------------------------------------------------------------
        print("\n>>> TASK 2.2: Recombination Rate Sensitivity...")
        recombination_rates = [0.5, 1.0]
        recombination_results = {}

        for recomb_rate in recombination_rates:
            print(f"\n--- Testing Recombination Rate: {recomb_rate} ---")
            recombination_results[recomb_rate] = run_experiment(
                num_runs=30,
                n=8,
                pop_size=100,
                mutation_prob=0.5,
                recombination_rate=recomb_rate,
                max_evaluations=10000,
                crossover_type='cut_and_fill',
                mutation_type='swap',
                survival_strategy='fitness_based'
            )

        plot_parameter_sensitivity(
            recombination_results, "Recombination Rate", recombination_rates)
        create_comparison_table(recombination_results, "Recombination Rate")

        # -----------------------------------------------------------------
        # TASK 2.3: Mutation Type Comparison (Swap vs Bitwise)
        # Compare two mutation modes
        # -----------------------------------------------------------------
        print("\n>>> TASK 2.3: Mutation Type Comparison...")
        mutation_types = ['swap', 'bitwise']
        mutation_type_results = {}

        for mut_type in mutation_types:
            print(f"\n--- Testing Mutation Type: {mut_type} ---")
            mutation_type_results[mut_type] = run_experiment(
                num_runs=30,
                n=8,
                pop_size=100,
                mutation_prob=0.5,
                recombination_rate=1.0,
                max_evaluations=10000,
                crossover_type='cut_and_fill',
                mutation_type=mut_type,
                survival_strategy='fitness_based'
            )

        plot_parameter_sensitivity(
            mutation_type_results, "Mutation Type", mutation_types)
        create_comparison_table(mutation_type_results, "Mutation Type")

        print("\n" + "="*90)
        print("TASK 2 COMPLETED")
        print("="*90)

        # =====================================================================
        # TASK 3: CROSSOVER STRATEGY EXPLORATION
        # Lines: 151-200 (approx)
        # =====================================================================
        print("\n" + "="*90)
        print("TASK 3: CROSSOVER STRATEGY EXPLORATION")
        print("="*90)

        # -----------------------------------------------------------------
        # TASK 3: Compare Cut-and-Fill, PMX, 2-cut, and 3-cut crossover
        # -----------------------------------------------------------------
        crossover_types = ['cut_and_fill', 'pmx', '2_cut', '3_cut']
        crossover_results = {}

        for crossover in crossover_types:
            print(f"\n--- Testing Crossover Type: {crossover} ---")
            crossover_results[crossover] = run_experiment(
                num_runs=30,
                n=8,
                pop_size=100,
                mutation_prob=0.5,
                recombination_rate=1.0,
                max_evaluations=10000,
                crossover_type=crossover,
                mutation_type='swap',
                survival_strategy='fitness_based'
            )

        plot_crossover_comparison(crossover_results, crossover_types)
        create_comparison_table(crossover_results, "Crossover Type")

        print("\n" + "="*90)
        print("TASK 3 COMPLETED")
        print("="*90)

        # =====================================================================
        # TASK 4: SURVIVAL STRATEGY COMPARISON
        # Lines: 201-250 (approx)
        # =====================================================================
        print("\n" + "="*90)
        print("TASK 4: SURVIVAL STRATEGY COMPARISON")
        print("="*90)

        # -----------------------------------------------------------------
        # TASK 4: Compare Fitness-based, Generational, and Elitism
        # -----------------------------------------------------------------
        survival_strategies = ['fitness_based', 'generational', 'elitism']
        survival_results = {}

        for strategy in survival_strategies:
            print(f"\n--- Testing Survival Strategy: {strategy} ---")
            survival_results[strategy] = run_experiment(
                num_runs=30,  # Assignment asks for 3, but 30 gives better statistics
                n=8,
                pop_size=100,
                mutation_prob=0.5,
                recombination_rate=1.0,
                max_evaluations=10000,
                crossover_type='cut_and_fill',
                mutation_type='swap',
                survival_strategy=strategy
            )

        plot_survival_strategies(survival_results, survival_strategies)
        create_comparison_table(survival_results, "Survival Strategy")

        print("\n" + "="*90)
        print("TASK 4 COMPLETED")
        print("="*90)

        # =====================================================================
        # TASK 5: SCALABILITY STUDY
        # Lines: 251-300 (approx)
        # =====================================================================
        print("\n" + "="*90)
        print("TASK 5: SCALABILITY STUDY")
        print("="*90)

        # -----------------------------------------------------------------
        # TASK 5: Solve N-Queens for N = 8, 10, 12, 20
        # -----------------------------------------------------------------
        n_values = [8, 10, 12, 20]
        scalability_results = {}

        for n in n_values:
            print(f"\n--- Testing N-Queens for N={n} ---")
            # Adjust parameters based on problem size
            max_evals = 10000 if n <= 12 else 50000
            pop_size = 100 if n <= 12 else 200

            scalability_results[n] = run_experiment(
                num_runs=30,
                n=n,
                pop_size=pop_size,
                mutation_prob=0.5,
                recombination_rate=1.0,
                max_evaluations=max_evals,
                crossover_type='cut_and_fill',
                mutation_type='swap',
                survival_strategy='fitness_based'
            )

        plot_scalability_study(scalability_results, n_values)
        create_comparison_table(scalability_results, "N Value")

        print("\n" + "="*90)
        print("TASK 5 COMPLETED")
        print("="*90)

        # =====================================================================
        # BASELINE EXPERIMENT (Optional - for reference)
        # =====================================================================
        print("\n" + "="*90)
        print("BASELINE EXPERIMENT (Reference)")
        print("="*90)

        baseline_results = run_experiment(
            num_runs=30,
            n=8,
            pop_size=100,
            mutation_prob=0.5,
            recombination_rate=1.0,
            max_evaluations=10000,
            crossover_type='cut_and_fill',
            mutation_type='swap',
            survival_strategy='fitness_based'
        )

        print_summary_table(baseline_results)
        plot_single_experiment(baseline_results, "Baseline 8-Queens")

        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        print("\n" + "="*90)
        print("ALL EXPERIMENTS COMPLETED!")
        print(f"Results and plots saved in: {os.path.abspath(results_dir)}")
        print("="*90)

    finally:
        # Return to original directory
        os.chdir(original_dir)


# =====================================================================
# INDIVIDUAL TASK RUNNERS (For incremental execution)
# =====================================================================

def run_task_2_only():
    """Run only Task 2: Parameter Sensitivity"""
    results_dir = ensure_results_directory()
    original_dir = os.getcwd()
    os.chdir(results_dir)

    try:
        print("\n" + "="*90)
        print("TASK 2: PARAMETER SENSITIVITY ANALYSIS")
        print("="*90)

        # Task 2.1
        mutation_probs = [0.2, 0.5, 1.0]
        mutation_results = {}
        for mut_prob in mutation_probs:
            print(f"\n--- Mutation Probability: {mut_prob} ---")
            mutation_results[mut_prob] = run_experiment(
                num_runs=30, n=8, pop_size=100, mutation_prob=mut_prob,
                recombination_rate=1.0, max_evaluations=10000,
                crossover_type='cut_and_fill', mutation_type='swap',
                survival_strategy='fitness_based'
            )
        plot_parameter_sensitivity(
            mutation_results, "Mutation Probability", mutation_probs)
        create_comparison_table(mutation_results, "Mutation Probability")

        # Task 2.2
        recombination_rates = [0.5, 1.0]
        recombination_results = {}
        for rate in recombination_rates:
            print(f"\n--- Recombination Rate: {rate} ---")
            recombination_results[rate] = run_experiment(
                num_runs=30, n=8, pop_size=100, mutation_prob=0.5,
                recombination_rate=rate, max_evaluations=10000,
                crossover_type='cut_and_fill', mutation_type='swap',
                survival_strategy='fitness_based'
            )
        plot_parameter_sensitivity(
            recombination_results, "Recombination Rate", recombination_rates)
        create_comparison_table(recombination_results, "Recombination Rate")

        # Task 2.3
        mutation_types = ['swap', 'bitwise']
        mutation_type_results = {}
        for mut_type in mutation_types:
            print(f"\n--- Mutation Type: {mut_type} ---")
            mutation_type_results[mut_type] = run_experiment(
                num_runs=30, n=8, pop_size=100, mutation_prob=0.5,
                recombination_rate=1.0, max_evaluations=10000,
                crossover_type='cut_and_fill', mutation_type=mut_type,
                survival_strategy='fitness_based'
            )
        plot_parameter_sensitivity(
            mutation_type_results, "Mutation Type", mutation_types)
        create_comparison_table(mutation_type_results, "Mutation Type")

    finally:
        os.chdir(original_dir)


def run_task_3_only():
    """Run only Task 3: Crossover Strategies"""
    results_dir = ensure_results_directory()
    original_dir = os.getcwd()
    os.chdir(results_dir)

    try:
        print("\n" + "="*90)
        print("TASK 3: CROSSOVER STRATEGY EXPLORATION")
        print("="*90)

        crossover_types = ['cut_and_fill', 'pmx', '2_cut', '3_cut']
        crossover_results = {}

        for crossover in crossover_types:
            print(f"\n--- Crossover Type: {crossover} ---")
            crossover_results[crossover] = run_experiment(
                num_runs=30, n=8, pop_size=100, mutation_prob=0.5,
                recombination_rate=1.0, max_evaluations=10000,
                crossover_type=crossover, mutation_type='swap',
                survival_strategy='fitness_based'
            )

        plot_crossover_comparison(crossover_results, crossover_types)
        create_comparison_table(crossover_results, "Crossover Type")

    finally:
        os.chdir(original_dir)


def run_task_4_only():
    """Run only Task 4: Survival Strategies"""
    results_dir = ensure_results_directory()
    original_dir = os.getcwd()
    os.chdir(results_dir)

    try:
        print("\n" + "="*90)
        print("TASK 4: SURVIVAL STRATEGY COMPARISON")
        print("="*90)

        survival_strategies = ['fitness_based', 'generational', 'elitism']
        survival_results = {}

        for strategy in survival_strategies:
            print(f"\n--- Survival Strategy: {strategy} ---")
            survival_results[strategy] = run_experiment(
                num_runs=30, n=8, pop_size=100, mutation_prob=0.5,
                recombination_rate=1.0, max_evaluations=10000,
                crossover_type='cut_and_fill', mutation_type='swap',
                survival_strategy=strategy
            )

        plot_survival_strategies(survival_results, survival_strategies)
        create_comparison_table(survival_results, "Survival Strategy")

    finally:
        os.chdir(original_dir)


def run_task_5_only():
    """Run only Task 5: Scalability Study"""
    results_dir = ensure_results_directory()
    original_dir = os.getcwd()
    os.chdir(results_dir)

    try:
        print("\n" + "="*90)
        print("TASK 5: SCALABILITY STUDY")
        print("="*90)

        n_values = [8, 10, 12, 20]
        scalability_results = {}

        for n in n_values:
            print(f"\n--- N-Queens for N={n} ---")
            max_evals = 10000 if n <= 12 else 50000
            pop_size = 100 if n <= 12 else 200

            scalability_results[n] = run_experiment(
                num_runs=30, n=n, pop_size=pop_size, mutation_prob=0.5,
                recombination_rate=1.0, max_evaluations=max_evals,
                crossover_type='cut_and_fill', mutation_type='swap',
                survival_strategy='fitness_based'
            )

        plot_scalability_study(scalability_results, n_values)
        create_comparison_table(scalability_results, "N Value")

    finally:
        os.chdir(original_dir)
