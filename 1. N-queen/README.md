# N-Queens Genetic Algorithm - Comprehensive Experiments

## Overview
This project implements a Genetic Algorithm to solve the N-Queens problem with comprehensive experiments and analysis.

## File Structure
```
.
├── individual.py                   # Individual class definition
├── crossover_functions.py          # Crossover implementations
├── mutation_functions.py           # Mutation implementations
├── survival_functions.py           # Survival strategy implementations
├── utility_functions.py            # Helper functions
├── queens_ga.py                    # Main GA implementation
├── plotting_functions.py           # Visualization functions (UPDATED)
├── comprehensive_experiments.py    # Main experiment runner (NEW)
└── Results/                        # Output directory (auto-created)
    ├── *.png                       # All generated plots
    └── *.csv                       # Comparison tables
```

## Setup

### Prerequisites
```bash
pip install numpy matplotlib pandas
```

### Required Files
Make sure you have all the following files in your directory:
- `individual.py`
- `crossover_functions.py`
- `mutation_functions.py`
- `survival_functions.py`
- `utility_functions.py`
- `queens_ga.py`
- `plotting_functions.py` (use the updated version)
- `comprehensive_experiments.py`

## How to Run

### Option 1: Run All Experiments (Recommended)
This will run all experiments required for the assignment:

```bash
python comprehensive_experiments.py
```

**Note:** This will take a significant amount of time (30-60 minutes) as it runs:
- 30 runs × 3 mutation probabilities = 90 runs
- 30 runs × 2 recombination rates = 60 runs
- 30 runs × 2 mutation types = 60 runs
- 30 runs × 4 crossover types = 120 runs
- 30 runs × 3 survival strategies = 90 runs
- 30 runs × 4 N-values = 120 runs
- **Total: ~540 runs**

### Option 2: Run Specific Experiments
You can modify `comprehensive_experiments.py` to run only specific experiments by commenting out sections.

### Option 3: Quick Test Run
To test that everything works, modify the `num_runs` parameter:

```python
# In comprehensive_experiments.py, change:
mutation_results[mut_prob] = run_experiment(
    num_runs=3,  # Changed from 30 to 3 for quick testing
    ...
)
```

## Output

### Generated Files
All outputs are saved in the `Results/` directory:

#### Plots (PNG files):
1. **Parameter Sensitivity:**
   - `parameter_sensitivity_mutation_probability.png`
   - `parameter_sensitivity_recombination_rate.png`
   - `parameter_sensitivity_mutation_type.png`

2. **Crossover Comparison:**
   - `crossover_comparison.png`

3. **Survival Strategies:**
   - `survival_strategy_comparison.png`

4. **Scalability Study:**
   - `scalability_study.png`

5. **Baseline Results:**
   - `Baseline_8-Queens_results.png`

#### Data Tables (CSV files):
- `mutation_probability_comparison.csv`
- `recombination_rate_comparison.csv`
- `mutation_type_comparison.csv`
- `crossover_type_comparison.csv`
- `survival_strategy_comparison.csv`
- `n_value_comparison.csv`

## Experiments Explained

### Task 2: Parameter Sensitivity Analysis
Tests how different parameters affect GA performance:
- **Mutation Probability:** 20%, 50%, 100%
- **Recombination Rate:** 50%, 100%
- **Mutation Type:** Swap vs Bitwise

**Metrics tracked:**
- Success rate
- Average generations to solution
- Evaluations used
- Convergence speed

### Task 3: Crossover Strategy Exploration
Compares different crossover methods:
- Cut-and-Fill (baseline)
- PMX (Partially Mapped Crossover)
- 2-Cut
- 3-Cut

**Analysis:**
- Which converges faster?
- Which is more stable?
- Effect of increasing cut points

### Task 4: Survival Strategy Comparison
Evaluates different replacement strategies:
- Fitness-based replacement
- Generational replacement
- Elitism (keeping 2 best)

**Questions answered:**
- Which converges faster?
- Which finds better solutions?
- When to use each strategy?

### Task 5: Scalability Study
Tests how the algorithm scales with problem size:
- N = 8, 10, 12, 20

**Metrics:**
- Success rate vs N
- Generations needed vs N
- Evaluations vs N (log scale)
- Normalized complexity (generations/N²)

## Understanding the Plots

### Convergence Curves
- **X-axis:** Generation number
- **Y-axis:** Best fitness (0 = optimal solution)
- **Lower is better** (fewer conflicts)
- Shaded areas show standard deviation across runs

### Success Rate Charts
- Percentage of runs that found a valid solution
- Higher is better

### Box Plots
- Show distribution of values across all runs
- Middle line = median
- Box = 25th to 75th percentile
- Whiskers = min/max (excluding outliers)

## Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution:** Install required packages:
```bash
pip install numpy matplotlib pandas
```

### Problem: "No such file or directory: 'Results/...'"
**Solution:** The script should auto-create the Results directory. If it doesn't:
```bash
mkdir Results
```

### Problem: Script takes too long
**Solution:** Reduce `num_runs` from 30 to 3-5 for testing:
```python
num_runs=5  # Instead of 30
```

### Problem: Plots not showing
**Solution:** Plots are automatically saved to `Results/` directory. They won't display on screen by default. To view them:
- Navigate to the `Results/` folder
- Open the `.png` files with an image viewer

## Customization

### Change Number of Runs
```python
# In comprehensive_experiments.py
num_runs=10  # Change this value
```

### Change Parameters Tested
```python
# Example: Test different mutation probabilities
mutation_probs = [0.1, 0.3, 0.7, 0.9]  # Add more values
```

### Change GA Settings
```python
# In run_experiment calls, modify:
pop_size=200,           # Default: 100
max_evaluations=20000,  # Default: 10000
```

### Add Custom Experiments
```python
# Add your own experiment section:
custom_results = run_experiment(
    num_runs=30,
    n=15,  # Try 15-Queens
    pop_size=150,
    mutation_prob=0.6,
    # ... other parameters
)
plot_single_experiment(custom_results, "Custom 15-Queens Experiment")
```

## Expected Runtime

### Full Experiment Suite (num_runs=30):
- **Estimated time:** 30-60 minutes
- **Total runs:** ~540

### Quick Test (num_runs=3):
- **Estimated time:** 3-6 minutes
- **Total runs:** ~54

### Single Experiment (num_runs=30):
- **Estimated time:** ~3-5 minutes

## Tips for Your Report

1. **Include all plots** from the Results directory
2. **Reference the CSV tables** for exact numbers
3. **Discuss trends** visible in convergence curves
4. **Compare success rates** across different configurations
5. **Analyze scalability** - what happens as N increases?
6. **Explain why** certain configurations work better

## Questions Answered by Each Plot

### Parameter Sensitivity Plots:
- Which mutation probability works best?
- Does 100% recombination help or hurt?
- Is swap or bitwise mutation better?

### Crossover Comparison:
- Which crossover preserves good solutions better?
- Do more cuts improve performance?

### Survival Strategy:
- Does elitism prevent premature convergence?
- Is generational replacement too disruptive?

### Scalability Study:
- Does the algorithm scale well to larger boards?
- Is the growth polynomial or exponential?
- At what N does it become impractical?

## Contact & Support

For questions about the code or experiments, review:
1. The comments in each Python file
2. The plotting function docstrings
3. The GA implementation in `queens_ga.py`

## License
This code is for educational purposes for the Evolutionary Computing course.