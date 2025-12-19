# Evolution Strategies for Heart Disease Classification

A complete implementation of Evolution Strategies (ES) for binary classification using logistic regression with self-adaptive mutation.

## Project Structure

```
project/
│
├── evolution_strategies_boilerplate.py  # Main implementation file
├── config.py                            # Hyperparameters and configuration
├── requirements.txt                     # Python dependencies
├── Heart_Disease_dataset.csv           # Dataset (place here)
├── README.md                           # This file
│
└── outputs/                            # Generated outputs (created automatically)
    ├── es_results_fitness.png
    ├── es_results_accuracy.png
    └── es_results_confusion_matrix.png
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Dataset
Make sure `Heart_Disease_dataset.csv` is in the same directory as the Python script.

### 3. Run
```bash
python evolution_strategies_boilerplate.py
```

## Implementation Checklist

### EvolutionStrategyLogisticRegression Class

- [ ] `sigmoid(z)` - Implement logistic sigmoid with numerical stability
- [ ] `cross_entropy_loss(X, y, theta)` - Compute loss with L2 regularization
- [ ] `fitness(X, y, theta)` - Return negative loss
- [ ] `predict(X, theta)` - Make binary predictions
- [ ] `initialize_population(d)` - Create initial population with random θ and σ
- [ ] `mutate(individual, n)` - Self-adaptive mutation
- [ ] `fit(X_train, y_train, X_test, y_test)` - Main training loop
- [ ] `evaluate(X_test, y_test)` - Compute all metrics
- [ ] `plot_results(save_prefix)` - Generate training plots
- [ ] `plot_confusion_matrix(cm, save_prefix)` - Visualize confusion matrix

### Helper Functions

- [ ] `load_and_preprocess_data(csv_data)` - Load CSV and preprocess
- [ ] `load_csv_file(script_dir)` - Find and load CSV file
- [ ] `print_diagnostics(script_dir)` - Print file listing
- [ ] `main()` - Orchestrate the entire pipeline

## Algorithm Overview

### Evolution Strategy (μ + λ)-ES

1. **Initialize** population of μ parents with random θ (weights, bias) and σ (step sizes)
2. **For each generation:**
   - Evaluate fitness of all individuals
   - Generate λ offspring through mutation
   - Select μ best from parents + offspring (or offspring only for comma-selection)
3. **Self-Adaptive Mutation:**
   - Update σ: σ' = σ * exp(τ·N(0,1) + τ'·Nᵢ(0,1))
   - Update θ: θ' = θ + σ'·N(0,1)
   
Where:
- τ = 1/√(2n)
- τ' = 1/√(2√n)

## Hyperparameters

Edit `config.py` to modify:
- μ (mu): Number of parents (default: 30)
- λ (lambda): Number of offspring (default: 210)
- Regularization strength (default: 0.01)
- Max generations (default: 100)
- Selection type: 'plus' or 'comma'

## Expected Outputs

### Console Output
- Dataset statistics
- Training progress every 10 generations
- Final test metrics (accuracy, precision, recall, F1)
- Confusion matrix

### Plots
1. **Training Loss Plot**: Best and mean loss over generations
2. **Accuracy Plot**: Training and test accuracy over generations
3. **Confusion Matrix**: Final classification results

## Dataset

The Heart Disease dataset should have:
- Multiple feature columns (age, sex, cp, trestbps, etc.)
- Last column: target (0 = no disease, 1+ = disease present)
- Preprocessing will binarize target to 0 vs 1

## Tips for Implementation

### Numerical Stability
- Clip z values before sigmoid to avoid overflow
- Use epsilon to avoid log(0)
- Enforce minimum step sizes in mutation

### Debugging
- Start with small max_generations (e.g., 10) to test quickly
- Print intermediate values to verify correctness
- Check population diversity (variance in fitness)

### Performance Tuning
- Increase λ/μ ratio for more exploration
- Adjust lambda_reg to prevent overfitting
- Experiment with initialization ranges

## References

- Schwefel, H. P. (1993). Evolution and Optimum Seeking
- Rechenberg, I. (1973). Evolutionsstrategie
- Hansen, N., & Ostermeier, A. (2001). Completely Derandomized Self-Adaptation in Evolution Strategies

## License

This is an educational project for learning Evolution Strategies.
