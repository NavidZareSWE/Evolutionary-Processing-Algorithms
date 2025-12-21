# Evolution Strategies for Heart Disease Classification

A Python implementation of **(μ, λ)-Evolution Strategies** for training a Logistic Regression classifier on a heart disease dataset.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Output](#output)
- [Notes for Reproduction](#notes-for-reproduction)

---

## Overview

This project implements an evolutionary algorithm approach to train a logistic regression model for binary classification (heart disease prediction). Instead of gradient-based optimization, the model uses **(μ, λ)-ES** with:

- **Self-adaptive mutation** (σ evolves alongside θ)
- **Local discrete recombination** for crossover
- **Cross-entropy loss** with L2 regularization
- **Z-score standardization** for feature scaling

---

## Requirements

### Python Version

- Python 3.8 or higher (recommended: Python 3.10+)

### Dependencies

| Package         | Version (Tested) | Purpose                          |
|-----------------|------------------|----------------------------------|
| numpy           | ≥1.21.0          | Numerical operations             |
| pandas          | ≥1.3.0           | Data loading and manipulation    |
| matplotlib      | ≥3.4.0           | Plotting and visualization       |
| scikit-learn    | ≥0.24.0          | Train/test split, metrics        |

---

## Installation

### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd <project-folder>

# Or simply place all files in a folder
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt** (create this file if needed):
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
```

---

## Project Structure

```
project/
├── evolution_strategies.py    # Main script with ES implementation
├── config.py                  # Experiment configurations (EXPERIMENTS list)
├── Heart_Disease_dataset.csv  # Dataset file
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## Configuration

You must create a `config.py` file in the same directory as `evolution_strategies.py`. This file defines the experiments to run.

### Example `config.py`

```python
EXPERIMENTS = [
    {
        'name': 'baseline',
        'mu': 30,
        'lambda_offspring': 210,  # 7 * mu
        'lambda_reg': 0.01,
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },
    {
        'name': 'high_mutation',
        'mu': 50,
        'lambda_offspring': 350,
        'lambda_reg': 0.001,
        'max_generations': 150,
        'tau_multiplier': 1.5,
        'tau_prime_multiplier': 1.5,
    },
    # Add more experiments as needed
]
```

### Configuration Parameters

| Parameter             | Description                                      | Default        |
|-----------------------|--------------------------------------------------|----------------|
| `name`                | Experiment identifier (used in output files)     | Required       |
| `mu`                  | Number of parent individuals                     | 30             |
| `lambda_offspring`    | Number of offspring per generation               | 7 × μ          |
| `lambda_reg`          | L2 regularization strength                       | 0.01           |
| `max_generations`     | Number of generations to evolve                  | 100            |
| `tau_multiplier`      | Global learning rate multiplier for σ            | 1.0            |
| `tau_prime_multiplier`| Local learning rate multiplier for σ             | 1.0            |

---

## How to Run

### 1. Prepare the Dataset

Place your heart disease CSV file in one of the following locations:

- Same directory as `evolution_strategies.py`
- `./Extras/` subdirectory
- `./data/` subdirectory

**Supported file names:**
- `Heart_Disease_dataset.csv`
- `Heart Disease dataset.csv`
- `heart_disease_dataset.csv`
- `heart.csv`
- `heart-disease.csv`

The CSV should have:
- Feature columns (all columns except the last)
- Target column (last column) — will be binarized: 0 → 0, any value > 0 → 1

### 2. Create `config.py`

Ensure you have a `config.py` file with at least one experiment defined (see [Configuration](#configuration) section above).

### 3. Run the Script

```bash
python evolution_strategies.py
```

### Expected Console Output

```
============================================================
Evolution Strategies for Heart Disease Classification
============================================================

[SUCCESS] Found CSV file: /path/to/Heart_Disease_dataset.csv
[SUCCESS] Loaded CSV data (X characters)

Training set size: XXX
Test set size: XXX
Number of features: XX
Class distribution (train): [X X]
Class distribution (test): [X X]

============================================================
Running Experiment: baseline
============================================================
Hyperparameters:
  - mu (parents): 30
  - lambda (offspring): 210
  ...

Training Evolution Strategy...

Gen   1/100 | Best Loss: X.XXXX | Train Acc: X.XXXX
Gen  10/100 | Best Loss: X.XXXX | Train Acc: X.XXXX
...

============================================================
Final Results on Test Set - baseline
============================================================
Accuracy:  X.XXXX
Precision: X.XXXX
Recall:    X.XXXX
F1-Score:  X.XXXX

Confusion Matrix:
[[XX XX]
 [XX XX]]

Generating plots...
Plots saved for baseline!
```

---

## Output

For each experiment, the script generates three PNG files:

| File                                | Description                              |
|-------------------------------------|------------------------------------------|
| `{name}_results_fitness.png`        | Best and mean training loss over generations |
| `{name}_results_accuracy.png`       | Training and test accuracy over generations  |
| `{name}_results_confusion_matrix.png` | Confusion matrix heatmap                   |

Files are saved in the current working directory.

---

## Notes for Reproduction

### Random Seed

The script sets a fixed random seed for reproducibility:

```python
np.random.seed(42)
```

The train/test split also uses `random_state=42`.

### Data Split

- **70% Training / 30% Test**
- **Stratified split** to preserve class distribution

### Feature Scaling

- Z-score standardization is applied using training set statistics
- The same mean and standard deviation are applied to the test set

### Algorithm Details

- **Selection**: (μ, λ)-ES — only offspring compete; parents do not survive
- **Recombination**: Local discrete (gene-by-gene from random parents)
- **Mutation**: Self-adaptive with log-normal update for step sizes
- **Fitness**: Negative cross-entropy loss (maximization problem)

### Troubleshooting

| Issue                          | Solution                                              |
|--------------------------------|-------------------------------------------------------|
| `ModuleNotFoundError: config`  | Create `config.py` with `EXPERIMENTS` list            |
| CSV file not found             | Place dataset in script directory or `./data/`        |
| Division by zero warnings      | Normal if features have zero variance (handled in code) |
| Low accuracy                   | Try increasing `max_generations` or adjusting `mu`/`lambda` |

---

