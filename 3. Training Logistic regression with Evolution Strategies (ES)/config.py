"""
Configuration file for Evolution Strategies hyperparameters
Modify these values to experiment with different settings
"""

# Evolution Strategy Parameters
ES_CONFIG = {
    # Population parameters
    'mu': 30,                      # Number of parents
    'lambda_offspring': 210,       # Number of offspring (typically 7 * mu)
    
    # Regularization
    'lambda_reg': 0.01,            # L2 regularization coefficient
    
    # Training parameters
    'max_generations': 100,        # Maximum number of generations
    'selection_type': 'plus',      # 'plus' for (mu+lambda), 'comma' for (mu,lambda)
    
    # Random seed
    'random_seed': 42,
}

# Data preprocessing parameters
DATA_CONFIG = {
    'test_size': 0.3,              # Proportion of test set (30%)
    'stratify': True,              # Use stratified split
}

# Initialization parameters
INIT_CONFIG = {
    'theta_init_range': (-0.1, 0.1),   # Range for initializing theta
    'sigma_init_range': (0.01, 0.1),   # Range for initializing step sizes
    'min_sigma': 1e-6,                  # Minimum step size
    'theta_clip_range': (-5, 5),       # Range to clip parameters
}

# Logging parameters
LOG_CONFIG = {
    'print_every': 10,             # Print progress every N generations
    'plot_dpi': 300,               # DPI for saved plots
}
