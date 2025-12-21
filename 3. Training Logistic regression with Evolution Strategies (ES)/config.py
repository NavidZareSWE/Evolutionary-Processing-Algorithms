"""
Configuration file for Evolution Strategies Sensitivity Analysis
Tests variations in population size, learning rates, and regularization
"""

# Base configuration (your current settings)
BASE_CONFIG = {
    'name': 'baseline',
    'mu': 30,
    'lambda_offspring': 210,
    'lambda_reg': 0.01,
    'max_generations': 100,
    'tau_multiplier': 1.0,           # NEW: Standard learning rate
    'tau_prime_multiplier': 1.0,      # NEW: Standard learning rate
}

# All experimental configurations
EXPERIMENTS = [
    # Baseline configuration
    BASE_CONFIG,

    # ==========================================
    # Population Size Variations (μ, λ)
    # ==========================================
    {
        'name': 'small_population',
        'mu': 15,
        'lambda_offspring': 105,  # 7 * mu
        'lambda_reg': 0.01,
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },
    {
        'name': 'large_population',
        'mu': 50,
        'lambda_offspring': 350,  # 7 * mu
        'lambda_reg': 0.01,
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },
    {
        'name': 'very_large_population',
        'mu': 100,
        'lambda_offspring': 700,  # 7 * mu
        'lambda_reg': 0.01,
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },

    # ==========================================
    # Regularization Coefficient Variations (λ_reg)
    # ==========================================
    {
        'name': 'no_regularization',
        'mu': 30,
        'lambda_offspring': 210,
        'lambda_reg': 0.0,  # No L2 regularization
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },
    {
        'name': 'weak_regularization',
        'mu': 30,
        'lambda_offspring': 210,
        'lambda_reg': 0.001,  # Weaker regularization
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },
    {
        'name': 'strong_regularization',
        'mu': 30,
        'lambda_offspring': 210,
        'lambda_reg': 0.1,  # Stronger regularization
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },
    {
        'name': 'very_strong_regularization',
        'mu': 30,
        'lambda_offspring': 210,
        'lambda_reg': 0.5,  # Very strong regularization
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },

    # ==========================================
    # Learning Rate Variations (τ, τ')
    # NEW: These multiply the base learning rate formulas
    # τ = (1/√(2n)) * tau_multiplier
    # τ' = (1/√(2√n)) * tau_prime_multiplier
    # ==========================================
    {
        'name': 'low_learning_rate',
        'mu': 30,
        'lambda_offspring': 210,
        'lambda_reg': 0.01,
        'max_generations': 100,
        'tau_multiplier': 0.5,        # Half the standard learning rate
        'tau_prime_multiplier': 0.5,
    },
    {
        'name': 'high_learning_rate',
        'mu': 30,
        'lambda_offspring': 210,
        'lambda_reg': 0.01,
        'max_generations': 100,
        'tau_multiplier': 2.0,        # Double the standard learning rate
        'tau_prime_multiplier': 2.0,
    },
    {
        'name': 'very_high_learning_rate',
        'mu': 30,
        'lambda_offspring': 210,
        'lambda_reg': 0.01,
        'max_generations': 100,
        'tau_multiplier': 5.0,        # 5x the standard learning rate
        'tau_prime_multiplier': 5.0,
    },

    # ==========================================
    # Combined Variations
    # ==========================================
    {
        'name': 'small_pop_high_reg',
        'mu': 15,
        'lambda_offspring': 105,
        'lambda_reg': 0.1,
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },
    {
        'name': 'large_pop_low_reg',
        'mu': 50,
        'lambda_offspring': 350,
        'lambda_reg': 0.001,
        'max_generations': 100,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    }
]

# Quick test configuration (fewer generations for rapid testing)
QUICK_TEST_EXPERIMENTS = [
    {
        'name': 'quick_baseline',
        'mu': 20,
        'lambda_offspring': 140,
        'lambda_reg': 0.01,
        'max_generations': 20,  # Reduced for quick testing
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    },
    {
        'name': 'quick_high_reg',
        'mu': 20,
        'lambda_offspring': 140,
        'lambda_reg': 0.1,
        'max_generations': 20,
        'tau_multiplier': 1.0,
        'tau_prime_multiplier': 1.0,
    }
]
