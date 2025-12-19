# QUICK START GUIDE

## Step-by-Step Implementation

### 1. Setup Your Project (5 minutes)
```bash
# Create project directory
mkdir heart_disease_es
cd heart_disease_es

# Copy all boilerplate files here
# Copy your Heart_Disease_dataset.csv here

# Install dependencies
pip install -r requirements.txt
```

### 2. Implementation Order (2-3 hours)

Implement functions in this order for easiest progress:

#### Phase 1: Basic Components (30 min)
1. `sigmoid(z)` - Start here, it's the simplest
2. `predict(X, theta)` - Use sigmoid
3. `fitness(X, y, theta)` - Just return negative loss

#### Phase 2: Loss Function (20 min)
4. `cross_entropy_loss(X, y, theta)` - Core optimization objective

**Test Phase 1 & 2:**
```bash
python test_implementation.py
```

#### Phase 3: Evolution Strategy Core (40 min)
5. `initialize_population(d)` - Create random individuals
6. `mutate(individual, n)` - Self-adaptive mutation

**Test Phase 3:**
```bash
python test_implementation.py
```

#### Phase 4: Training Loop (60 min)
7. `fit(X_train, y_train, X_test, y_test)` - Main ES algorithm
   - This is the longest function
   - Implement step by step
   - Test with small generations first

**Test Phase 4:**
```bash
python test_implementation.py
```

#### Phase 5: Data & Evaluation (30 min)
8. `load_and_preprocess_data(csv_data)` - Data pipeline
9. `evaluate(X_test, y_test)` - Compute metrics
10. `load_csv_file(script_dir)` - File loading
11. `print_diagnostics(script_dir)` - Helper function

#### Phase 6: Visualization (20 min)
12. `plot_results(save_prefix)` - Training curves
13. `plot_confusion_matrix(cm, save_prefix)` - Classification results

#### Phase 7: Main Function (10 min)
14. `main()` - Orchestrate everything

### 3. Test Your Implementation
```bash
# Run all unit tests
python test_implementation.py

# If all pass, run full training
python evolution_strategies_boilerplate.py
```

### 4. Expected Output

#### Console:
```
============================================================
Evolution Strategies for Heart Disease Classification
============================================================

[SUCCESS] Found CSV file: Heart_Disease_dataset.csv
[SUCCESS] Loaded CSV data

Training set size: XXX
Test set size: XXX
...

Generation 10/100 | Best Loss: 0.XXXX | Train Acc: 0.XXXX
...

Final Results on Test Set
Accuracy:  0.XXXX
Precision: 0.XXXX
Recall:    0.XXXX
F1-Score:  0.XXXX
```

#### Files Created:
- `es_results_fitness.png`
- `es_results_accuracy.png`
- `es_results_confusion_matrix.png`

## Tips

### Debugging
- Start with `max_generations=5` to test quickly
- Print shapes of arrays to verify dimensions
- Use `np.set_printoptions(precision=3, suppress=True)` for cleaner output

### Common Issues

**Issue:** "All individuals have same fitness"
**Solution:** Check initialization - make sure theta and sigma have randomness

**Issue:** "Step sizes become zero"
**Solution:** Enforce minimum sigma in mutation (already in pseudo-code)

**Issue:** "Accuracy not improving"
**Solution:** Try different learning rate (tau, tau_prime) or increase population size

### Performance Tuning

After your implementation works:
1. Increase `max_generations` to 100+
2. Experiment with `mu` and `lambda_offspring` ratio
3. Adjust `lambda_reg` for regularization
4. Try different initialization ranges

## Need Help?

Refer to:
1. **IMPLEMENTATION_GUIDE.py** - Detailed pseudo-code
2. **README.md** - Algorithm overview
3. **test_implementation.py** - Expected behavior

## Estimated Time

- **Total implementation:** 3-4 hours
- **Testing & debugging:** 1-2 hours
- **Full training run:** 5-10 minutes
- **Total project:** ~4-6 hours

Good luck! 🚀
