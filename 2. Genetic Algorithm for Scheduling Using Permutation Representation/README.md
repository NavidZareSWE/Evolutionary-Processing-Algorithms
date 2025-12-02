# Genetic Algorithm for Job Scheduling

A Python implementation of a Genetic Algorithm to solve single-machine scheduling problems with sequence-dependent setup times, minimizing total weighted completion time.

## 📋 Problem Description

Given a set of **7 jobs**, each with:
- Processing time (pᵢ)
- Weight/priority (wᵢ)
- Sequence-dependent setup times (sᵢ,ⱼ) between jobs

**Objective:** Find the optimal job sequence that minimizes total weighted completion time.

## 🧬 Genetic Algorithm Components

### Representation
- **Encoding:** Permutation representation
- Each chromosome is a sequence of jobs: `[1, 2, 3, 4, 5, 6, 7]`

### Parameters
- **Population Size:** 50
- **Maximum Generations:** 100
- **Elitism:** 2 (top 2 individuals preserved)
- **Selection:** Tournament selection (k=3)
- **Crossover Rate:** 0.8
- **Mutation Rate:** 0.1

### Crossover Operators
1. **Order Crossover (OX)** - Preserves relative job ordering
2. **Partially Mapped Crossover (PMX)** - Uses position mapping
3. **Cycle Crossover (CX)** - Maintains absolute positions

### Mutation Operators
1. **Swap Mutation** - Exchanges two random jobs
2. **Inversion Mutation** - Reverses a subsequence
3. **Scramble Mutation** - Randomly shuffles a subsequence

## 📊 Dataset

### Job Processing Times
```
Job 1: 12  |  Job 2: 7   |  Job 3: 15  |  Job 4: 5
Job 5: 9   |  Job 6: 11  |  Job 7: 8
```

### Job Weights
```
Job 1: 4   |  Job 2: 9   |  Job 3: 3   |  Job 4: 7
Job 5: 5   |  Job 6: 6   |  Job 7: 8
```

### Setup Time Matrix (sᵢ,ⱼ)
```
     1   2   3   4   5   6   7
1 [  0   4   6   5   7   3   4 ]
2 [  4   0   5   6   4   5   6 ]
3 [  6   5   0   4   5   7   6 ]
4 [  5   6   4   0   3   4   5 ]
5 [  7   4   5   3   0   6   4 ]
6 [  3   5   7   4   6   0   5 ]
7 [  4   6   6   5   4   5   0 ]
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy matplotlib
```

### Running the Algorithm
```bash
python ga_scheduling.py
```

## 📈 Output

The program will output:

1. **Best job sequence found** for each of the 9 operator combinations
2. **Minimum total weighted completion time** achieved
3. **Number of generations executed**
4. **Performance visualizations:**
   - Maximum fitness per generation
   - Average fitness per generation
   - Comparison across all operator combinations

## 📁 Project Structure
```
.
├── ga_scheduling.py          # Main implementation
├── README.md                 # This file
├── ga_all_combinations.png   # Generated: All 9 combinations plot
├── ga_best_solution.png      # Generated: Best solution plot
└── ga_comparison_chart.png   # Generated: Performance comparison
```

## 🧪 Testing All Combinations

The algorithm automatically tests all **9 possible combinations**:
- Order Crossover + {Swap, Inversion, Scramble}
- PMX Crossover + {Swap, Inversion, Scramble}
- Cycle Crossover + {Swap, Inversion, Scramble}

## 📊 Fitness Function

Since Genetic Algorithms maximize fitness, we use:
```
fitness(X) = 1 / T(X)
```

Where `T(X)` is the total weighted completion time:
```
T(X) = Σ(wₖ × Cₖ)
```

## 🎯 Implementation Status

- [ ] Core GA infrastructure
- [ ] Objective function calculation
- [ ] Population initialization
- [ ] Tournament selection
- [ ] Order Crossover (OX)
- [ ] PMX Crossover
- [ ] Cycle Crossover (CX)
- [ ] Swap Mutation
- [ ] Inversion Mutation
- [ ] Scramble Mutation
- [ ] Main evolution loop
- [ ] Experiment runner
- [ ] Visualization functions

## 📝 Assignment Requirements

### Required Deliverables
1. [ ] Source code implementing GA from scratch
2. [ ] Max/average fitness plot
3. [ ] Report including:
   - Description of encoding and operators
   - Parameter settings
   - Fitness plots
   - Final solution and performance discussion

## 🔍 Key Formulas

**Completion Time Calculation:**
```
C(x₁) = p(x₁)
C(xₖ) = C(xₖ₋₁) + s(xₖ₋₁,xₖ) + p(xₖ)
```

**Total Weighted Completion Time:**
```
T(X) = Σ w(xₖ) × C(xₖ)  for k=1 to n
```


---

**Note:** This implementation is built from scratch without using external GA libraries, as required by the assignment specifications.
