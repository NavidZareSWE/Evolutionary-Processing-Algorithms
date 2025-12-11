export interface Formula {
  id: string;
  name: string;
  latex: string;
  latexDisplay: string;
  variables: Record<string, string>;
  derivation?: string[];
  plainEnglish: string;
  whenToUse: string;
  example?: string;
  relatedFormulas?: string[];
}

export const FORMULAS: Record<string, Formula> = {
  expectedCopiesFPS: {
    id: "expectedCopiesFPS",
    name: "Expected Copies (Fitness-Proportionate Selection)",
    latex: "E(n_i) = \\mu \\cdot \\frac{f(i)}{\\langle f \\rangle}",
    latexDisplay: "$$E(n_i) = \\mu \\cdot \\frac{f(i)}{\\langle f \\rangle}$$",
    variables: {
      "E(n_i)": "Expected number of copies of individual i in next generation",
      "μ": "Population size",
      "f(i)": "Fitness of individual i",
      "⟨f⟩": "Average fitness of the population"
    },
    derivation: [
      "Step 1: In FPS, probability of selecting individual i is P(i) = f(i) / Σⱼf(j)",
      "Step 2: Total fitness Σⱼf(j) = μ × ⟨f⟩ (population size times average)",
      "Step 3: So P(i) = f(i) / (μ × ⟨f⟩)",
      "Step 4: We make μ independent selections",
      "Step 5: Expected copies E(n_i) = μ × P(i) = μ × f(i) / (μ × ⟨f⟩)",
      "Step 6: Simplifying: E(n_i) = f(i) / ⟨f⟩"
    ],
    plainEnglish: "The expected number of copies of an individual equals its fitness divided by the average fitness, times the population size. If your fitness equals the average, you expect 1 copy. If twice the average, expect 2 copies.",
    whenToUse: "When analyzing fitness-proportionate selection behavior or comparing individual fitness to population average.",
    example: "If population has average fitness 50, individual with fitness 100 expects 2 copies, individual with fitness 25 expects 0.5 copies.",
    relatedFormulas: ["varianceTheorem", "linearRanking"]
  },

  linearRanking: {
    id: "linearRanking",
    name: "Linear Ranking Selection Probability",
    latex: "P(i) = \\frac{1}{\\mu}\\left[s - (s-1)\\frac{rank(i)-1}{\\mu-1}\\right]",
    latexDisplay: "$$P(i) = \\frac{1}{\\mu}\\left[s - (s-1)\\frac{rank(i)-1}{\\mu-1}\\right]$$",
    variables: {
      "P(i)": "Probability of selecting individual i",
      "μ": "Population size",
      "s": "Selection pressure parameter (1.0 < s ≤ 2.0)",
      "rank(i)": "Rank of individual i (1 = worst, μ = best)"
    },
    derivation: [
      "Step 1: Assign ranks from 1 (worst) to μ (best)",
      "Step 2: Best individual gets probability s/μ",
      "Step 3: Worst individual gets probability (2-s)/μ",
      "Step 4: Intermediate individuals linearly interpolated",
      "Step 5: For s=2: best gets 2/μ, worst gets 0",
      "Step 6: For s=1: all get 1/μ (uniform selection)"
    ],
    plainEnglish: "Selection probability depends on RANK, not fitness value. Parameter s controls selection pressure: s=1 means uniform random, s=2 means best has twice the average probability.",
    whenToUse: "When you want selection pressure independent of fitness scaling. More robust than FPS.",
    example: "With μ=10 and s=2: Best (rank 10) gets P=0.2, Worst (rank 1) gets P=0. Middle (rank 5) gets P=0.1.",
    relatedFormulas: ["expectedCopiesFPS", "tournamentExpected"]
  },

  schemaTheorem: {
    id: "schemaTheorem",
    name: "Holland's Schema Theorem",
    latex: "m(H,t+1) \\geq m(H,t) \\cdot \\frac{f(H)}{\\langle f \\rangle} \\cdot \\left[1 - p_c \\cdot \\frac{\\delta(H)}{l-1}\\right] \\cdot (1-p_m)^{o(H)}",
    latexDisplay: "$$m(H,t+1) \\geq m(H,t) \\cdot \\frac{f(H)}{\\langle f \\rangle} \\cdot \\left[1 - p_c \\cdot \\frac{\\delta(H)}{l-1}\\right] \\cdot (1-p_m)^{o(H)}$$",
    variables: {
      "m(H,t)": "Proportion of population matching schema H at generation t",
      "f(H)": "Average fitness of individuals matching schema H",
      "⟨f⟩": "Average fitness of entire population",
      "p_c": "Crossover probability",
      "δ(H)": "Defining length of schema H",
      "l": "Chromosome length",
      "p_m": "Per-bit mutation probability",
      "o(H)": "Order of schema H"
    },
    derivation: [
      "Factor 1 - SELECTION: f(H)/⟨f⟩",
      "  - Above-average schemata (f(H) > ⟨f⟩) have factor > 1, so they GROW",
      "  - Below-average schemata have factor < 1, so they SHRINK",
      "",
      "Factor 2 - CROSSOVER SURVIVAL: 1 - p_c × δ(H)/(l-1)",
      "  - Schema survives if crossover point falls outside defining length",
      "  - There are δ(H) 'dangerous' cut points out of l-1 possible",
      "  - Short defining length → higher survival probability",
      "",
      "Factor 3 - MUTATION SURVIVAL: (1-p_m)^o(H)",
      "  - All o(H) defined positions must NOT be mutated",
      "  - Each survives with probability (1-p_m)",
      "  - Low order → higher survival probability",
      "",
      "WHY ≥ (inequality, not equality):",
      "  - Ignores CONSTRUCTIVE effects (crossover/mutation creating schema members)",
      "  - Only counts destructive effects",
      "  - Actual proportion may be HIGHER than this lower bound"
    ],
    plainEnglish: "Schemata with above-average fitness, short defining length, and low order will increase in frequency. The theorem gives a LOWER BOUND on growth because it ignores constructive effects.",
    whenToUse: "For understanding GA behavior, designing representations (keep related genes close), explaining why short fit schemata grow.",
    example: "Schema 1##0 with above-average fitness, short δ=3, low o=2 will grow faster than schema 1#####0 with long δ=6.",
    relatedFormulas: ["schemaDisruptionXover", "schemaDisruptionMutation"]
  },

  schemaDisruptionXover: {
    id: "schemaDisruptionXover",
    name: "Schema Survival Under Crossover",
    latex: "P_{survive} = 1 - p_c \\cdot \\frac{\\delta(H)}{l-1}",
    latexDisplay: "$$P_{survive} = 1 - p_c \\cdot \\frac{\\delta(H)}{l-1}$$",
    variables: {
      "P_survive": "Probability schema survives crossover",
      "p_c": "Crossover probability",
      "δ(H)": "Defining length of schema",
      "l": "Chromosome length"
    },
    derivation: [
      "Step 1: 1-point crossover selects one of l-1 possible cut points",
      "Step 2: Schema is disrupted if cut falls within defining length",
      "Step 3: There are δ(H) such 'dangerous' positions",
      "Step 4: P(disruption | crossover occurs) = δ(H)/(l-1)",
      "Step 5: Crossover occurs with probability p_c",
      "Step 6: P(disruption) = p_c × δ(H)/(l-1)",
      "Step 7: P(survival) = 1 - P(disruption)"
    ],
    plainEnglish: "A schema is more likely to survive crossover if it has a short defining length (genes close together). The formula assumes 1-point crossover.",
    whenToUse: "When analyzing how crossover affects schema propagation, or when designing representations.",
    example: "Schema 1#0 (δ=2) in chromosome of length 10 with p_c=0.8: P_survive = 1 - 0.8×2/9 = 0.82",
    relatedFormulas: ["schemaTheorem", "schemaDisruptionMutation"]
  },

  schemaDisruptionMutation: {
    id: "schemaDisruptionMutation",
    name: "Schema Survival Under Mutation",
    latex: "P_{survive} = (1-p_m)^{o(H)} \\approx 1 - o(H) \\cdot p_m",
    latexDisplay: "$$P_{survive} = (1-p_m)^{o(H)} \\approx 1 - o(H) \\cdot p_m$$",
    variables: {
      "P_survive": "Probability schema survives mutation",
      "p_m": "Per-bit mutation probability",
      "o(H)": "Order of schema (number of defined positions)"
    },
    derivation: [
      "Step 1: Each of o(H) defined positions must NOT be mutated",
      "Step 2: Each position survives with probability (1-p_m)",
      "Step 3: All positions must survive (independent events)",
      "Step 4: P(survival) = (1-p_m)^o(H)",
      "Step 5: Taylor approximation for small p_m: (1-p_m)^o(H) ≈ 1 - o(H)×p_m"
    ],
    plainEnglish: "A schema is more likely to survive mutation if it has low order (few defined positions). The approximation works when mutation probability is small.",
    whenToUse: "When analyzing mutation's effect on schemata, or when setting mutation rates.",
    example: "Schema 1#1#0 (o=3) with p_m=0.01: P_survive = (0.99)³ ≈ 0.97",
    relatedFormulas: ["schemaTheorem", "schemaDisruptionXover"]
  },

  tournamentExpected: {
    id: "tournamentExpected",
    name: "Tournament Selection Expected Distribution",
    latex: "\\hat{s}(f_i) = N \\cdot \\left[\\left(\\frac{C(f_i)}{N}\\right)^t - \\left(\\frac{C(f_{i-1})}{N}\\right)^t\\right]",
    latexDisplay: "$$\\hat{s}(f_i) = N \\cdot \\left[\\left(\\frac{C(f_i)}{N}\\right)^t - \\left(\\frac{C(f_{i-1})}{N}\\right)^t\\right]$$",
    variables: {
      "ŝ(f_i)": "Expected number with fitness f_i AFTER selection",
      "N": "Population size",
      "C(f_i)": "Cumulative count: individuals with fitness ≤ f_i",
      "t": "Tournament size"
    },
    derivation: [
      "Step 1: In tournament, winner has highest fitness among t random picks",
      "Step 2: Winner has fitness ≤ f_i means ALL t competitors have fitness ≤ f_i",
      "Step 3: P(all t ≤ f_i) = [C(f_i)/N]^t",
      "Step 4: Cumulative distribution after selection: Ĉ(f_i) = N × [C(f_i)/N]^t",
      "Step 5: Number with EXACTLY f_i: ŝ(f_i) = Ĉ(f_i) - Ĉ(f_{i-1})"
    ],
    plainEnglish: "The expected number of individuals with a given fitness after tournament selection. Higher tournament size (t) increases selection pressure on high-fitness individuals.",
    whenToUse: "For analyzing tournament selection pressure and understanding how t affects selection.",
    example: "t=1: ŝ(f_i) = s(f_i), distribution unchanged (random selection). As t increases, distribution shifts toward higher fitness.",
    relatedFormulas: ["tournamentConcatenation", "reproductionRate"]
  },

  tournamentConcatenation: {
    id: "tournamentConcatenation",
    name: "Tournament Concatenation Theorem",
    latex: "\\Omega_{t_2}(\\Omega_{t_1}(s)) = \\Omega_{t_1 \\cdot t_2}(s)",
    latexDisplay: "$$\\Omega_{t_2}(\\Omega_{t_1}(s)) = \\Omega_{t_1 \\cdot t_2}(s)$$",
    variables: {
      "Ω_t": "Tournament selection operator with size t",
      "s": "Initial fitness distribution",
      "t_1, t_2": "Tournament sizes"
    },
    derivation: [
      "Step 1: Ω_t1 transforms s to distribution with [C(f)/N]^t1",
      "Step 2: Ω_t2 applied to result gives [([C(f)/N]^t1)]^t2",
      "Step 3: This equals [C(f)/N]^(t1×t2)",
      "Step 4: Which is exactly Ω_{t1×t2}(s)"
    ],
    plainEnglish: "Applying tournament selection with size t1, then with size t2, is equivalent to applying tournament selection once with size t1×t2. Selection pressures multiply.",
    whenToUse: "When analyzing multiple rounds of selection or understanding cumulative selection effects.",
    example: "Two rounds of binary tournament (t=2 twice) equals one round of t=4 tournament.",
    relatedFormulas: ["tournamentExpected"]
  },

  reproductionRate: {
    id: "reproductionRate",
    name: "Reproduction Rate",
    latex: "R(f_i) = \\frac{\\hat{s}(f_i)}{s(f_i)}",
    latexDisplay: "$$R(f_i) = \\frac{\\hat{s}(f_i)}{s(f_i)}$$",
    variables: {
      "R(f_i)": "Reproduction rate for fitness class f_i",
      "ŝ(f_i)": "Expected count after selection",
      "s(f_i)": "Count before selection"
    },
    derivation: [
      "Step 1: s(f_i) = number of individuals with fitness f_i BEFORE selection",
      "Step 2: ŝ(f_i) = expected number AFTER selection",
      "Step 3: Ratio R tells us if this fitness class grows or shrinks",
      "Step 4: R > 1: fitness class GROWS (gains individuals)",
      "Step 5: R < 1: fitness class SHRINKS (loses individuals)",
      "Step 6: R = 1: fitness class unchanged"
    ],
    plainEnglish: "The reproduction rate tells us whether a fitness class will grow or shrink under selection. Values above 1 mean growth, below 1 mean shrinkage.",
    whenToUse: "For understanding which fitness levels gain or lose representation under selection.",
    example: "If s(f_i)=10 before and ŝ(f_i)=15 after selection, R=1.5 meaning 50% growth.",
    relatedFormulas: ["tournamentExpected", "selectionIntensity"]
  },

  selectionIntensity: {
    id: "selectionIntensity",
    name: "Selection Intensity",
    latex: "I = \\frac{\\bar{f'} - \\bar{f}}{\\sigma_f}",
    latexDisplay: "$$I = \\frac{\\bar{f'} - \\bar{f}}{\\sigma_f}$$",
    variables: {
      "I": "Selection intensity",
      "f̄'": "Mean fitness after selection (selected parents)",
      "f̄": "Mean fitness before selection (whole population)",
      "σ_f": "Standard deviation of fitness before selection"
    },
    derivation: [
      "Step 1: Selection increases mean fitness from f̄ to f̄'",
      "Step 2: Absolute increase: Δf = f̄' - f̄",
      "Step 3: Normalize by standard deviation: I = Δf / σ_f",
      "Step 4: I measures selection in 'standard deviations' of improvement"
    ],
    plainEnglish: "Selection intensity measures how much selection improves the mean, in units of standard deviations. Higher I means stronger selection pressure.",
    whenToUse: "For comparing selection strength across different methods or populations, predicting convergence speed.",
    example: "I=1 means selection increases mean by one standard deviation. Truncation selection (top 10%) gives I≈1.76.",
    relatedFormulas: ["reproductionRate", "lossOfDiversity"]
  },

  lossOfDiversity: {
    id: "lossOfDiversity",
    name: "Loss of Diversity (Proportion Not Selected)",
    latex: "P_d = 1 - \\frac{C(f_z)}{N}",
    latexDisplay: "$$P_d = 1 - \\frac{C(f_z)}{N}$$",
    variables: {
      "P_d": "Proportion of population not selected",
      "C(f_z)": "Cumulative count up to fitness f_z",
      "N": "Population size",
      "f_z": "Fitness threshold (e.g., median)"
    },
    derivation: [
      "Step 1: C(f_z)/N is proportion of population with fitness ≤ f_z",
      "Step 2: 1 - C(f_z)/N is proportion above f_z",
      "Step 3: For tournament selection: P_d = [C(f_z)/N]^t",
      "Step 4: Larger t → more diversity loss"
    ],
    plainEnglish: "The fraction of the population that doesn't get selected at all. Higher selection pressure leads to more diversity loss.",
    whenToUse: "For analyzing premature convergence risk and choosing appropriate selection pressure.",
    example: "With binary tournament, if 50% have below-median fitness, P_d = 0.5² = 0.25 of population never selected.",
    relatedFormulas: ["selectionIntensity", "tournamentExpected"]
  },

  varianceTheorem: {
    id: "varianceTheorem",
    name: "Variance of Selection",
    latex: "Var(\\hat{s}(f_i)) = N \\cdot p(f_i) \\cdot (1 - p(f_i))",
    latexDisplay: "$$Var(\\hat{s}(f_i)) = N \\cdot p(f_i) \\cdot (1 - p(f_i))$$",
    variables: {
      "Var(ŝ(f_i))": "Variance in number selected with fitness f_i",
      "N": "Number of selections",
      "p(f_i)": "Probability of selecting an individual with fitness f_i"
    },
    derivation: [
      "Step 1: Selection follows binomial distribution",
      "Step 2: Each of N selections is independent",
      "Step 3: Each succeeds (selects f_i) with probability p(f_i)",
      "Step 4: Binomial variance: Var = n × p × (1-p)"
    ],
    plainEnglish: "The variance in how many individuals of a given fitness get selected. Roulette wheel has high variance; SUS has lower variance.",
    whenToUse: "For comparing selection methods (roulette vs SUS), understanding selection reliability.",
    example: "If p=0.3 and N=100: Var = 100×0.3×0.7 = 21, so std dev ≈ 4.6 individuals.",
    relatedFormulas: ["expectedCopiesFPS"]
  },

  oneMaxFitness: {
    id: "oneMaxFitness",
    name: "OneMax Fitness Function",
    latex: "f(x) = \\sum_{i=1}^{l} x_i",
    latexDisplay: "$$f(x) = \\sum_{i=1}^{l} x_i$$",
    variables: {
      "f(x)": "Fitness of chromosome x",
      "x_i": "Value of bit i (0 or 1)",
      "l": "Chromosome length"
    },
    derivation: [
      "Simply count the number of 1s in the binary string",
      "Maximum fitness = l (all 1s)",
      "Minimum fitness = 0 (all 0s)"
    ],
    plainEnglish: "The OneMax problem counts the number of 1-bits in a binary string. It's a simple test problem where the optimal solution is all 1s.",
    whenToUse: "As a benchmark for testing EA performance, understanding basic EA behavior.",
    example: "f(10110) = 3 (three 1s), f(11111) = 5 (optimal)",
    relatedFormulas: []
  },

  eightQueensFitness: {
    id: "eightQueensFitness",
    name: "8-Queens Fitness Function",
    latex: "f = 28 - \\text{(number of attacking pairs)}",
    latexDisplay: "$$f = 28 - \\text{(number of attacking pairs)}$$",
    variables: {
      "f": "Fitness (higher is better)",
      "28": "Maximum possible non-attacking pairs (C(8,2) = 28)"
    },
    derivation: [
      "Step 1: With 8 queens, there are C(8,2) = 28 pairs of queens",
      "Step 2: Each pair can either attack each other or not",
      "Step 3: Fitness = 28 - (number of pairs that attack)",
      "Step 4: Perfect solution has fitness 28 (no attacks)",
      "Step 5: Worst solution has all pairs attacking (fitness depends on configuration)"
    ],
    plainEnglish: "For the 8-Queens problem, fitness counts non-attacking queen pairs. Maximum 28 means no queen attacks another.",
    whenToUse: "When implementing the 8-Queens problem as an EA example.",
    example: "If 3 pairs of queens attack each other: f = 28 - 3 = 25",
    relatedFormulas: []
  }
};

export function getFormula(id: string): Formula | undefined {
  return FORMULAS[id];
}

export function searchFormulas(query: string): Formula[] {
  const lowerQuery = query.toLowerCase();
  return Object.values(FORMULAS).filter(f =>
    f.name.toLowerCase().includes(lowerQuery) ||
    f.plainEnglish.toLowerCase().includes(lowerQuery)
  );
}

export function getRelatedFormulas(id: string): Formula[] {
  const formula = FORMULAS[id];
  if (!formula?.relatedFormulas) return [];
  return formula.relatedFormulas
    .map(fId => FORMULAS[fId])
    .filter((f): f is Formula => f !== undefined);
}
