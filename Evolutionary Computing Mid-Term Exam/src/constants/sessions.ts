export interface Session {
  id: number;
  title: string;
  slideRange: [number, number];
  summary: string;
  keyTakeaway: string;
  topics: string[];
}

export const SESSIONS: Session[] = [
  {
    id: 1,
    title: "Introduction to Intelligence and AI",
    slideRange: [1, 10],
    summary: "Establishes the foundational concepts by defining intelligence as the capability to adapt behavior to changing environments. Introduces Russell's four approaches to AI (Think/Act × Humanly/Rationally) and positions AI as fundamentally about optimization. Distinguishes between human-created and nature-inspired optimization approaches.",
    keyTakeaway: "Intelligence = Adaptive Capability. Acting rationally = Optimization. This is why evolutionary approaches work for AI.",
    topics: [
      "Definition of Intelligence",
      "Russell's Four AI Approaches",
      "Intelligent Agents",
      "AI as Optimization",
      "Human-Created vs Nature-Inspired Approaches"
    ]
  },
  {
    id: 2,
    title: "Bio-Inspired Computing & Natural Selection",
    slideRange: [11, 19],
    summary: "Bridges biology and computer science by explaining bio-inspired computing as a two-way relationship. Defines evolutionary computation as computational models of natural selection and genetics. Introduces the three fundamental operators (crossover, mutation, selection) and distinguishes between evolution (process) and adaptation (outcome).",
    keyTakeaway: "Three operators drive ALL evolution: Crossover (combines), Mutation (creates), Selection (guides). Remove any one and evolution stops.",
    topics: [
      "Bio-Inspired Computing Definition",
      "Evolutionary Computation",
      "Natural Selection Mechanism",
      "Three Fundamental Operators",
      "Evolution vs Adaptation",
      "Why Human Evolution Stopped"
    ]
  },
  {
    id: 3,
    title: "Four EA Brands & General Framework",
    slideRange: [20, 31],
    summary: "Introduces the four main EA families (GA, ES, GP, EP) with their historical origins and characteristics. Establishes the universal EA framework: Initialize → [Parent Selection → Variation → Survivor Selection] → Terminate. Explains the critical phenotype/genotype distinction and the evolution-problem solving mapping.",
    keyTakeaway: "All EAs share the same structure despite different origins. Choose representation for your problem, operators for your representation.",
    topics: [
      "Genetic Algorithms (GA)",
      "Evolution Strategies (ES)",
      "Genetic Programming (GP)",
      "Evolutionary Programming (EP)",
      "General EA Scheme",
      "Phenotype vs Genotype",
      "Evolution-Problem Solving Mapping"
    ]
  },
  {
    id: 4,
    title: "Fitness Functions & Population Concepts",
    slideRange: [32, 47],
    summary: "Defines the fitness function as the problem specification that populations adapt to. Explains the diversity hierarchy: Fitness Diversity → Phenotype Diversity → Genotype Diversity. Discusses parent vs survivor selection roles and the critical mutation vs crossover debate (crossover explores, mutation exploits).",
    keyTakeaway: "We only know CORRECT fitness (ranking), not EXACT fitness (values). This distinction is critical for selection method choice.",
    topics: [
      "Fitness Function Definition",
      "Quality and Discrimination",
      "Population Diversity Hierarchy",
      "Parent Selection Purpose",
      "Survivor Selection Purpose",
      "Mutation vs Crossover Roles"
    ]
  },
  {
    id: 5,
    title: "8-Queens Problem Complete Example",
    slideRange: [48, 55],
    summary: "Comprehensive walkthrough of solving the 8-Queens problem using EA. Demonstrates permutation representation, fitness function design (counting non-attacking pairs), and the complete evolutionary cycle. Shows how EA concepts apply to a concrete combinatorial optimization problem.",
    keyTakeaway: "Permutation representation naturally enforces the one-queen-per-column constraint, reducing search space from 64^8 to 8! possibilities.",
    topics: [
      "8-Queens Problem Definition",
      "Permutation Representation",
      "Fitness Function Design",
      "Crossover for Permutations",
      "Mutation for Permutations",
      "Complete EA Cycle Example"
    ]
  },
  {
    id: 6,
    title: "Simple GA & Binary Representations",
    slideRange: [56, 70],
    summary: "Introduces Holland's Simple Genetic Algorithm (SGA) with binary representation. Covers fitness-proportionate selection (roulette wheel), one-point crossover, and bit-flip mutation. Explains Goldberg's x² example and introduces Gray coding to address Hamming cliff problems.",
    keyTakeaway: "SGA uses binary strings with crossover emphasis. Gray coding fixes Hamming cliff problems by ensuring adjacent values differ by one bit.",
    topics: [
      "Simple Genetic Algorithm (SGA)",
      "Binary Representation",
      "Roulette Wheel Selection",
      "One-Point Crossover",
      "Bit-Flip Mutation",
      "Goldberg's x² Example",
      "Gray Coding"
    ]
  },
  {
    id: 7,
    title: "Permutation Representations & Crossover Operators",
    slideRange: [71, 99],
    summary: "Deep dive into permutation representations for ordering problems (TSP) and adjacency problems. Introduces major crossover operators: Order-1 (preserves relative order), PMX (preserves absolute positions), Cycle (preserves positions), Edge (preserves adjacencies). Explains positional and distributional bias concepts.",
    keyTakeaway: "Different crossover operators preserve different properties. Match operator to what matters: Order-1 for relative order, Edge for adjacencies.",
    topics: [
      "Permutation Problems",
      "Order vs Adjacency Problems",
      "Order-1 Crossover",
      "PMX Crossover",
      "Cycle Crossover",
      "Edge Crossover",
      "Positional Bias",
      "Distributional Bias"
    ]
  },
  {
    id: 8,
    title: "Mutation Operators & Crossover-Mutation Debate",
    slideRange: [100, 114],
    summary: "Covers mutation operators for permutations: Insert, Swap, Inversion, Scramble. Resolves the crossover vs mutation debate: crossover is EXPLORATIVE (large steps, combines existing), mutation is EXPLOITATIVE (small steps, creates new). Only mutation can introduce new genetic material.",
    keyTakeaway: "Crossover EXPLORES (large jumps, combines), Mutation EXPLOITS (small steps, creates new). EA without mutation cannot introduce new alleles.",
    topics: [
      "Insert Mutation",
      "Swap Mutation",
      "Inversion Mutation",
      "Scramble Mutation",
      "Crossover vs Mutation Debate",
      "Exploration vs Exploitation",
      "One-Max Example"
    ]
  },
  {
    id: 9,
    title: "Multi-Parent Recombination & Population Models",
    slideRange: [115, 131],
    summary: "Extends recombination beyond two parents with scanning techniques (U-Scan, OB-Scan, FB-Scan). Compares generational vs steady-state population models. Explains when multi-parent recombination helps (same basin) vs hurts (different basins).",
    keyTakeaway: "Multi-parent helps when good solutions are in the SAME basin of attraction. Hurts when in DIFFERENT basins—child ends up between optima.",
    topics: [
      "Multi-Parent Recombination",
      "U-Scan Crossover",
      "Occurrence-Based Scan",
      "Fitness-Based Scan",
      "Generational Model",
      "Steady-State Model",
      "Generation Gap"
    ]
  },
  {
    id: 10,
    title: "Selection Mechanisms In-Depth",
    slideRange: [132, 155],
    summary: "Comprehensive coverage of selection mechanisms. Exposes FPS problems: premature convergence, lost selection pressure, scaling sensitivity. Introduces rank-based selection and tournament selection as robust alternatives. Tournament selection only needs comparisons, not exact fitness values.",
    keyTakeaway: "Tournament selection addresses ALL FPS problems: no sorting, no scaling sensitivity, controllable pressure via single parameter k.",
    topics: [
      "Fitness-Proportionate Selection Problems",
      "Premature Convergence",
      "Lost Selection Pressure",
      "Scaling Sensitivity",
      "Rank-Based Selection",
      "Tournament Selection",
      "Stochastic Universal Sampling",
      "Selection Pressure Control"
    ]
  },
  {
    id: 11,
    title: "Schema Theorem & Building Block Hypothesis",
    slideRange: [156, 177],
    summary: "Introduces Holland's Schema Theorem—the foundational theoretical result in GA. Defines schemata (templates over {0,1,#}), order, and defining length. Explains how selection favors above-average schemata while crossover/mutation survival depends on defining length and order. Presents the Building Block Hypothesis and its criticisms.",
    keyTakeaway: "Schema Theorem: short, low-order, above-average schemata grow exponentially. But it's only a lower bound and has significant limitations.",
    topics: [
      "Schema Definition",
      "Order and Defining Length",
      "Schema Theorem",
      "Survival Probabilities",
      "Building Block Hypothesis",
      "Implicit Parallelism",
      "Schema Theorem Limitations"
    ]
  },
  {
    id: 12,
    title: "Royal Road Functions & GA Performance",
    slideRange: [178, 201],
    summary: "Examines Royal Road functions designed to be GA-easy based on Building Block Hypothesis. Reveals surprising results: removing intermediate levels IMPROVED performance. Analyzes why—intermediate levels destroy GA's parallelism by dictating solution paths. More domain knowledge can hurt GA performance.",
    keyTakeaway: "Royal Road paradox: Functions designed to be GA-easy weren't. Intermediate stepping stones hurt by imposing sequential structure on parallel search.",
    topics: [
      "Royal Road Functions",
      "Mitchell's Experiments",
      "Unexpected Results",
      "Hitchhiking Problem",
      "Intermediate Stepping Stones",
      "GA Parallelism",
      "Tanese Functions"
    ]
  },
  {
    id: 13,
    title: "GA Difficulty & Deception",
    slideRange: [202, 216],
    summary: "Identifies four reasons for poor GA performance: 1) Representation doesn't match, 2) Fitness function misleads, 3) Operators don't fit representation, 4) Parameters wrong. Introduces deception—when lower-order schemata lead away from global optimum. Discusses fitness-distance correlation as practical difficulty measure.",
    keyTakeaway: "Deception: Lower-order schemata mislead search away from global optimum. But deception alone doesn't fully predict GA difficulty.",
    topics: [
      "Four Reasons for GA Failure",
      "Representation Problems",
      "Deceptive Fitness Functions",
      "Fully Deceptive Problems",
      "Partially Deceptive Problems",
      "Fitness-Distance Correlation",
      "Problem Difficulty Prediction"
    ]
  },
  {
    id: 14,
    title: "Selection Analysis & Mathematical Foundations",
    slideRange: [217, 248],
    summary: "Mathematical analysis of selection mechanisms. Derives expected copies formula for FPS. Proves tournament selection properties including the concatenation theorem. Analyzes reproduction rate, selection intensity, and diversity loss. Provides complete theoretical foundation for selection mechanism design.",
    keyTakeaway: "Tournament selection theorem: ŝ(fᵢ) = N × [(C(fᵢ)/N)^t - (C(fᵢ₋₁)/N)^t]. Selection pressure controlled by single parameter t.",
    topics: [
      "Expected Copies Formula",
      "Variance Analysis",
      "Tournament Selection Theorem",
      "Concatenation Theorem",
      "Reproduction Rate",
      "Selection Intensity",
      "Diversity Loss",
      "Takeover Time"
    ]
  }
];

export function getSessionById(id: number): Session | undefined {
  return SESSIONS.find(s => s.id === id);
}

export function getSessionBySlide(slideNumber: number): Session | undefined {
  return SESSIONS.find(s => 
    slideNumber >= s.slideRange[0] && slideNumber <= s.slideRange[1]
  );
}
