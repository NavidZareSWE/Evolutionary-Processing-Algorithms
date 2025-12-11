export interface Definition {
  term: string;
  definition: string;
  professorEmphasis?: string;
  relatedTerms?: string[];
  examples?: string[];
  commonMisconceptions?: string[];
}

export const DEFINITIONS: Record<string, Definition> = {
  intelligence: {
    term: "Intelligence",
    definition:
      "The capability of a system to adapt its behaviour to ever-changing environment.",
    professorEmphasis:
      "Intelligence is a CAPABILITY (potential), not a demonstrated state. It's about having the right MECHANISMS for adaptation, not about current performance.",
    relatedTerms: ["adaptation", "rational agent", "optimization"],
    examples: [
      "A thermostat adapting to temperature changes",
      "An EA adapting to fitness landscape",
      "A human learning new skills",
    ],
    commonMisconceptions: [
      "Intelligence requires consciousness",
      "Intelligence is about knowledge storage",
      "Only biological entities can be intelligent",
    ],
  },

  intelligentAgent: {
    term: "Intelligent Agent",
    definition:
      "An autonomous entity which observes through sensors and acts upon an environment using actuators, and directs its activity towards achieving goals (i.e., it is rational).",
    professorEmphasis:
      "Complexity ranges from thermostat to human to communities. An EA population is like a community of solutions working toward optimization.",
    relatedTerms: ["autonomy", "sensors", "actuators", "rationality"],
    examples: [
      "Thermostat: Senses temperature, acts by turning heater on/off, goal is target temperature",
      "EA: Senses fitness, acts by generating solutions, goal is optimization",
      "Robot: Senses environment, acts through motors, goal varies by task",
    ],
  },

  fitness: {
    term: "Fitness",
    definition:
      "A measure of how well-adapted an individual is to its environment, determining its likelihood of survival and reproduction.",
    professorEmphasis:
      "CRITICAL DISTINCTION: We only know CORRECT fitness (relative ordering/ranking), NOT EXACT fitness (precise numerical values). This is why FPS is problematic and tournament selection is robust.",
    relatedTerms: ["fitness function", "selection pressure", "quality"],
    examples: [
      "In TSP: fitness = 1/tour_length (shorter tours are fitter)",
      "In classification: fitness = accuracy on test set",
      "In scheduling: fitness = makespan efficiency",
    ],
    commonMisconceptions: [
      "Fitness values are absolute measures",
      "Higher fitness numbers always mean better solutions",
      "We can always know exact fitness values",
    ],
  },

  genotype: {
    term: "Genotype",
    definition:
      "The encoded representation of a solution; the internal data structure that evolutionary operators work on.",
    professorEmphasis:
      "The genotype is what we MANIPULATE (apply crossover, mutation to). It may or may not resemble the actual solution.",
    relatedTerms: ["phenotype", "chromosome", "encoding", "representation"],
    examples: [
      "Binary string: '10110100'",
      "Real vector: [0.5, 1.2, -0.3, 0.8]",
      "Permutation: [3, 1, 4, 2, 5]",
      "Tree: (+ (* x x) x)",
    ],
  },

  phenotype: {
    term: "Phenotype",
    definition:
      "The decoded representation of a solution; the actual solution in the problem space that gets evaluated for fitness.",
    professorEmphasis:
      "The phenotype is what we EVALUATE. The fitness function operates on phenotypes, not genotypes.",
    relatedTerms: ["genotype", "decoding", "solution space"],
    examples: [
      "For binary-encoded real number: the actual real value",
      "For permutation in TSP: the actual tour",
      "For tree in GP: the function/program it computes",
    ],
  },

  crossover: {
    term: "Crossover (Recombination)",
    definition:
      "A variation operator that combines genetic material from two (or more) parent solutions to create offspring.",
    professorEmphasis:
      "Crossover COMBINES existing genetic material but CANNOT create new alleles. It is EXPLORATIVE—takes large steps in search space.",
    relatedTerms: ["recombination", "variation operator", "parent selection"],
    examples: [
      "1-point crossover: Parents AB|CD and EF|GH → Children AB|GH and EF|CD",
      "Uniform crossover: Randomly select each gene from either parent",
      "PMX: Preserve absolute positions from one parent, fill rest from other",
    ],
    commonMisconceptions: [
      "Crossover creates new genetic material (it doesn't)",
      "Crossover is always beneficial (it can disrupt good solutions)",
      "Higher crossover rate is always better",
    ],
  },

  mutation: {
    term: "Mutation",
    definition:
      "A variation operator that makes random modifications to a single solution, introducing variation not present in the current population.",
    professorEmphasis:
      "ONLY mutation can introduce new genetic material. Without mutation, alleles lost from population are gone forever. Mutation is EXPLOITATIVE—takes small steps.",
    relatedTerms: [
      "variation operator",
      "diversity maintenance",
      "exploration",
    ],
    examples: [
      "Bit-flip: Change 0 to 1 or 1 to 0",
      "Gaussian: Add random value from N(0,σ)",
      "Swap: Exchange positions of two genes",
      "Insert: Move one gene to new position",
    ],
    commonMisconceptions: [
      "Mutation is optional",
      "Mutation should have very low probability",
      "Mutation is purely random noise",
    ],
  },

  selection: {
    term: "Selection",
    definition:
      "The process of choosing which individuals will become parents (parent selection) or survive to the next generation (survivor selection), typically based on fitness.",
    professorEmphasis:
      "Selection provides DIRECTION to evolution but does NOT create new genetic material. Without selection, evolution becomes random search.",
    relatedTerms: [
      "selection pressure",
      "tournament selection",
      "fitness-proportionate selection",
    ],
    examples: [
      "Tournament: Pick k random, select best",
      "Roulette wheel: Probability proportional to fitness",
      "Truncation: Select top fraction of population",
    ],
  },

  schema: {
    term: "Schema",
    definition:
      "A template over the alphabet {0, 1, #} where # is a wildcard ('don't care') that matches either 0 or 1. Represents a hyperplane in the solution space.",
    professorEmphasis:
      "Each chromosome matches 2^l schemata. N chromosomes effectively process O(N³) schemata—this is implicit parallelism.",
    relatedTerms: ["order", "defining length", "building block", "hyperplane"],
    examples: [
      "1##0#: Matches 10000, 10001, 10100, 10101, 11000, 11001, 11100, 11101",
      "####: Matches all chromosomes (wildcard everywhere)",
      "1010: Matches only 1010 (no wildcards)",
    ],
  },

  order: {
    term: "Order (of a schema)",
    definition:
      "The number of defined (non-wildcard) positions in a schema. Denoted o(H).",
    professorEmphasis:
      "Low-order schemata survive mutation better. Mutation survival probability is (1-p_m)^o(H).",
    relatedTerms: ["schema", "defining length", "mutation survival"],
    examples: [
      "o(1##0#) = 2 (two defined positions: 1 and 0)",
      "o(#####) = 0 (no defined positions)",
      "o(10101) = 5 (all positions defined)",
    ],
  },

  definingLength: {
    term: "Defining Length",
    definition:
      "The distance between the first and last defined positions in a schema. Denoted δ(H).",
    professorEmphasis:
      "Short defining length survives crossover better. Crossover survival probability is 1 - p_c × δ(H)/(l-1).",
    relatedTerms: ["schema", "order", "crossover survival"],
    examples: [
      "δ(1###0) = 4 (positions 1 and 5, distance = 4)",
      "δ(##1#0##) = 2 (positions 3 and 5, distance = 2)",
      "δ(#1#####) = 0 (only one defined position)",
    ],
  },

  buildingBlock: {
    term: "Building Block",
    definition:
      "A short, low-order schema with above-average fitness that can combine with other building blocks to form high-quality solutions.",
    professorEmphasis:
      "The Building Block Hypothesis: GA works by discovering and combining building blocks. However, this doesn't always hold—Royal Road experiments showed counter-examples.",
    relatedTerms: ["schema theorem", "implicit parallelism", "deception"],
    examples: [
      "In OneMax: Schema 1#### has above-average fitness",
      "In TSP: Keeping certain city pairs adjacent might be a building block",
    ],
    commonMisconceptions: [
      "Building blocks always combine well",
      "All problems have useful building blocks",
      "GA always finds and preserves building blocks",
    ],
  },

  tournamentSelection: {
    term: "Tournament Selection",
    definition:
      "A selection method where k individuals are randomly chosen from the population and the best one (highest fitness) is selected.",
    professorEmphasis:
      "Tournament selection only needs COMPARISONS (correct fitness), not exact values. It's robust to scaling, requires no sorting, and selection pressure is controlled by single parameter k.",
    relatedTerms: [
      "selection pressure",
      "fitness-proportionate selection",
      "rank-based selection",
    ],
    examples: [
      "Binary tournament (k=2): Pick 2, select better one",
      "k=1: Random selection (no pressure)",
      "k=N: Always select best (maximum pressure)",
    ],
  },

  fitnessProportionateSelection: {
    term: "Fitness-Proportionate Selection (FPS)",
    definition:
      "A selection method where the probability of selecting an individual is proportional to its fitness value. Also called roulette wheel selection.",
    professorEmphasis:
      "FPS uses EXACT fitness values, but we only have CORRECT fitness. This causes problems: scaling sensitivity, premature convergence, lost selection pressure.",
    relatedTerms: [
      "roulette wheel",
      "selection pressure",
      "tournament selection",
    ],
    examples: [
      "Fitness values 10, 20, 30 → Probabilities 1/6, 2/6, 3/6",
      "Adding 100 to all: 110, 120, 130 → Probabilities change dramatically!",
    ],
    commonMisconceptions: [
      "FPS is the best selection method",
      "Fitness values don't need to be positive for FPS",
      "FPS maintains constant selection pressure",
    ],
  },

  positionalBias: {
    term: "Positional Bias",
    definition:
      "The tendency of a crossover operator to keep certain gene positions together based on their location in the chromosome.",
    professorEmphasis:
      "1-point crossover has strong positional bias—genes near chromosome ends can NEVER be kept together. This can be exploited if problem structure is known.",
    relatedTerms: ["crossover", "uniform crossover", "linkage"],
    examples: [
      "1-point at position 3: AB|CD always keeps A with B, C with D",
      "Can never keep A with D through 1-point crossover",
    ],
  },

  distributionalBias: {
    term: "Distributional Bias",
    definition:
      "The tendency of a crossover operator to produce offspring with a certain distribution of genetic material from each parent.",
    professorEmphasis:
      "Uniform crossover tends toward 50% from each parent. n-point crossover may bias toward more from one parent depending on chromosome structure.",
    relatedTerms: ["crossover", "positional bias", "uniform crossover"],
    examples: [
      "Uniform crossover: Expected 50% from each parent",
      "1-point crossover: Depends on crossover point position",
    ],
  },

  deception: {
    term: "Deception",
    definition:
      "A condition where lower-order schemata (building blocks) lead the search in the opposite direction from the global optimum.",
    professorEmphasis:
      "In deceptive problems, combining good building blocks leads AWAY from the optimum. However, deception alone doesn't fully predict GA difficulty.",
    relatedTerms: ["building block", "fitness landscape", "GA difficulty"],
    examples: [
      "Trap functions where local optimum has better building blocks than global",
      "Problems where schema fitness increases as you move away from optimum",
    ],
  },

  diversityHierarchy: {
    term: "Diversity Hierarchy",
    definition:
      "The relationship between different types of diversity: Fitness Diversity → Phenotype Diversity → Genotype Diversity. Each implies the next but NOT the reverse.",
    professorEmphasis:
      "Different genotypes can encode the same phenotype (no phenotype diversity despite genotype diversity). Different phenotypes can have same fitness (no fitness diversity despite phenotype diversity).",
    relatedTerms: ["genotype", "phenotype", "fitness", "population diversity"],
    examples: [
      "Gray code: Different bit strings can encode same number",
      "TSP: Different city orderings can have same tour length",
    ],
  },

  implicitParallelism: {
    term: "Implicit Parallelism",
    definition:
      "The phenomenon where a population of N chromosomes effectively processes O(N³) schemata simultaneously, without explicitly representing or evaluating them.",
    professorEmphasis:
      "Each chromosome matches 2^l schemata. The population 'samples' many schemata at once—this is a key theoretical advantage of GA.",
    relatedTerms: ["schema", "building block hypothesis", "population"],
    examples: [
      "Population of 100 chromosomes of length 20 processes millions of schemata",
    ],
  },

  selectionPressure: {
    term: "Selection Pressure",
    definition:
      "The degree to which selection favors better individuals over worse ones. Higher pressure means greater advantage for fitter individuals.",
    professorEmphasis:
      "Too much pressure: Premature convergence. Too little pressure: Random search. Tournament size k directly controls pressure.",
    relatedTerms: ["selection", "convergence", "diversity"],
    examples: [
      "Tournament k=2: Mild pressure",
      "Tournament k=10: Strong pressure",
      "Truncation 10%: Very strong pressure",
    ],
  },

  prematureConvergence: {
    term: "Premature Convergence",
    definition:
      "When a population converges to a suboptimal solution because selection pressure is too high, diversity is lost too quickly, or the population gets trapped in local optima.",
    professorEmphasis:
      "Premature convergence is a major failure mode. Signs include: low diversity, fitness stagnation, all individuals similar.",
    relatedTerms: ["diversity", "selection pressure", "local optima"],
    examples: [
      "One super-fit individual takes over population via FPS",
      "Tournament size too large causes rapid diversity loss",
    ],
  },
};

export function getDefinition(key: string): Definition | undefined {
  return DEFINITIONS[key];
}

export function searchDefinitions(query: string): Definition[] {
  const lowerQuery = query.toLowerCase();
  return Object.values(DEFINITIONS).filter(
    (def) =>
      def.term.toLowerCase().includes(lowerQuery) ||
      def.definition.toLowerCase().includes(lowerQuery),
  );
}

export function getRelatedDefinitions(key: string): Definition[] {
  const def = DEFINITIONS[key];
  if (!def?.relatedTerms) return [];

  return def.relatedTerms
    .map((term) => {
      const normalizedKey = term.replace(/\s+/g, "").toLowerCase();
      return Object.entries(DEFINITIONS).find(
        ([k, d]) =>
          k.toLowerCase() === normalizedKey ||
          d.term.toLowerCase() === term.toLowerCase(),
      )?.[1];
    })
    .filter((d): d is Definition => d !== undefined);
}
