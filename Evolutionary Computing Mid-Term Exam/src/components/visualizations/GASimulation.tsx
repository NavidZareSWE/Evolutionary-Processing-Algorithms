import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Pause, RotateCcw, Settings, TrendingUp } from 'lucide-react';

interface Individual {
  genes: number[];
  fitness: number;
}

interface GenerationStats {
  generation: number;
  bestFitness: number;
  avgFitness: number;
  worstFitness: number;
}

export default function GASimulation() {
  const [population, setPopulation] = useState<Individual[]>([]);
  const [generation, setGeneration] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [stats, setStats] = useState<GenerationStats[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  
  // Parameters
  const [popSize, setPopSize] = useState(20);
  const [geneLength, setGeneLength] = useState(10);
  const [mutationRate, setMutationRate] = useState(0.1);
  const [crossoverRate, setCrossoverRate] = useState(0.7);
  const [tournamentSize, setTournamentSize] = useState(3);
  const [targetFitness, setTargetFitness] = useState(10);
  
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // OneMax fitness: count number of 1s
  const calculateFitness = useCallback((genes: number[]): number => {
    return genes.reduce((sum, gene) => sum + gene, 0);
  }, []);

  // Initialize population
  const initializePopulation = useCallback(() => {
    const newPop: Individual[] = [];
    for (let i = 0; i < popSize; i++) {
      const genes = Array.from({ length: geneLength }, () => Math.random() < 0.5 ? 1 : 0);
      newPop.push({ genes, fitness: calculateFitness(genes) });
    }
    setPopulation(newPop);
    setGeneration(0);
    setStats([]);
    setIsRunning(false);
  }, [popSize, geneLength, calculateFitness]);

  // Tournament selection
  const tournamentSelect = useCallback((pop: Individual[]): Individual => {
    let best: Individual | null = null;
    for (let i = 0; i < tournamentSize; i++) {
      const idx = Math.floor(Math.random() * pop.length);
      if (!best || pop[idx].fitness > best.fitness) {
        best = pop[idx];
      }
    }
    return best!;
  }, [tournamentSize]);

  // One-point crossover
  const crossover = useCallback((parent1: Individual, parent2: Individual): [number[], number[]] => {
    if (Math.random() > crossoverRate) {
      return [[...parent1.genes], [...parent2.genes]];
    }
    const point = Math.floor(Math.random() * (geneLength - 1)) + 1;
    const child1 = [...parent1.genes.slice(0, point), ...parent2.genes.slice(point)];
    const child2 = [...parent2.genes.slice(0, point), ...parent1.genes.slice(point)];
    return [child1, child2];
  }, [crossoverRate, geneLength]);

  // Bit-flip mutation
  const mutate = useCallback((genes: number[]): number[] => {
    return genes.map(gene => Math.random() < mutationRate ? 1 - gene : gene);
  }, [mutationRate]);

  // Run one generation
  const runGeneration = useCallback(() => {
    if (population.length === 0) return;

    const newPopulation: Individual[] = [];
    
    // Elitism: keep best individual
    const sorted = [...population].sort((a, b) => b.fitness - a.fitness);
    newPopulation.push({ ...sorted[0] });

    // Create rest of population
    while (newPopulation.length < popSize) {
      const parent1 = tournamentSelect(population);
      const parent2 = tournamentSelect(population);
      const [child1Genes, child2Genes] = crossover(parent1, parent2);
      
      const mutatedChild1 = mutate(child1Genes);
      const mutatedChild2 = mutate(child2Genes);
      
      newPopulation.push({ genes: mutatedChild1, fitness: calculateFitness(mutatedChild1) });
      if (newPopulation.length < popSize) {
        newPopulation.push({ genes: mutatedChild2, fitness: calculateFitness(mutatedChild2) });
      }
    }

    const fitnesses = newPopulation.map(ind => ind.fitness);
    const newStats: GenerationStats = {
      generation: generation + 1,
      bestFitness: Math.max(...fitnesses),
      avgFitness: fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length,
      worstFitness: Math.min(...fitnesses),
    };

    setPopulation(newPopulation);
    setGeneration(g => g + 1);
    setStats(s => [...s.slice(-49), newStats]);

    // Check if target reached
    if (newStats.bestFitness >= targetFitness) {
      setIsRunning(false);
    }
  }, [population, popSize, tournamentSelect, crossover, mutate, calculateFitness, generation, targetFitness]);

  // Auto-run effect
  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(runGeneration, 200);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, runGeneration]);

  // Initialize on mount
  useEffect(() => {
    initializePopulation();
  }, []);

  const bestIndividual = population.length > 0 
    ? population.reduce((best, ind) => ind.fitness > best.fitness ? ind : best, population[0])
    : null;

  return (
    <div className="visualization-container">
      <div className="viz-header">
        <h3>
          <TrendingUp size={20} />
          Genetic Algorithm Simulation (OneMax Problem)
        </h3>
        <p className="viz-description">
          Watch evolution in action! The GA tries to find a binary string of all 1s.
          Each cell represents a gene (green=1, red=0). The fitness equals the count of 1s.
        </p>
      </div>

      <div className="viz-controls">
        <button 
          className={`viz-btn ${isRunning ? 'active' : ''}`}
          onClick={() => setIsRunning(!isRunning)}
        >
          {isRunning ? <Pause size={16} /> : <Play size={16} />}
          {isRunning ? 'Pause' : 'Run'}
        </button>
        <button className="viz-btn" onClick={runGeneration} disabled={isRunning}>
          Step
        </button>
        <button className="viz-btn" onClick={initializePopulation}>
          <RotateCcw size={16} />
          Reset
        </button>
        <button 
          className={`viz-btn ${showSettings ? 'active' : ''}`}
          onClick={() => setShowSettings(!showSettings)}
        >
          <Settings size={16} />
          Settings
        </button>
      </div>

      {showSettings && (
        <div className="viz-settings">
          <div className="setting-row">
            <label>Population Size: {popSize}</label>
            <input 
              type="range" min="10" max="50" value={popSize}
              onChange={e => setPopSize(Number(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label>Gene Length: {geneLength}</label>
            <input 
              type="range" min="5" max="20" value={geneLength}
              onChange={e => setGeneLength(Number(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label>Mutation Rate: {(mutationRate * 100).toFixed(0)}%</label>
            <input 
              type="range" min="0" max="50" value={mutationRate * 100}
              onChange={e => setMutationRate(Number(e.target.value) / 100)}
            />
          </div>
          <div className="setting-row">
            <label>Crossover Rate: {(crossoverRate * 100).toFixed(0)}%</label>
            <input 
              type="range" min="0" max="100" value={crossoverRate * 100}
              onChange={e => setCrossoverRate(Number(e.target.value) / 100)}
            />
          </div>
          <div className="setting-row">
            <label>Tournament Size: {tournamentSize}</label>
            <input 
              type="range" min="2" max="7" value={tournamentSize}
              onChange={e => setTournamentSize(Number(e.target.value))}
            />
          </div>
          <button className="viz-btn primary" onClick={initializePopulation}>
            Apply & Reset
          </button>
        </div>
      )}

      <div className="viz-stats">
        <div className="stat-box">
          <span className="stat-label">Generation</span>
          <span className="stat-value">{generation}</span>
        </div>
        <div className="stat-box">
          <span className="stat-label">Best Fitness</span>
          <span className="stat-value highlight">{bestIndividual?.fitness || 0} / {geneLength}</span>
        </div>
        <div className="stat-box">
          <span className="stat-label">Avg Fitness</span>
          <span className="stat-value">
            {population.length > 0 
              ? (population.reduce((sum, ind) => sum + ind.fitness, 0) / population.length).toFixed(1)
              : 0}
          </span>
        </div>
        <div className="stat-box">
          <span className="stat-label">Target</span>
          <span className="stat-value">{geneLength} (all 1s)</span>
        </div>
      </div>

      <div className="population-grid">
        <h4>Population (sorted by fitness)</h4>
        <div className="individuals">
          {[...population]
            .sort((a, b) => b.fitness - a.fitness)
            .slice(0, 15)
            .map((ind, i) => (
              <div key={i} className={`individual ${i === 0 ? 'best' : ''}`}>
                <span className="ind-rank">#{i + 1}</span>
                <div className="genes">
                  {ind.genes.map((gene, j) => (
                    <span key={j} className={`gene ${gene === 1 ? 'on' : 'off'}`} />
                  ))}
                </div>
                <span className="ind-fitness">f={ind.fitness}</span>
              </div>
            ))}
        </div>
      </div>

      {stats.length > 0 && (
        <div className="fitness-chart">
          <h4>Fitness Over Generations</h4>
          <div className="chart-container">
            <svg viewBox="0 0 400 150" className="chart-svg">
              {/* Grid lines */}
              {[0, 25, 50, 75, 100].map(y => (
                <line key={y} x1="40" y1={130 - y * 1.2} x2="390" y2={130 - y * 1.2} 
                  stroke="var(--surface-700)" strokeWidth="1" strokeDasharray="2,2" />
              ))}
              
              {/* Best fitness line */}
              <polyline
                fill="none"
                stroke="var(--success-400)"
                strokeWidth="2"
                points={stats.map((s, i) => 
                  `${40 + (i * 350 / Math.max(stats.length - 1, 1))},${130 - (s.bestFitness / geneLength) * 120}`
                ).join(' ')}
              />
              
              {/* Average fitness line */}
              <polyline
                fill="none"
                stroke="var(--primary-400)"
                strokeWidth="2"
                points={stats.map((s, i) => 
                  `${40 + (i * 350 / Math.max(stats.length - 1, 1))},${130 - (s.avgFitness / geneLength) * 120}`
                ).join(' ')}
              />
              
              {/* Axis labels */}
              <text x="20" y="15" fill="var(--surface-300)" fontSize="10">100%</text>
              <text x="20" y="130" fill="var(--surface-300)" fontSize="10">0%</text>
              <text x="200" y="148" fill="var(--surface-300)" fontSize="10" textAnchor="middle">Generation</text>
            </svg>
            <div className="chart-legend">
              <span className="legend-item"><span className="legend-color best"></span> Best</span>
              <span className="legend-item"><span className="legend-color avg"></span> Average</span>
            </div>
          </div>
        </div>
      )}

      {bestIndividual?.fitness === geneLength && (
        <div className="success-message">
          🎉 Solution found in {generation} generations! The GA discovered the optimal string of all 1s.
        </div>
      )}
    </div>
  );
}
