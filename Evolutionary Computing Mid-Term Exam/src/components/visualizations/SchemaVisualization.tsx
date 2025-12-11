import React, { useState, useMemo } from 'react';
import { Layers, Calculator, Info } from 'lucide-react';

interface SchemaAnalysis {
  schema: string;
  order: number;
  definingLength: number;
  matchingCount: number;
  avgFitness: number;
  selectionFactor: number;
  crossoverSurvival: number;
  mutationSurvival: number;
  expectedGrowth: number;
}

export default function SchemaVisualization() {
  const [chromosomeLength] = useState(8);
  const [schema, setSchema] = useState('1##0####');
  const [populationFitness] = useState([5, 7, 3, 8, 6, 4, 9, 2]); // Example fitnesses
  const [crossoverRate, setCrossoverRate] = useState(0.7);
  const [mutationRate, setMutationRate] = useState(0.01);

  // Generate all possible chromosomes for this schema
  const matchingChromosomes = useMemo(() => {
    const matches: string[] = [];
    const generateMatches = (current: string, pos: number) => {
      if (pos === schema.length) {
        matches.push(current);
        return;
      }
      if (schema[pos] === '#') {
        generateMatches(current + '0', pos + 1);
        generateMatches(current + '1', pos + 1);
      } else {
        generateMatches(current + schema[pos], pos + 1);
      }
    };
    generateMatches('', 0);
    return matches;
  }, [schema]);

  // Calculate schema properties
  const analysis = useMemo((): SchemaAnalysis => {
    // Order: number of fixed positions
    const order = schema.split('').filter(c => c !== '#').length;
    
    // Defining length: distance between first and last fixed position
    const fixedPositions = schema.split('').map((c, i) => c !== '#' ? i : -1).filter(i => i >= 0);
    const definingLength = fixedPositions.length > 1 
      ? fixedPositions[fixedPositions.length - 1] - fixedPositions[0]
      : 0;
    
    // Matching count (for demonstration, we'll assume 50% match)
    const matchingCount = Math.pow(2, chromosomeLength - order);
    
    // Average fitness (simplified - assume schema members have above-average fitness)
    const avgPopFitness = populationFitness.reduce((a, b) => a + b, 0) / populationFitness.length;
    const schemaFitness = avgPopFitness * 1.2; // Assume schema is above average
    
    // Selection factor
    const selectionFactor = schemaFitness / avgPopFitness;
    
    // Crossover survival probability
    const crossoverSurvival = 1 - crossoverRate * definingLength / (chromosomeLength - 1);
    
    // Mutation survival probability
    const mutationSurvival = Math.pow(1 - mutationRate, order);
    
    // Expected growth (Schema Theorem lower bound)
    const expectedGrowth = selectionFactor * crossoverSurvival * mutationSurvival;
    
    return {
      schema,
      order,
      definingLength,
      matchingCount,
      avgFitness: schemaFitness,
      selectionFactor,
      crossoverSurvival,
      mutationSurvival,
      expectedGrowth
    };
  }, [schema, chromosomeLength, crossoverRate, mutationRate, populationFitness]);

  const avgPopFitness = populationFitness.reduce((a, b) => a + b, 0) / populationFitness.length;

  return (
    <div className="visualization-container">
      <div className="viz-header">
        <h3>
          <Layers size={20} />
          Schema Theorem Visualization
        </h3>
        <p className="viz-description">
          Explore how schemata (building blocks) propagate through generations.
          The theorem predicts that short, low-order, above-average schemata grow exponentially.
        </p>
      </div>

      <div className="schema-input-section">
        <label className="schema-label">
          <span>Enter Schema (use # for wildcard):</span>
          <input
            type="text"
            value={schema}
            onChange={e => setSchema(e.target.value.replace(/[^01#]/g, '').slice(0, chromosomeLength))}
            className="schema-input"
            maxLength={chromosomeLength}
            placeholder="1##0####"
          />
        </label>
        <div className="schema-examples">
          <span>Examples:</span>
          <button onClick={() => setSchema('1#######')}>1#######</button>
          <button onClick={() => setSchema('1##0####')}>1##0####</button>
          <button onClick={() => setSchema('1######1')}>1######1</button>
          <button onClick={() => setSchema('11110000')}>11110000</button>
        </div>
      </div>

      <div className="schema-display">
        <h4>Schema Representation</h4>
        <div className="schema-visual">
          {schema.split('').map((char, i) => (
            <span key={i} className={`schema-char ${char === '#' ? 'wildcard' : 'fixed'}`}>
              {char}
            </span>
          ))}
        </div>
        <p className="schema-explanation">
          This schema matches <strong>{analysis.matchingCount}</strong> possible chromosomes
          ({matchingChromosomes.slice(0, 4).join(', ')}{matchingChromosomes.length > 4 ? '...' : ''})
        </p>
      </div>

      <div className="viz-settings inline">
        <div className="setting-row">
          <label>Crossover Rate (p_c): {(crossoverRate * 100).toFixed(0)}%</label>
          <input 
            type="range" min="0" max="100" value={crossoverRate * 100}
            onChange={e => setCrossoverRate(Number(e.target.value) / 100)}
          />
        </div>
        <div className="setting-row">
          <label>Mutation Rate (p_m): {(mutationRate * 100).toFixed(1)}%</label>
          <input 
            type="range" min="0" max="20" step="0.5" value={mutationRate * 100}
            onChange={e => setMutationRate(Number(e.target.value) / 100)}
          />
        </div>
      </div>

      <div className="schema-properties">
        <h4>Schema Properties</h4>
        <div className="properties-grid">
          <div className="property-card">
            <span className="property-name">Order o(H)</span>
            <span className="property-value">{analysis.order}</span>
            <span className="property-desc">Fixed positions (non-# chars)</span>
          </div>
          <div className="property-card">
            <span className="property-name">Defining Length δ(H)</span>
            <span className="property-value">{analysis.definingLength}</span>
            <span className="property-desc">Distance: first to last fixed</span>
          </div>
          <div className="property-card">
            <span className="property-name">Matching Chromosomes</span>
            <span className="property-value">2^{chromosomeLength - analysis.order} = {analysis.matchingCount}</span>
            <span className="property-desc">Chromosomes in this hyperplane</span>
          </div>
        </div>
      </div>

      <div className="schema-theorem-calc">
        <h4>
          <Calculator size={18} />
          Schema Theorem Calculation
        </h4>
        
        <div className="theorem-formula">
          <div className="formula-line">
            m(H, t+1) ≥ m(H, t) × <span className="factor selection">[f(H)/⟨f⟩]</span> × 
            <span className="factor crossover">[1 - p_c × δ(H)/(l-1)]</span> × 
            <span className="factor mutation">[(1-p_m)^o(H)]</span>
          </div>
        </div>

        <div className="factors-breakdown">
          <div className="factor-card selection">
            <h5>Selection Factor</h5>
            <div className="factor-calc">
              <span>f(H) / ⟨f⟩ = {analysis.avgFitness.toFixed(2)} / {avgPopFitness.toFixed(2)}</span>
              <span className="factor-result">= {analysis.selectionFactor.toFixed(3)}</span>
            </div>
            <p className="factor-interpretation">
              {analysis.selectionFactor > 1 
                ? '✓ Above average → Schema GROWS' 
                : '✗ Below average → Schema SHRINKS'}
            </p>
          </div>

          <div className="factor-card crossover">
            <h5>Crossover Survival</h5>
            <div className="factor-calc">
              <span>1 - {crossoverRate.toFixed(2)} × {analysis.definingLength} / {chromosomeLength - 1}</span>
              <span className="factor-result">= {analysis.crossoverSurvival.toFixed(3)}</span>
            </div>
            <p className="factor-interpretation">
              {analysis.definingLength <= 2 
                ? '✓ Short defining length → High survival' 
                : '⚠ Long defining length → Lower survival'}
            </p>
          </div>

          <div className="factor-card mutation">
            <h5>Mutation Survival</h5>
            <div className="factor-calc">
              <span>(1 - {mutationRate.toFixed(3)})^{analysis.order}</span>
              <span className="factor-result">= {analysis.mutationSurvival.toFixed(3)}</span>
            </div>
            <p className="factor-interpretation">
              {analysis.order <= 2 
                ? '✓ Low order → High survival' 
                : '⚠ High order → Lower survival'}
            </p>
          </div>
        </div>

        <div className="total-growth">
          <h5>Expected Growth Rate</h5>
          <div className="growth-calc">
            <span>{analysis.selectionFactor.toFixed(3)} × {analysis.crossoverSurvival.toFixed(3)} × {analysis.mutationSurvival.toFixed(3)}</span>
            <span className={`growth-result ${analysis.expectedGrowth > 1 ? 'positive' : 'negative'}`}>
              = {analysis.expectedGrowth.toFixed(3)}
            </span>
          </div>
          <p className="growth-interpretation">
            {analysis.expectedGrowth > 1 
              ? `✓ Schema proportion expected to INCREASE by ${((analysis.expectedGrowth - 1) * 100).toFixed(1)}% per generation`
              : `✗ Schema proportion expected to DECREASE by ${((1 - analysis.expectedGrowth) * 100).toFixed(1)}% per generation`}
          </p>
        </div>
      </div>

      <div className="building-block-insight">
        <div className="insight-header">
          <Info size={18} />
          <h4>Building Block Hypothesis</h4>
        </div>
        <p>
          <strong>Short, low-order, above-average schemata</strong> are called <em>building blocks</em>.
          The GA works by discovering these building blocks and combining them into complete solutions.
        </p>
        <div className="bb-checklist">
          <div className={`bb-check ${analysis.definingLength <= 2 ? 'pass' : 'fail'}`}>
            {analysis.definingLength <= 2 ? '✓' : '✗'} Short defining length (δ ≤ 2): δ = {analysis.definingLength}
          </div>
          <div className={`bb-check ${analysis.order <= 3 ? 'pass' : 'fail'}`}>
            {analysis.order <= 3 ? '✓' : '✗'} Low order (o ≤ 3): o = {analysis.order}
          </div>
          <div className={`bb-check ${analysis.selectionFactor > 1 ? 'pass' : 'fail'}`}>
            {analysis.selectionFactor > 1 ? '✓' : '✗'} Above-average fitness: {analysis.selectionFactor.toFixed(2)} {'>'} 1
          </div>
        </div>
        <p className={`bb-verdict ${analysis.definingLength <= 2 && analysis.order <= 3 && analysis.selectionFactor > 1 ? 'is-bb' : 'not-bb'}`}>
          {analysis.definingLength <= 2 && analysis.order <= 3 && analysis.selectionFactor > 1
            ? '🧱 This IS a building block! It will propagate rapidly.'
            : '❌ This is NOT a good building block.'}
        </p>
      </div>
    </div>
  );
}
