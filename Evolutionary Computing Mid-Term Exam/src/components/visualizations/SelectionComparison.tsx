import React, { useState, useCallback, useEffect } from "react";
import { BarChart3, RefreshCw, Play } from "lucide-react";

interface Individual {
  id: number;
  fitness: number;
  fpsProbability: number;
  rankProbability: number;
  tournamentWins: number;
  selected: {
    fps: number;
    rank: number;
    tournament: number;
  };
}

export default function SelectionComparison() {
  const [individuals, setIndividuals] = useState<Individual[]>([]);
  const [tournamentSize, setTournamentSize] = useState(2);
  const [rankingPressure, setRankingPressure] = useState(1.5);
  const [numSelections, setNumSelections] = useState(100);
  const [hasRun, setHasRun] = useState(false);

  const generatePopulation = useCallback(() => {
    // Generate population with varying fitness values
    const fitnesses = [10, 20, 30, 35, 40, 50, 60, 80, 90, 100];
    const totalFitness = fitnesses.reduce((a, b) => a + b, 0);
    const n = fitnesses.length;

    const newIndividuals: Individual[] = fitnesses.map((fitness, i) => {
      // FPS probability
      const fpsProbability = fitness / totalFitness;

      // Linear ranking probability
      const rank = i + 1; // 1 is worst, n is best
      const s = rankingPressure;
      const rankProbability =
        (1 / n) * (s - ((s - 1) * (2 * (rank - 1))) / (n - 1));

      return {
        id: i,
        fitness,
        fpsProbability,
        rankProbability,
        tournamentWins: 0,
        selected: { fps: 0, rank: 0, tournament: 0 },
      };
    });

    setIndividuals(newIndividuals);
    setHasRun(false);
  }, [rankingPressure]);

  useEffect(() => {
    generatePopulation();
  }, [generatePopulation]);

  const runSelections = useCallback(() => {
    const newIndividuals = individuals.map((ind) => ({
      ...ind,
      selected: { fps: 0, rank: 0, tournament: 0 },
      tournamentWins: 0,
    }));

    // Run FPS selections
    for (let i = 0; i < numSelections; i++) {
      const rand = Math.random();
      let cumulative = 0;
      for (const ind of newIndividuals) {
        cumulative += ind.fpsProbability;
        if (rand <= cumulative) {
          ind.selected.fps++;
          break;
        }
      }
    }

    // Run Rank selections
    for (let i = 0; i < numSelections; i++) {
      const rand = Math.random();
      let cumulative = 0;
      for (const ind of newIndividuals) {
        cumulative += ind.rankProbability;
        if (rand <= cumulative) {
          ind.selected.rank++;
          break;
        }
      }
    }

    // Run Tournament selections
    for (let i = 0; i < numSelections; i++) {
      // Select tournament participants
      const participants: number[] = [];
      while (participants.length < tournamentSize) {
        const idx = Math.floor(Math.random() * newIndividuals.length);
        if (!participants.includes(idx)) {
          participants.push(idx);
        }
      }
      // Winner is the one with highest fitness
      const winner = participants.reduce(
        (best, idx) =>
          newIndividuals[idx].fitness > newIndividuals[best].fitness
            ? idx
            : best,
        participants[0],
      );
      newIndividuals[winner].selected.tournament++;
      newIndividuals[winner].tournamentWins++;
    }

    setIndividuals(newIndividuals);
    setHasRun(true);
  }, [individuals, numSelections, tournamentSize]);

  const maxSelections = Math.max(
    ...individuals.map((ind) =>
      Math.max(ind.selected.fps, ind.selected.rank, ind.selected.tournament),
    ),
    1,
  );

  return (
    <div className="visualization-container">
      <div className="viz-header">
        <h3>
          <BarChart3 size={20} />
          Selection Methods Comparison
        </h3>
        <p className="viz-description">
          Compare how FPS (Fitness Proportionate), Rank-based, and Tournament
          selection distribute selections. Notice how FPS over-selects the
          fittest while Tournament uses only comparisons (CORRECT fitness).
        </p>
      </div>

      <div className="viz-controls">
        <button className="viz-btn primary" onClick={runSelections}>
          <Play size={16} />
          Run {numSelections} Selections
        </button>
        <button className="viz-btn" onClick={generatePopulation}>
          <RefreshCw size={16} />
          Reset
        </button>
      </div>

      <div className="viz-settings inline">
        <div className="setting-row">
          <label>Tournament Size (k): {tournamentSize}</label>
          <input
            type="range"
            min="2"
            max="5"
            value={tournamentSize}
            onChange={(e) => setTournamentSize(Number(e.target.value))}
          />
        </div>
        <div className="setting-row">
          <label>Ranking Pressure (s): {rankingPressure.toFixed(1)}</label>
          <input
            type="range"
            min="10"
            max="20"
            value={rankingPressure * 10}
            onChange={(e) => setRankingPressure(Number(e.target.value) / 10)}
          />
        </div>
        <div className="setting-row">
          <label>Selections: {numSelections}</label>
          <input
            type="range"
            min="50"
            max="500"
            step="50"
            value={numSelections}
            onChange={(e) => setNumSelections(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="selection-comparison-grid">
        <div className="comparison-header">
          <span className="col-fitness">Fitness</span>
          <span className="col-method">FPS</span>
          <span className="col-method">Rank</span>
          <span className="col-method">Tournament (k={tournamentSize})</span>
        </div>

        {[...individuals].reverse().map((ind) => (
          <div key={ind.id} className="comparison-row">
            <span className="col-fitness">
              <span className="fitness-value">{ind.fitness}</span>
              <div
                className="fitness-bar"
                style={{ width: `${ind.fitness}%` }}
              />
            </span>

            <span className="col-method fps">
              <span className="prob">
                P={(ind.fpsProbability * 100).toFixed(1)}%
              </span>
              <span className="count">{hasRun ? ind.selected.fps : "-"}</span>
              {hasRun && (
                <div
                  className="selection-bar fps"
                  style={{
                    width: `${(ind.selected.fps / maxSelections) * 100}%`,
                  }}
                />
              )}
            </span>

            <span className="col-method rank">
              <span className="prob">
                P={(ind.rankProbability * 100).toFixed(1)}%
              </span>
              <span className="count">{hasRun ? ind.selected.rank : "-"}</span>
              {hasRun && (
                <div
                  className="selection-bar rank"
                  style={{
                    width: `${(ind.selected.rank / maxSelections) * 100}%`,
                  }}
                />
              )}
            </span>

            <span className="col-method tournament">
              <span className="count">
                {hasRun ? ind.selected.tournament : "-"}
              </span>
              {hasRun && (
                <div
                  className="selection-bar tournament"
                  style={{
                    width: `${(ind.selected.tournament / maxSelections) * 100}%`,
                  }}
                />
              )}
            </span>
          </div>
        ))}
      </div>

      {hasRun && (
        <div className="selection-insights">
          <h4>Key Observations:</h4>
          <ul>
            <li>
              <strong>FPS Problem:</strong> The individual with fitness 100 has
              {(
                individuals[9].fpsProbability / individuals[0].fpsProbability
              ).toFixed(1)}
              x higher probability than fitness 10. This uses EXACT fitness
              values!
            </li>
            <li>
              <strong>Rank-based:</strong> More uniform distribution. Only uses
              RANKINGS, not exact values. Adding 1000 to all fitness values
              wouldn't change anything.
            </li>
            <li>
              <strong>Tournament:</strong> Only compares "who is better?"
              (CORRECT fitness). Higher k = more selection pressure. No need to
              know exact fitness values!
            </li>
          </ul>
          <div className="insight-highlight">
            <strong>Professor's Key Point:</strong> We know CORRECT fitness
            (rankings), not EXACT fitness (values). Tournament selection is
            robust because it only uses comparisons!
          </div>
        </div>
      )}
    </div>
  );
}
