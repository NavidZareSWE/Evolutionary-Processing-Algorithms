import React, { useState, useCallback } from "react";
import { Scissors, RefreshCw, Shuffle } from "lucide-react";

type CrossoverType = "one-point" | "two-point" | "uniform" | "pmx";

export default function CrossoverVisualization() {
  const [crossoverType, setCrossoverType] =
    useState<CrossoverType>("one-point");
  const [parent1, setParent1] = useState<number[]>([
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
  ]);
  const [parent2, setParent2] = useState<number[]>([
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
  ]);
  const [child1, setChild1] = useState<number[]>([]);
  const [child2, setChild2] = useState<number[]>([]);
  const [crossoverPoint, setCrossoverPoint] = useState(5);
  const [crossoverPoint2, setCrossoverPoint2] = useState(7);
  const [uniformMask, setUniformMask] = useState<boolean[]>([]);
  const [hasRun, setHasRun] = useState(false);

  // For PMX
  const [permParent1, setPermParent1] = useState([1, 2, 3, 4, 5, 6, 7, 8]);
  const [permParent2, setPermParent2] = useState([3, 7, 5, 1, 6, 8, 2, 4]);
  const [permChild1, setPermChild1] = useState<number[]>([]);
  const [permChild2, setPermChild2] = useState<number[]>([]);

  const randomizeParents = useCallback(() => {
    if (crossoverType === "pmx") {
      // Shuffle permutations
      const arr1 = [1, 2, 3, 4, 5, 6, 7, 8];
      const arr2 = [1, 2, 3, 4, 5, 6, 7, 8];
      for (let i = arr1.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr1[i], arr1[j]] = [arr1[j], arr1[i]];
        const k = Math.floor(Math.random() * (i + 1));
        [arr2[i], arr2[k]] = [arr2[k], arr2[i]];
      }
      setPermParent1(arr1);
      setPermParent2(arr2);
    } else {
      setParent1(
        Array.from({ length: 10 }, () => (Math.random() < 0.5 ? 1 : 0)),
      );
      setParent2(
        Array.from({ length: 10 }, () => (Math.random() < 0.5 ? 1 : 0)),
      );
    }
    setHasRun(false);
    setChild1([]);
    setChild2([]);
    setPermChild1([]);
    setPermChild2([]);
  }, [crossoverType]);

  const runCrossover = useCallback(() => {
    if (crossoverType === "pmx") {
      // PMX Crossover
      const p1 = [...permParent1];
      const p2 = [...permParent2];
      const c1: (number | null)[] = Array(8).fill(null);
      const c2: (number | null)[] = Array(8).fill(null);

      // Copy segment from parents
      const start = crossoverPoint - 1;
      const end = crossoverPoint2;

      for (let i = start; i < end; i++) {
        c1[i] = p1[i];
        c2[i] = p2[i];
      }

      // Build mapping
      const mapping1to2: Record<number, number> = {};
      const mapping2to1: Record<number, number> = {};
      for (let i = start; i < end; i++) {
        mapping1to2[p1[i]] = p2[i];
        mapping2to1[p2[i]] = p1[i];
      }

      // Fill remaining positions
      for (let i = 0; i < 8; i++) {
        if (c1[i] === null) {
          let val = p2[i];
          while (c1.includes(val)) {
            val = mapping2to1[val] || val;
            if (val === p2[i]) break;
          }
          if (!c1.includes(val)) c1[i] = val;
          else {
            // Find any unused value
            for (let v = 1; v <= 8; v++) {
              if (!c1.includes(v)) {
                c1[i] = v;
                break;
              }
            }
          }
        }
        if (c2[i] === null) {
          let val = p1[i];
          while (c2.includes(val)) {
            val = mapping1to2[val] || val;
            if (val === p1[i]) break;
          }
          if (!c2.includes(val)) c2[i] = val;
          else {
            for (let v = 1; v <= 8; v++) {
              if (!c2.includes(v)) {
                c2[i] = v;
                break;
              }
            }
          }
        }
      }

      setPermChild1(c1 as number[]);
      setPermChild2(c2 as number[]);
    } else {
      // Binary crossovers
      let c1: number[] = [];
      let c2: number[] = [];

      switch (crossoverType) {
        case "one-point":
          c1 = [
            ...parent1.slice(0, crossoverPoint),
            ...parent2.slice(crossoverPoint),
          ];
          c2 = [
            ...parent2.slice(0, crossoverPoint),
            ...parent1.slice(crossoverPoint),
          ];
          break;

        case "two-point":
          const start = Math.min(crossoverPoint, crossoverPoint2) - 1;
          const end = Math.max(crossoverPoint, crossoverPoint2);
          c1 = [
            ...parent1.slice(0, start),
            ...parent2.slice(start, end),
            ...parent1.slice(end),
          ];
          c2 = [
            ...parent2.slice(0, start),
            ...parent1.slice(start, end),
            ...parent2.slice(end),
          ];
          break;

        case "uniform":
          const mask = Array.from({ length: 10 }, () => Math.random() < 0.5);
          setUniformMask(mask);
          c1 = parent1.map((gene, i) => (mask[i] ? gene : parent2[i]));
          c2 = parent2.map((gene, i) => (mask[i] ? gene : parent1[i]));
          break;
      }

      setChild1(c1);
      setChild2(c2);
    }
    setHasRun(true);
  }, [
    crossoverType,
    parent1,
    parent2,
    crossoverPoint,
    crossoverPoint2,
    permParent1,
    permParent2,
  ]);

  const renderBinaryChromosome = (
    genes: number[],
    label: string,
    highlights?: boolean[],
  ) => (
    <div className="chromosome-row">
      <span className="chromosome-label">{label}</span>
      <div className="chromosome">
        {genes.map((gene, i) => (
          <span
            key={i}
            className={`gene-cell ${gene === 1 ? "one" : "zero"} ${highlights?.[i] ? "highlighted" : ""}`}
          >
            {gene}
          </span>
        ))}
      </div>
    </div>
  );

  const renderPermChromosome = (
    genes: number[],
    label: string,
    highlightRange?: [number, number],
  ) => (
    <div className="chromosome-row">
      <span className="chromosome-label">{label}</span>
      <div className="chromosome permutation">
        {genes.map((gene, i) => (
          <span
            key={i}
            className={`gene-cell perm ${highlightRange && i >= highlightRange[0] && i < highlightRange[1] ? "highlighted" : ""}`}
          >
            {gene}
          </span>
        ))}
      </div>
    </div>
  );

  return (
    <div className="visualization-container">
      <div className="viz-header">
        <h3>
          <Scissors size={20} />
          Crossover Operators Visualization
        </h3>
        <p className="viz-description">
          See how different crossover operators combine parent chromosomes.
          Crossover EXPLORES by taking large steps between parents—it cannot
          create new alleles!
        </p>
      </div>

      <div className="crossover-type-selector">
        <button
          className={`type-btn ${crossoverType === "one-point" ? "active" : ""}`}
          onClick={() => {
            setCrossoverType("one-point");
            setHasRun(false);
          }}
        >
          One-Point
        </button>
        <button
          className={`type-btn ${crossoverType === "two-point" ? "active" : ""}`}
          onClick={() => {
            setCrossoverType("two-point");
            setHasRun(false);
          }}
        >
          Two-Point
        </button>
        <button
          className={`type-btn ${crossoverType === "uniform" ? "active" : ""}`}
          onClick={() => {
            setCrossoverType("uniform");
            setHasRun(false);
          }}
        >
          Uniform
        </button>
        <button
          className={`type-btn ${crossoverType === "pmx" ? "active" : ""}`}
          onClick={() => {
            setCrossoverType("pmx");
            setHasRun(false);
          }}
        >
          PMX (Permutation)
        </button>
      </div>

      <div className="viz-controls">
        <button className="viz-btn primary" onClick={runCrossover}>
          <Scissors size={16} />
          Run Crossover
        </button>
        <button className="viz-btn" onClick={randomizeParents}>
          <Shuffle size={16} />
          Randomize Parents
        </button>
      </div>

      {crossoverType !== "pmx" && crossoverType !== "uniform" && (
        <div className="viz-settings inline">
          <div className="setting-row">
            <label>Crossover Point 1: {crossoverPoint}</label>
            <input
              type="range"
              min="1"
              max="9"
              value={crossoverPoint}
              onChange={(e) => setCrossoverPoint(Number(e.target.value))}
            />
          </div>
          {crossoverType === "two-point" && (
            <div className="setting-row">
              <label>Crossover Point 2: {crossoverPoint2}</label>
              <input
                type="range"
                min="2"
                max="10"
                value={crossoverPoint2}
                onChange={(e) => setCrossoverPoint2(Number(e.target.value))}
              />
            </div>
          )}
        </div>
      )}

      {crossoverType === "pmx" && (
        <div className="viz-settings inline">
          <div className="setting-row">
            <label>Segment Start: {crossoverPoint}</label>
            <input
              type="range"
              min="1"
              max="6"
              value={crossoverPoint}
              onChange={(e) => setCrossoverPoint(Number(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label>Segment End: {crossoverPoint2}</label>
            <input
              type="range"
              min="3"
              max="8"
              value={crossoverPoint2}
              onChange={(e) => setCrossoverPoint2(Number(e.target.value))}
            />
          </div>
        </div>
      )}

      <div className="crossover-visualization">
        {crossoverType !== "pmx" ? (
          <>
            <div className="parents-section">
              <h4>Parents</h4>
              {renderBinaryChromosome(
                parent1,
                "P1",
                crossoverType === "one-point"
                  ? parent1.map((_, i) => i < crossoverPoint)
                  : crossoverType === "two-point"
                    ? parent1.map(
                        (_, i) =>
                          i < Math.min(crossoverPoint, crossoverPoint2) - 1 ||
                          i >= Math.max(crossoverPoint, crossoverPoint2),
                      )
                    : hasRun
                      ? uniformMask
                      : undefined,
              )}
              {renderBinaryChromosome(
                parent2,
                "P2",
                crossoverType === "one-point"
                  ? parent2.map((_, i) => i >= crossoverPoint)
                  : crossoverType === "two-point"
                    ? parent2.map(
                        (_, i) =>
                          i >= Math.min(crossoverPoint, crossoverPoint2) - 1 &&
                          i < Math.max(crossoverPoint, crossoverPoint2),
                      )
                    : hasRun
                      ? uniformMask.map((m) => !m)
                      : undefined,
              )}
            </div>

            {crossoverType === "one-point" && (
              <div className="crossover-point-indicator">
                <div
                  className="point-line"
                  style={{ left: `${(crossoverPoint / 10) * 100}%` }}
                >
                  <span className="point-label">Cut</span>
                </div>
              </div>
            )}

            {hasRun && (
              <div className="children-section">
                <h4>Children</h4>
                {renderBinaryChromosome(child1, "C1")}
                {renderBinaryChromosome(child2, "C2")}
              </div>
            )}
          </>
        ) : (
          <>
            <div className="parents-section">
              <h4>Parents (Permutations)</h4>
              {renderPermChromosome(permParent1, "P1", [
                crossoverPoint - 1,
                crossoverPoint2,
              ])}
              {renderPermChromosome(permParent2, "P2", [
                crossoverPoint - 1,
                crossoverPoint2,
              ])}
            </div>

            {hasRun && (
              <div className="children-section">
                <h4>Children</h4>
                {renderPermChromosome(permChild1, "C1", [
                  crossoverPoint - 1,
                  crossoverPoint2,
                ])}
                {renderPermChromosome(permChild2, "C2", [
                  crossoverPoint - 1,
                  crossoverPoint2,
                ])}
              </div>
            )}
          </>
        )}
      </div>

      <div className="crossover-explanation">
        {crossoverType === "one-point" && (
          <div className="explanation-box">
            <h4>One-Point Crossover</h4>
            <p>
              A single cut point is chosen. Child 1 gets P1's genes before the
              cut and P2's genes after. Child 2 gets the opposite.
            </p>
            <p className="warning">
              <strong>Positional Bias:</strong> Genes at opposite ends can NEVER
              stay together!
            </p>
          </div>
        )}
        {crossoverType === "two-point" && (
          <div className="explanation-box">
            <h4>Two-Point Crossover</h4>
            <p>
              Two cut points define a segment. The middle segment is swapped
              between parents. Reduces positional bias compared to one-point.
            </p>
          </div>
        )}
        {crossoverType === "uniform" && (
          <div className="explanation-box">
            <h4>Uniform Crossover</h4>
            <p>
              Each gene is independently chosen from either parent with 50%
              probability.
            </p>
            <p className="info">
              <strong>No positional bias</strong>, but has{" "}
              <strong>distributional bias</strong> toward 50% from each parent.
            </p>
          </div>
        )}
        {crossoverType === "pmx" && (
          <div className="explanation-box">
            <h4>PMX (Partially Mapped Crossover)</h4>
            <p>
              For permutations: A segment is copied directly from one parent.
              Remaining positions are filled using a mapping to avoid
              duplicates.
            </p>
            <p className="info">
              <strong>Preserves absolute positions</strong> — good when position
              matters (like 8-Queens).
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
