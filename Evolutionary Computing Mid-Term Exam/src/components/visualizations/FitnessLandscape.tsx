import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Mountain,
  Play,
  Pause,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

type LandscapeType = "unimodal" | "multimodal" | "deceptive" | "rugged";

interface Point {
  x: number;
  y: number;
  fitness: number;
}

export default function FitnessLandscape() {
  const [landscapeType, setLandscapeType] = useState<LandscapeType>("unimodal");
  const [population, setPopulation] = useState<Point[]>([]);
  const [generation, setGeneration] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [bestPoint, setBestPoint] = useState<Point | null>(null);
  const [showContours, setShowContours] = useState(true);
  const [zoom, setZoom] = useState(1);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fitness functions for different landscapes
  const fitnessFunction = useCallback(
    (x: number, y: number): number => {
      switch (landscapeType) {
        case "unimodal":
          // Single peak at center
          return Math.exp(-((x - 50) ** 2 + (y - 50) ** 2) / 500);

        case "multimodal":
          // Multiple peaks
          return (
            0.5 * Math.exp(-((x - 25) ** 2 + (y - 25) ** 2) / 200) +
            0.7 * Math.exp(-((x - 75) ** 2 + (y - 25) ** 2) / 200) +
            0.6 * Math.exp(-((x - 25) ** 2 + (y - 75) ** 2) / 200) +
            1.0 * Math.exp(-((x - 75) ** 2 + (y - 75) ** 2) / 200)
          );

        case "deceptive":
          // Global optimum in corner, local optimum in center (deceptive)
          const centerAttractor =
            0.8 * Math.exp(-((x - 50) ** 2 + (y - 50) ** 2) / 800);
          const globalOptimum =
            1.0 * Math.exp(-((x - 90) ** 2 + (y - 90) ** 2) / 100);
          return Math.max(centerAttractor, globalOptimum);

        case "rugged":
          // Many local optima (NK-like)
          const base = 0.5 + 0.3 * Math.sin(x / 10) * Math.cos(y / 10);
          const noise = 0.2 * Math.sin(x / 3) * Math.sin(y / 3);
          const trend = 0.3 * Math.exp(-((x - 70) ** 2 + (y - 70) ** 2) / 1000);
          return base + noise + trend;

        default:
          return 0;
      }
    },
    [landscapeType],
  );

  // Initialize population
  const initPopulation = useCallback(() => {
    const newPop: Point[] = [];
    for (let i = 0; i < 20; i++) {
      const x = Math.random() * 100;
      const y = Math.random() * 100;
      newPop.push({ x, y, fitness: fitnessFunction(x, y) });
    }
    setPopulation(newPop);
    setGeneration(0);
    setBestPoint(
      newPop.reduce(
        (best, p) => (p.fitness > best.fitness ? p : best),
        newPop[0],
      ),
    );
    setIsRunning(false);
  }, [fitnessFunction]);

  // Run one generation of simple GA
  const runGeneration = useCallback(() => {
    if (population.length === 0) return;

    const newPop: Point[] = [];

    // Tournament selection and reproduction
    for (let i = 0; i < 20; i++) {
      // Tournament selection
      const tournament = [];
      for (let t = 0; t < 3; t++) {
        tournament.push(
          population[Math.floor(Math.random() * population.length)],
        );
      }
      const parent1 = tournament.reduce((best, p) =>
        p.fitness > best.fitness ? p : best,
      );

      const tournament2 = [];
      for (let t = 0; t < 3; t++) {
        tournament2.push(
          population[Math.floor(Math.random() * population.length)],
        );
      }
      const parent2 = tournament2.reduce((best, p) =>
        p.fitness > best.fitness ? p : best,
      );

      // Crossover (average + noise)
      let childX = (parent1.x + parent2.x) / 2;
      let childY = (parent1.y + parent2.y) / 2;

      // Mutation (Gaussian)
      if (Math.random() < 0.8) {
        childX += (Math.random() - 0.5) * 20;
        childY += (Math.random() - 0.5) * 20;
      }

      // Clamp to bounds
      childX = Math.max(0, Math.min(100, childX));
      childY = Math.max(0, Math.min(100, childY));

      newPop.push({
        x: childX,
        y: childY,
        fitness: fitnessFunction(childX, childY),
      });
    }

    // Elitism - keep best
    const currentBest = population.reduce((best, p) =>
      p.fitness > best.fitness ? p : best,
    );
    newPop[0] = { ...currentBest };

    setPopulation(newPop);
    setGeneration((g) => g + 1);

    const newBest = newPop.reduce((best, p) =>
      p.fitness > best.fitness ? p : best,
    );
    if (!bestPoint || newBest.fitness > bestPoint.fitness) {
      setBestPoint(newBest);
    }
  }, [population, fitnessFunction, bestPoint]);

  // Draw landscape
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = 300 * zoom;
    canvas.width = size;
    canvas.height = size;

    // Draw fitness landscape as heatmap
    const imageData = ctx.createImageData(size, size);
    for (let py = 0; py < size; py++) {
      for (let px = 0; px < size; px++) {
        const x = (px / size) * 100;
        const y = (py / size) * 100;
        const fitness = fitnessFunction(x, y);

        // Color mapping (blue-green-yellow-red)
        const r = Math.floor(255 * Math.min(1, fitness * 2));
        const g = Math.floor(255 * Math.min(1, fitness * 1.5));
        const b = Math.floor(255 * (1 - fitness));

        const idx = (py * size + px) * 4;
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    // Draw contour lines if enabled
    if (showContours) {
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 1;
      for (let level = 0.2; level <= 1; level += 0.2) {
        ctx.beginPath();
        for (let angle = 0; angle < Math.PI * 2; angle += 0.1) {
          // This is a simplified contour - real implementation would trace level sets
        }
      }
    }

    // Draw population
    population.forEach((point, i) => {
      const px = (point.x / 100) * size;
      const py = (point.y / 100) * size;

      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle =
        bestPoint && point.x === bestPoint.x && point.y === bestPoint.y
          ? "#fbbf24" // Gold for best
          : "#60a5fa"; // Blue for others
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Mark global optimum location
    let optX = 50,
      optY = 50;
    if (landscapeType === "multimodal") {
      optX = 75;
      optY = 75;
    }
    if (landscapeType === "deceptive") {
      optX = 90;
      optY = 90;
    }
    if (landscapeType === "rugged") {
      optX = 70;
      optY = 70;
    }

    ctx.beginPath();
    ctx.arc((optX / 100) * size, (optY / 100) * size, 8, 0, Math.PI * 2);
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [
    population,
    fitnessFunction,
    showContours,
    zoom,
    bestPoint,
    landscapeType,
  ]);

  // Auto-run effect
  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(runGeneration, 300);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, runGeneration]);

  // Initialize on landscape change
  useEffect(() => {
    initPopulation();
  }, [landscapeType]);

  return (
    <div className="visualization-container">
      <div className="viz-header">
        <h3>
          <Mountain size={20} />
          Fitness Landscape Explorer
        </h3>
        <p className="viz-description">
          Visualize how a GA population evolves on different fitness landscapes.
          Blue dots are individuals, gold is the best found. Green circle marks
          the global optimum.
        </p>
      </div>

      <div className="landscape-type-selector">
        <button
          className={`type-btn ${landscapeType === "unimodal" ? "active" : ""}`}
          onClick={() => setLandscapeType("unimodal")}
        >
          Unimodal (Easy)
        </button>
        <button
          className={`type-btn ${landscapeType === "multimodal" ? "active" : ""}`}
          onClick={() => setLandscapeType("multimodal")}
        >
          Multimodal
        </button>
        <button
          className={`type-btn ${landscapeType === "deceptive" ? "active" : ""}`}
          onClick={() => setLandscapeType("deceptive")}
        >
          Deceptive
        </button>
        <button
          className={`type-btn ${landscapeType === "rugged" ? "active" : ""}`}
          onClick={() => setLandscapeType("rugged")}
        >
          Rugged (NK-like)
        </button>
      </div>

      <div className="viz-controls">
        <button
          className={`viz-btn ${isRunning ? "active" : ""}`}
          onClick={() => setIsRunning(!isRunning)}
        >
          {isRunning ? <Pause size={16} /> : <Play size={16} />}
          {isRunning ? "Pause" : "Run"}
        </button>
        <button
          className="viz-btn"
          onClick={runGeneration}
          disabled={isRunning}
        >
          Step
        </button>
        <button className="viz-btn" onClick={initPopulation}>
          <RotateCcw size={16} />
          Reset
        </button>
        <button
          className="viz-btn"
          onClick={() => setZoom((z) => Math.min(2, z + 0.25))}
        >
          <ZoomIn size={16} />
        </button>
        <button
          className="viz-btn"
          onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
        >
          <ZoomOut size={16} />
        </button>
      </div>

      <div className="landscape-display">
        <canvas ref={canvasRef} className="landscape-canvas" />

        <div className="landscape-stats">
          <div className="stat-item">
            <span className="stat-label">Generation</span>
            <span className="stat-value">{generation}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Best Fitness</span>
            <span className="stat-value">
              {bestPoint?.fitness.toFixed(4) || "-"}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Best Position</span>
            <span className="stat-value">
              ({bestPoint?.x.toFixed(1) || "-"},{" "}
              {bestPoint?.y.toFixed(1) || "-"})
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Pop. Diversity</span>
            <span className="stat-value">
              {population.length > 0
                ? Math.sqrt(
                    population.reduce(
                      (sum, p) =>
                        sum +
                        (p.x - population[0].x) ** 2 +
                        (p.y - population[0].y) ** 2,
                      0,
                    ) / population.length,
                  ).toFixed(1)
                : "-"}
            </span>
          </div>
        </div>
      </div>

      <div className="landscape-explanation">
        {landscapeType === "unimodal" && (
          <div className="explanation-box success">
            <h4>Unimodal Landscape</h4>
            <p>
              Single global optimum at center. Easy for GA—any hill-climbing
              works!
            </p>
            <p>The population should quickly converge to the peak.</p>
          </div>
        )}
        {landscapeType === "multimodal" && (
          <div className="explanation-box warning">
            <h4>Multimodal Landscape</h4>
            <p>
              Multiple peaks of varying heights. GA must maintain diversity to
              find the best.
            </p>
            <p>Watch for premature convergence to suboptimal peaks!</p>
          </div>
        )}
        {landscapeType === "deceptive" && (
          <div className="explanation-box danger">
            <h4>Deceptive Landscape</h4>
            <p>
              The center attracts the population, but the global optimum is in
              the corner! Building blocks point away from the true optimum.
            </p>
            <p>
              <strong>This is GA-hard!</strong> Notice how the population gets
              stuck.
            </p>
          </div>
        )}
        {landscapeType === "rugged" && (
          <div className="explanation-box warning">
            <h4>Rugged (NK-like) Landscape</h4>
            <p>
              Many local optima with epistatic interactions between variables.
            </p>
            <p>GA needs good balance of exploration/exploitation.</p>
          </div>
        )}
      </div>
    </div>
  );
}
