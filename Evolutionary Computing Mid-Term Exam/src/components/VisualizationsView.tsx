import React, { useState } from "react";
import { Play, BarChart3, Scissors, Mountain, Layers } from "lucide-react";
import GASimulation from "./visualizations/GASimulation";
import SelectionComparison from "./visualizations/SelectionComparison";
import CrossoverVisualization from "./visualizations/CrossoverVisualization";
import FitnessLandscape from "./visualizations/FitnessLandscape";
import SchemaVisualization from "./visualizations/SchemaVisualization";

type VisualizationType =
  | "ga-sim"
  | "selection"
  | "crossover"
  | "landscape"
  | "schema";

interface VisualizationInfo {
  id: VisualizationType;
  title: string;
  description: string;
  icon: React.ReactNode;
  component: React.ReactNode;
}

export default function VisualizationsView() {
  const [activeViz, setActiveViz] = useState<VisualizationType>("ga-sim");

  const visualizations: VisualizationInfo[] = [
    {
      id: "ga-sim",
      title: "GA Simulation",
      description:
        "Watch a genetic algorithm evolve to solve the OneMax problem",
      icon: <Play size={20} />,
      component: <GASimulation />,
    },
    {
      id: "selection",
      title: "Selection Methods",
      description: "Compare FPS, Rank-based, and Tournament selection",
      icon: <BarChart3 size={20} />,
      component: <SelectionComparison />,
    },
    {
      id: "crossover",
      title: "Crossover Operators",
      description: "Visualize how different crossover methods combine parents",
      icon: <Scissors size={20} />,
      component: <CrossoverVisualization />,
    },
    {
      id: "landscape",
      title: "Fitness Landscape",
      description: "Explore GA behavior on different landscape types",
      icon: <Mountain size={20} />,
      component: <FitnessLandscape />,
    },
    {
      id: "schema",
      title: "Schema Theorem",
      description: "Calculate schema properties and building block analysis",
      icon: <Layers size={20} />,
      component: <SchemaVisualization />,
    },
  ];

  const currentViz = visualizations.find((v) => v.id === activeViz);

  return (
    <div className="visualizations-view">
      <div className="view-header">
        <h1 className="view-title">
          <Play size={28} className="text-primary-400" />
          Interactive Visualizations
        </h1>
        <p className="view-subtitle">
          Learn by experimenting with these interactive simulations of
          evolutionary computing concepts
        </p>
      </div>

      <div className="viz-selector">
        {visualizations.map((viz) => (
          <button
            key={viz.id}
            className={`viz-selector-btn ${activeViz === viz.id ? "active" : ""}`}
            onClick={() => setActiveViz(viz.id)}
          >
            {viz.icon}
            <div className="viz-selector-text">
              <span className="viz-selector-title">{viz.title}</span>
              <span className="viz-selector-desc">{viz.description}</span>
            </div>
          </button>
        ))}
      </div>

      <div className="viz-content">{currentViz?.component}</div>
    </div>
  );
}
