import React, { useState, useMemo } from "react";
import {
  Search,
  FileText,
  BookOpen,
  Calculator,
  ArrowRight,
} from "lucide-react";
import { searchSlides } from "../constants/slides";
import { searchDefinitions } from "../constants/definitions";
import { searchFormulas } from "../constants/formulas";
import MathBlock from "./MathBlock";

type ResultType = "slide" | "definition" | "formula";

interface SearchResult {
  type: ResultType;
  title: string;
  preview: string;
  data: any;
}

interface SearchViewProps {
  onNavigateToSlide: (slideNumber: number) => void;
}

export default function SearchView({ onNavigateToSlide }: SearchViewProps) {
  const [query, setQuery] = useState("");
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(
    null,
  );

  const results = useMemo(() => {
    if (!query.trim()) return [];

    const searchResults: SearchResult[] = [];

    // Search slides
    const slideResults = searchSlides(query);
    slideResults.forEach((slide) => {
      searchResults.push({
        type: "slide",
        title: `Slide ${slide.number}: ${slide.title}`,
        preview: slide.explanation.slice(0, 150) + "...",
        data: slide,
      });
    });

    // Search definitions
    const defResults = searchDefinitions(query);
    defResults.forEach((def) => {
      searchResults.push({
        type: "definition",
        title: def.term,
        preview: def.definition.slice(0, 150) + "...",
        data: def,
      });
    });

    // Search formulas
    const formulaResults = searchFormulas(query);
    formulaResults.forEach((formula) => {
      searchResults.push({
        type: "formula",
        title: formula.name,
        preview: formula.plainEnglish.slice(0, 150) + "...",
        data: formula,
      });
    });

    return searchResults;
  }, [query]);

  const getIcon = (type: ResultType) => {
    switch (type) {
      case "slide":
        return <FileText size={18} className="text-primary-400" />;
      case "definition":
        return <BookOpen size={18} className="text-accent-400" />;
      case "formula":
        return <Calculator size={18} className="text-green-400" />;
    }
  };

  const handleResultClick = (result: SearchResult) => {
    setSelectedResult(result);
    if (result.type === "slide") {
      onNavigateToSlide(result.data.number);
    }
  };

  return (
    <div className="search-view">
      <div className="view-header">
        <h1 className="view-title">
          <Search size={28} className="text-primary-400" />
          Search Course Content
        </h1>
        <p className="view-subtitle">
          Search across slides, definitions, and formulas
        </p>
      </div>

      <div className="search-container large">
        <Search size={24} className="search-icon" />
        <input
          type="text"
          placeholder="Search for topics, terms, or concepts..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="search-input large"
          autoFocus
        />
      </div>

      {query && (
        <div className="search-stats">
          Found {results.length} result{results.length !== 1 ? "s" : ""} for "
          {query}"
        </div>
      )}

      <div className="search-results-layout">
        <div className="results-list">
          {results.map((result, i) => (
            <button
              key={`${result.type}-${i}`}
              className={`result-item ${selectedResult === result ? "selected" : ""}`}
              onClick={() => handleResultClick(result)}
            >
              <div className="result-icon">{getIcon(result.type)}</div>
              <div className="result-content">
                <div className="result-type">{result.type}</div>
                <h3 className="result-title">{result.title}</h3>
                <p className="result-preview">{result.preview}</p>
              </div>
              <ArrowRight size={16} className="result-arrow" />
            </button>
          ))}
          {query && results.length === 0 && (
            <div className="no-results-message">
              <Search size={48} className="text-surface-600" />
              <p>No results found for "{query}"</p>
              <p className="suggestion">
                Try different keywords or check spelling
              </p>
            </div>
          )}
        </div>

        {selectedResult && (
          <div className="result-detail">
            <div className="detail-header">
              {getIcon(selectedResult.type)}
              <span className="detail-type">{selectedResult.type}</span>
            </div>
            <h2 className="detail-title">{selectedResult.title}</h2>

            {selectedResult.type === "slide" && (
              <div className="slide-detail">
                <div className="detail-section">
                  <h3>Content</h3>
                  <div className="content-box">
                    {selectedResult.data.content
                      .split("\n")
                      .map((line: string, i: number) => (
                        <p key={i}>{line}</p>
                      ))}
                  </div>
                </div>
                <div className="detail-section">
                  <h3>Explanation</h3>
                  <p>{selectedResult.data.explanation}</p>
                </div>
                <div className="detail-section">
                  <h3>Key Points</h3>
                  <ul>
                    {selectedResult.data.keyPoints.map(
                      (kp: string, i: number) => (
                        <li key={i}>{kp}</li>
                      ),
                    )}
                  </ul>
                </div>
                <button
                  className="btn btn-primary"
                  onClick={() => onNavigateToSlide(selectedResult.data.number)}
                >
                  Go to Slide {selectedResult.data.number}
                </button>
              </div>
            )}

            {selectedResult.type === "definition" && (
              <div className="definition-detail">
                <div className="detail-section">
                  <h3>Definition</h3>
                  <p>{selectedResult.data.definition}</p>
                </div>
                {selectedResult.data.professorEmphasis && (
                  <div className="detail-section professor">
                    <h3>Professor's Emphasis</h3>
                    <p>{selectedResult.data.professorEmphasis}</p>
                  </div>
                )}
                {selectedResult.data.examples && (
                  <div className="detail-section">
                    <h3>Examples</h3>
                    <ul>
                      {selectedResult.data.examples.map(
                        (ex: string, i: number) => (
                          <li key={i}>{ex}</li>
                        ),
                      )}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {selectedResult.type === "formula" && (
              <div className="formula-detail">
                <div className="formula-display-large">
                  <MathBlock latex={selectedResult.data.latexDisplay} display />
                </div>
                <div className="detail-section">
                  <h3>Plain English</h3>
                  <p>{selectedResult.data.plainEnglish}</p>
                </div>
                <div className="detail-section">
                  <h3>When to Use</h3>
                  <p>{selectedResult.data.whenToUse}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
