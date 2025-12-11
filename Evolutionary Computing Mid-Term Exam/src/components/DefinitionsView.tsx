import React, { useState } from "react";
import { Search, BookOpen, Link, AlertTriangle, Lightbulb } from "lucide-react";
import {
  DEFINITIONS,
  searchDefinitions,
  getRelatedDefinitions,
} from "../constants/definitions";
import type { Definition } from "../types";

export default function DefinitionsView() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedDefinition, setSelectedDefinition] =
    useState<Definition | null>(null);

  const filteredDefinitions = searchQuery
    ? searchDefinitions(searchQuery)
    : Object.values(DEFINITIONS);

  const relatedDefs = selectedDefinition?.relatedTerms
    ? getRelatedDefinitions(selectedDefinition.term)
    : [];

  return (
    <div className="definitions-view">
      <div className="view-header">
        <h1 className="view-title">
          <BookOpen size={28} className="text-primary-400" />
          Key Definitions
        </h1>
        <p className="view-subtitle">
          Essential terminology for Evolutionary Computing
        </p>
      </div>

      <div className="search-container">
        <Search size={20} className="search-icon" />
        <input
          type="text"
          placeholder="Search definitions..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="search-input"
        />
      </div>

      <div className="definitions-layout">
        <div className="definitions-list">
          {filteredDefinitions.map((def) => (
            <button
              key={def.term}
              className={`definition-item ${selectedDefinition?.term === def.term ? "selected" : ""}`}
              onClick={() => setSelectedDefinition(def)}
            >
              <h3 className="def-term">{def.term}</h3>
              <p className="def-preview">{def.definition.slice(0, 100)}...</p>
            </button>
          ))}
          {filteredDefinitions.length === 0 && (
            <p className="no-results">
              No definitions found for "{searchQuery}"
            </p>
          )}
        </div>

        <div className="definition-detail">
          {selectedDefinition ? (
            <>
              <h2 className="detail-term">{selectedDefinition.term}</h2>

              <div className="detail-section">
                <h3>Definition</h3>
                <p className="detail-definition">
                  {selectedDefinition.definition}
                </p>
              </div>

              {selectedDefinition.professorEmphasis && (
                <div className="detail-section professor-emphasis">
                  <div className="emphasis-header">
                    <Lightbulb size={18} className="text-yellow-400" />
                    <h3>Professor's Emphasis</h3>
                  </div>
                  <p>{selectedDefinition.professorEmphasis}</p>
                </div>
              )}

              {selectedDefinition.examples &&
                selectedDefinition.examples.length > 0 && (
                  <div className="detail-section">
                    <h3>Examples</h3>
                    <ul className="examples-list">
                      {selectedDefinition.examples.map((ex, i) => (
                        <li key={i}>{ex}</li>
                      ))}
                    </ul>
                  </div>
                )}

              {selectedDefinition.commonMisconceptions &&
                selectedDefinition.commonMisconceptions.length > 0 && (
                  <div className="detail-section misconceptions">
                    <div className="misconception-header">
                      <AlertTriangle size={18} className="text-orange-400" />
                      <h3>Common Misconceptions</h3>
                    </div>
                    <ul className="misconceptions-list">
                      {selectedDefinition.commonMisconceptions.map((m, i) => (
                        <li key={i}>{m}</li>
                      ))}
                    </ul>
                  </div>
                )}

              {relatedDefs.length > 0 && (
                <div className="detail-section">
                  <div className="related-header">
                    <Link size={18} className="text-primary-400" />
                    <h3>Related Terms</h3>
                  </div>
                  <div className="related-terms">
                    {relatedDefs.map((rd) => (
                      <button
                        key={rd.term}
                        className="related-term-btn"
                        onClick={() => setSelectedDefinition(rd)}
                      >
                        {rd.term}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="no-selection">
              <BookOpen size={48} className="text-surface-600" />
              <p>Select a definition to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
