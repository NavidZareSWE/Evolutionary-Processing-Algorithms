import React, { useState } from 'react';
import { Search, Calculator, Info, BookOpen, Link } from 'lucide-react';
import { FORMULAS, searchFormulas, getRelatedFormulas } from '../constants/formulas';
import MathBlock from './MathBlock';
import type { Formula } from '../types';

export default function FormulasView() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFormula, setSelectedFormula] = useState<Formula | null>(null);

  const filteredFormulas = searchQuery 
    ? searchFormulas(searchQuery)
    : Object.values(FORMULAS);

  const relatedFormulas = selectedFormula?.relatedFormulas 
    ? getRelatedFormulas(selectedFormula.id)
    : [];

  return (
    <div className="formulas-view">
      <div className="view-header">
        <h1 className="view-title">
          <Calculator size={28} className="text-primary-400" />
          Key Formulas
        </h1>
        <p className="view-subtitle">
          Mathematical foundations of Evolutionary Computing
        </p>
      </div>

      <div className="search-container">
        <Search size={20} className="search-icon" />
        <input
          type="text"
          placeholder="Search formulas..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="search-input"
        />
      </div>

      <div className="formulas-layout">
        <div className="formulas-list">
          {filteredFormulas.map((formula) => (
            <button
              key={formula.id}
              className={`formula-item ${selectedFormula?.id === formula.id ? 'selected' : ''}`}
              onClick={() => setSelectedFormula(formula)}
            >
              <h3 className="formula-title">{formula.name}</h3>
              <div className="formula-preview">
                <MathBlock latex={formula.latex} />
              </div>
            </button>
          ))}
          {filteredFormulas.length === 0 && (
            <p className="no-results">No formulas found for "{searchQuery}"</p>
          )}
        </div>

        <div className="formula-detail">
          {selectedFormula ? (
            <>
              <h2 className="detail-title">{selectedFormula.name}</h2>
              
              <div className="formula-display-large">
                <MathBlock latex={selectedFormula.latexDisplay} display />
              </div>

              <div className="detail-section">
                <div className="section-header">
                  <Info size={18} className="text-primary-400" />
                  <h3>Plain English</h3>
                </div>
                <p className="plain-english">{selectedFormula.plainEnglish}</p>
              </div>

              <div className="detail-section">
                <h3>Variables</h3>
                <div className="variables-table">
                  {Object.entries(selectedFormula.variables).map(([symbol, meaning]) => (
                    <div key={symbol} className="variable-row">
                      <span className="variable-symbol">
                        <MathBlock latex={symbol} />
                      </span>
                      <span className="variable-meaning">{meaning}</span>
                    </div>
                  ))}
                </div>
              </div>

              {selectedFormula.derivation && selectedFormula.derivation.length > 0 && (
                <div className="detail-section">
                  <h3>Derivation</h3>
                  <ol className="derivation-steps">
                    {selectedFormula.derivation.map((step, i) => (
                      <li key={i} className="derivation-step">{step}</li>
                    ))}
                  </ol>
                </div>
              )}

              <div className="detail-section">
                <div className="section-header">
                  <BookOpen size={18} className="text-accent-400" />
                  <h3>When to Use</h3>
                </div>
                <p className="when-to-use">{selectedFormula.whenToUse}</p>
              </div>

              {selectedFormula.example && (
                <div className="detail-section example-section">
                  <h3>Example</h3>
                  <div className="example-box">
                    <p>{selectedFormula.example}</p>
                  </div>
                </div>
              )}

              {relatedFormulas.length > 0 && (
                <div className="detail-section">
                  <div className="section-header">
                    <Link size={18} className="text-primary-400" />
                    <h3>Related Formulas</h3>
                  </div>
                  <div className="related-formulas-list">
                    {relatedFormulas.map((rf) => (
                      <button
                        key={rf.id}
                        className="related-formula-btn"
                        onClick={() => setSelectedFormula(rf)}
                      >
                        <span className="rf-name">{rf.name}</span>
                        <MathBlock latex={rf.latex} />
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="no-selection">
              <Calculator size={48} className="text-surface-600" />
              <p>Select a formula to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
