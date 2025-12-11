import React, { useEffect, useRef } from "react";
import {
  ChevronLeft,
  ChevronRight,
  AlertCircle,
  Lightbulb,
  BookOpen,
  Sparkles,
  CheckCircle2,
  FileText,
  Calculator,
} from "lucide-react";
import { useNavigation } from "../contexts/NavigationContext";
import { getSlideByNumber } from "../constants/slides";
import { getDefinition } from "../constants/definitions";
import { getFormula } from "../constants/formulas";
import MathBlock from "./MathBlock";

export default function SlideViewer() {
  const {
    currentSlide,
    goToNextSlide,
    goToPrevSlide,
    isFirstSlide,
    isLastSlide,
    totalSlides,
    goToSlide,
  } = useNavigation();

  const contentRef = useRef<HTMLDivElement>(null);

  const slide = getSlideByNumber(currentSlide);

  // Scroll to top when slide changes
  useEffect(() => {
    contentRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [currentSlide]);

  if (!slide) {
    return (
      <div className="slide-viewer">
        <div className="slide-error">
          <AlertCircle size={48} className="text-red-400" />
          <h2>Slide Not Found</h2>
          <p>Slide {currentSlide} could not be loaded.</p>
        </div>
      </div>
    );
  }

  const relatedDefinitions =
    slide.definitions?.map((term) => getDefinition(term)).filter(Boolean) || [];
  const relatedFormulas =
    slide.formulas?.map((id) => getFormula(id)).filter(Boolean) || [];

  // Handle progress bar click for direct navigation
  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const percentage = (e.clientX - rect.left) / rect.width;
    const targetSlide = Math.max(
      1,
      Math.min(totalSlides, Math.round(percentage * totalSlides)),
    );
    goToSlide(targetSlide);
  };

  return (
    <div className="slide-viewer">
      <div className="slide-header animate-fade-in">
        <div className="slide-info">
          <span className="slide-number gold-text">Slide {slide.number}</span>
          <span className="slide-divider">•</span>
          <span className="slide-session">Session {slide.session}</span>
        </div>
        <h2 className="slide-title">{slide.title}</h2>
      </div>

      <div className="slide-content" ref={contentRef}>
        <div className="slide-original animate-slide-up delay-1">
          <h3 className="section-title">
            <FileText size={18} className="text-primary-400" />
            Original Content
          </h3>
          <div className="content-box">
            {slide.content.split("\n").map((line, i) => (
              <p key={i} className={line.startsWith("•") ? "bullet-point" : ""}>
                {line}
              </p>
            ))}
          </div>
        </div>

        <div className="slide-explanation animate-slide-up delay-2">
          <h3 className="section-title">
            <BookOpen size={18} className="text-primary-400" />
            Detailed Explanation
          </h3>
          <div className="explanation-text">
            {slide.explanation.split("\n").map((para, i) => (
              <p key={i}>{para}</p>
            ))}
          </div>
        </div>

        <div className="slide-key-points animate-slide-up delay-3">
          <h3 className="section-title">
            <Sparkles size={18} className="text-gold" />
            Key Points
            <span className="key-points-count">
              {slide.keyPoints.length} points
            </span>
          </h3>
          <ul className="key-points-list enhanced">
            {slide.keyPoints.map((point, i) => (
              <li
                key={i}
                className="key-point-item"
                style={{ animationDelay: `${0.4 + i * 0.08}s` }}
              >
                <span className="key-point-number">{i + 1}</span>
                <span className="key-point-text">{point}</span>
                <CheckCircle2 size={16} className="key-point-check" />
              </li>
            ))}
          </ul>
        </div>

        {slide.professorNote && (
          <div className="professor-note animate-slide-up delay-4">
            <div className="note-header">
              <Lightbulb size={20} className="text-gold" />
              <span>Professor's Note</span>
              <span className="note-badge">Important</span>
            </div>
            <p>{slide.professorNote}</p>
          </div>
        )}

        {relatedDefinitions.length > 0 && (
          <div className="related-definitions animate-slide-up delay-5">
            <h3 className="section-title">
              <BookOpen size={18} className="text-accent-400" />
              Related Definitions
              <span className="related-count">{relatedDefinitions.length}</span>
            </h3>
            <div className="definitions-grid">
              {relatedDefinitions.map(
                (def, i) =>
                  def && (
                    <div
                      key={i}
                      className="definition-card"
                      style={{ animationDelay: `${0.6 + i * 0.1}s` }}
                    >
                      <h4 className="term">{def.term}</h4>
                      <p className="definition">{def.definition}</p>
                      {def.professorEmphasis && (
                        <p className="def-emphasis">
                          <Lightbulb size={14} />
                          {def.professorEmphasis}
                        </p>
                      )}
                    </div>
                  ),
              )}
            </div>
          </div>
        )}

        {relatedFormulas.length > 0 && (
          <div className="related-formulas animate-slide-up delay-6">
            <h3 className="section-title">
              <Calculator size={18} className="text-accent-400" />
              Related Formulas
              <span className="related-count">{relatedFormulas.length}</span>
            </h3>
            <div className="formulas-grid">
              {relatedFormulas.map(
                (formula, i) =>
                  formula && (
                    <div
                      key={i}
                      className="formula-card"
                      style={{ animationDelay: `${0.7 + i * 0.1}s` }}
                    >
                      <h4 className="formula-name">{formula.name}</h4>
                      <div className="formula-display">
                        <MathBlock latex={formula.latexDisplay} display />
                      </div>
                      <p className="formula-description">
                        {formula.plainEnglish}
                      </p>
                    </div>
                  ),
              )}
            </div>
          </div>
        )}
      </div>

      <div className="slide-navigation">
        <button
          className="nav-button prev"
          onClick={goToPrevSlide}
          disabled={isFirstSlide}
        >
          <ChevronLeft size={24} />
          <span>Previous</span>
        </button>

        <div className="slide-progress">
          <span className="progress-text">
            {currentSlide} / {totalSlides}
          </span>
          <div
            className="progress-bar clickable"
            onClick={handleProgressClick}
            title="Click to jump to any slide"
          >
            <div
              className="progress-fill gold-gradient"
              style={{ width: `${(currentSlide / totalSlides) * 100}%` }}
            />
            <div
              className="progress-thumb"
              style={{ left: `${(currentSlide / totalSlides) * 100}%` }}
            />
          </div>
        </div>

        <button
          className="nav-button next gold-action"
          onClick={goToNextSlide}
          disabled={isLastSlide}
        >
          <span>Next</span>
          <ChevronRight size={24} />
        </button>
      </div>
    </div>
  );
}
