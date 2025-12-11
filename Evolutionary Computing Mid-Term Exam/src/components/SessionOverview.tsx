import React, { useState } from "react";
import {
  BookOpen,
  Target,
  List,
  ArrowRight,
  Sparkles,
  MousePointer,
  Zap,
} from "lucide-react";
import { useNavigation } from "../contexts/NavigationContext";
import { SESSIONS, getSessionById } from "../constants/sessions";
import { getSlidesBySession, SLIDES } from "../constants/slides";
import { DEFINITIONS } from "../constants/definitions";

interface SessionOverviewProps {
  onNavigateToSlide: (slideNumber: number) => void;
}

export default function SessionOverview({
  onNavigateToSlide,
}: SessionOverviewProps) {
  const { currentSession, goToSession } = useNavigation();
  const [hoveredTopic, setHoveredTopic] = useState<string | null>(null);
  const [hoveredSlide, setHoveredSlide] = useState<number | null>(null);
  const session = getSessionById(currentSession);
  const slides = getSlidesBySession(currentSession);

  if (!session) {
    return (
      <div className="session-overview">
        <p>Session not found</p>
      </div>
    );
  }

  // Find first slide that mentions a topic
  const findSlideForTopic = (topic: string): number | null => {
    const topicWords = topic
      .toLowerCase()
      .split(" ")
      .filter((w) => w.length > 3);
    const slide = slides.find((s) => {
      const searchText =
        `${s.title} ${s.content} ${s.explanation}`.toLowerCase();
      return topicWords.some((word) => searchText.includes(word));
    });
    return slide?.number || slides[0]?.number || null;
  };

  // Check if topic has a related definition
  const hasDefinition = (topic: string): boolean => {
    const topicLower = topic.toLowerCase();
    return Object.values(DEFINITIONS).some(
      (d) =>
        topicLower.includes(d.term.toLowerCase()) ||
        d.term.toLowerCase().includes(topicLower.split(" ")[0]),
    );
  };

  const handleTopicClick = (topic: string) => {
    const slideNum = findSlideForTopic(topic);
    if (slideNum) {
      onNavigateToSlide(slideNum);
    }
  };

  return (
    <div className="session-overview">
      <div className="session-header animate-fade-in">
        <div className="session-header-meta">
          <div className="session-badge-wrapper">
            <span className="session-badge gold-badge">
              Session {session.id}
            </span>
            <span className="session-slide-count">{slides.length} slides</span>
          </div>
        </div>
        <h1 className="session-title-main">{session.title}</h1>
        <p className="session-slide-range">
          Covering Slides {session.slideRange[0]} – {session.slideRange[1]}
        </p>
      </div>

      <div className="session-content-grid">
        <div className="session-card summary-card animate-slide-up delay-1">
          <div className="card-header">
            <BookOpen size={20} className="text-primary-400" />
            <h3>Summary</h3>
          </div>
          <p>{session.summary}</p>
        </div>

        <div className="session-card takeaway-card animate-slide-up delay-2">
          <div className="card-header">
            <Zap size={20} className="text-gold" />
            <h3>Key Takeaway</h3>
          </div>
          <p className="takeaway-text">{session.keyTakeaway}</p>
        </div>

        <div className="session-card topics-card animate-slide-up delay-3">
          <div className="card-header">
            <List size={20} className="text-primary-400" />
            <h3>Topics Covered</h3>
            <span className="card-hint">
              <MousePointer size={12} />
              Click to jump
            </span>
          </div>
          <ul className="topics-list interactive-topics">
            {session.topics.map((topic, i) => (
              <li
                key={i}
                className={`topic-item ${hoveredTopic === topic ? "hovered" : ""}`}
                onMouseEnter={() => setHoveredTopic(topic)}
                onMouseLeave={() => setHoveredTopic(null)}
                onClick={() => handleTopicClick(topic)}
                style={{ animationDelay: `${0.4 + i * 0.05}s` }}
              >
                <span className="topic-bullet"></span>
                <span className="topic-text">{topic}</span>
                {hasDefinition(topic) && (
                  <span className="topic-tag">📖 Defined</span>
                )}
                <ArrowRight size={14} className="topic-arrow" />
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="slides-section animate-slide-up delay-4">
        <div className="section-header-row">
          <h3 className="slides-title">
            <Sparkles size={20} className="text-gold" />
            Slides in This Session
          </h3>
          <span className="section-hint">
            Click any slide to view full details
          </span>
        </div>
        <div className="slides-grid">
          {slides.map((slide, index) => (
            <button
              key={slide.number}
              className={`slide-card ${hoveredSlide === slide.number ? "hovered" : ""}`}
              onClick={() => onNavigateToSlide(slide.number)}
              onMouseEnter={() => setHoveredSlide(slide.number)}
              onMouseLeave={() => setHoveredSlide(null)}
              style={{ animationDelay: `${0.5 + index * 0.03}s` }}
            >
              <span className="slide-num">#{slide.number}</span>
              <span className="slide-name">{slide.title}</span>
              <ArrowRight size={16} className="slide-arrow" />
            </button>
          ))}
        </div>
      </div>

      <div className="session-navigation animate-slide-up delay-5">
        <div className="section-header-row">
          <h3>All Sessions</h3>
          <span className="section-hint">Navigate the complete course</span>
        </div>
        <div className="all-sessions-grid">
          {SESSIONS.map((s, index) => (
            <button
              key={s.id}
              className={`session-nav-item ${s.id === currentSession ? "active" : ""}`}
              onClick={() => goToSession(s.id)}
              style={{ animationDelay: `${0.6 + index * 0.04}s` }}
            >
              <span className="nav-session-num">{s.id}</span>
              <span className="nav-session-title">{s.title}</span>
              <span className="nav-session-meta">
                {s.slideRange[1] - s.slideRange[0] + 1} slides
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
