import React, { useState, useEffect } from "react";
import {
  NavigationProvider,
  useNavigation,
} from "./contexts/NavigationContext";
import Sidebar from "./components/Sidebar";
import SessionOverview from "./components/SessionOverview";
import SlideViewer from "./components/SlideViewer";
import DefinitionsView from "./components/DefinitionsView";
import FormulasView from "./components/FormulasView";
import SearchView from "./components/SearchView";
import VisualizationsView from "./components/VisualizationsView";
import HelpGuide from "./components/HelpGuide";

type ViewType =
  | "sessions"
  | "definitions"
  | "formulas"
  | "search"
  | "visualizations";
type ContentType = "overview" | "slides";

function AppContent() {
  const [activeView, setActiveView] = useState<ViewType>("sessions");
  const [contentType, setContentType] = useState<ContentType>("overview");
  const { goToNextSlide, goToPrevSlide, goToSession, goToSlide } =
    useNavigation();

  // Function to navigate to a slide AND switch to slide view
  const navigateToSlide = (slideNumber: number) => {
    goToSlide(slideNumber);
    setActiveView("sessions");
    setContentType("slides");
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.key) {
        case "ArrowLeft":
          if (activeView === "sessions" && contentType === "slides") {
            goToPrevSlide();
          }
          break;
        case "ArrowRight":
          if (activeView === "sessions" && contentType === "slides") {
            goToNextSlide();
          }
          break;
        case "1":
        case "2":
        case "3":
        case "4":
        case "5":
        case "6":
        case "7":
        case "8":
        case "9":
          goToSession(parseInt(e.key));
          setActiveView("sessions");
          break;
        case "/":
          e.preventDefault();
          setActiveView("search");
          break;
        case "Escape":
          // Could close modals or clear search
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeView, contentType, goToNextSlide, goToPrevSlide, goToSession]);

  const renderMainContent = () => {
    switch (activeView) {
      case "definitions":
        return <DefinitionsView />;
      case "formulas":
        return <FormulasView />;
      case "search":
        return <SearchView onNavigateToSlide={navigateToSlide} />;
      case "visualizations":
        return <VisualizationsView />;
      case "sessions":
      default:
        return (
          <div className="sessions-content">
            <div className="content-toggle">
              <button
                className={`toggle-btn ${contentType === "overview" ? "active" : ""}`}
                onClick={() => setContentType("overview")}
              >
                Session Overview
              </button>
              <button
                className={`toggle-btn ${contentType === "slides" ? "active" : ""}`}
                onClick={() => setContentType("slides")}
              >
                Slide Viewer
              </button>
            </div>
            {contentType === "overview" ? (
              <SessionOverview onNavigateToSlide={navigateToSlide} />
            ) : (
              <SlideViewer />
            )}
          </div>
        );
    }
  };

  return (
    <div className="app">
      <Sidebar activeView={activeView} onViewChange={setActiveView} />
      <main className="main-content">{renderMainContent()}</main>
      <HelpGuide />
    </div>
  );
}

export default function App() {
  return (
    <NavigationProvider>
      <AppContent />
    </NavigationProvider>
  );
}
