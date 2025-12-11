import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  ReactNode,
} from "react";
import { SESSIONS } from "../constants/sessions";
import { SLIDES } from "../constants/slides";

interface NavigationContextType {
  currentSession: number;
  currentSlide: number;
  sidebarOpen: boolean;
  setCurrentSession: (session: number) => void;
  setCurrentSlide: (slide: number) => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  goToNextSlide: () => void;
  goToPrevSlide: () => void;
  goToNextSession: () => void;
  goToPrevSession: () => void;
  goToSlide: (slideNum: number) => void;
  goToSession: (sessionId: number) => void;
  getSessionForSlide: (slideNum: number) => number;
  isFirstSlide: boolean;
  isLastSlide: boolean;
  isFirstSession: boolean;
  isLastSession: boolean;
  totalSlides: number;
  totalSessions: number;
}

const NavigationContext = createContext<NavigationContextType | undefined>(
  undefined,
);

export function NavigationProvider({ children }: { children: ReactNode }) {
  const [currentSession, setCurrentSession] = useState(1);
  const [currentSlide, setCurrentSlide] = useState(1);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const totalSlides = SLIDES.length;
  const totalSessions = SESSIONS.length;

  const getSessionForSlide = useCallback((slideNum: number): number => {
    const slide = SLIDES.find((s) => s.number === slideNum);
    return slide?.session || 1;
  }, []);

  const goToSlide = useCallback(
    (slideNum: number) => {
      if (slideNum >= 1 && slideNum <= totalSlides) {
        setCurrentSlide(slideNum);
        setCurrentSession(getSessionForSlide(slideNum));
      }
    },
    [totalSlides, getSessionForSlide],
  );

  const goToSession = useCallback(
    (sessionId: number) => {
      if (sessionId >= 1 && sessionId <= totalSessions) {
        const session = SESSIONS.find((s) => s.id === sessionId);
        if (session) {
          setCurrentSession(sessionId);
          setCurrentSlide(session.slideRange[0]);
        }
      }
    },
    [totalSessions],
  );

  const goToNextSlide = useCallback(() => {
    if (currentSlide < totalSlides) {
      goToSlide(currentSlide + 1);
    }
  }, [currentSlide, totalSlides, goToSlide]);

  const goToPrevSlide = useCallback(() => {
    if (currentSlide > 1) {
      goToSlide(currentSlide - 1);
    }
  }, [currentSlide, goToSlide]);

  const goToNextSession = useCallback(() => {
    if (currentSession < totalSessions) {
      goToSession(currentSession + 1);
    }
  }, [currentSession, totalSessions, goToSession]);

  const goToPrevSession = useCallback(() => {
    if (currentSession > 1) {
      goToSession(currentSession - 1);
    }
  }, [currentSession, goToSession]);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  const value: NavigationContextType = {
    currentSession,
    currentSlide,
    sidebarOpen,
    setCurrentSession,
    setCurrentSlide,
    setSidebarOpen,
    toggleSidebar,
    goToNextSlide,
    goToPrevSlide,
    goToNextSession,
    goToPrevSession,
    goToSlide,
    goToSession,
    getSessionForSlide,
    isFirstSlide: currentSlide === 1,
    isLastSlide: currentSlide === totalSlides,
    isFirstSession: currentSession === 1,
    isLastSession: currentSession === totalSessions,
    totalSlides,
    totalSessions,
  };

  return (
    <NavigationContext.Provider value={value}>
      {children}
    </NavigationContext.Provider>
  );
}

export function useNavigation() {
  const context = useContext(NavigationContext);
  if (context === undefined) {
    throw new Error("useNavigation must be used within a NavigationProvider");
  }
  return context;
}
