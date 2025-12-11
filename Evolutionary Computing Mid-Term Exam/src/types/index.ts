export interface Session {
  id: number;
  title: string;
  slideRange: [number, number];
  summary: string;
  keyTakeaway: string;
  topics: string[];
}

export interface Slide {
  number: number;
  title: string;
  session: number;
  content: string;
  explanation: string;
  keyPoints: string[];
  formulas?: string[];
  definitions?: string[];
  professorNote?: string;
}

export interface Definition {
  term: string;
  definition: string;
  professorEmphasis?: string;
  relatedTerms?: string[];
  examples?: string[];
  commonMisconceptions?: string[];
}

export interface Formula {
  id: string;
  name: string;
  latex: string;
  latexDisplay: string;
  variables: Record<string, string>;
  derivation?: string[];
  plainEnglish: string;
  whenToUse: string;
  example?: string;
  relatedFormulas?: string[];
}

export interface NavigationState {
  currentSession: number;
  currentSlide: number;
  sidebarOpen: boolean;
}
