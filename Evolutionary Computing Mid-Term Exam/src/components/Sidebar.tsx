import React from 'react';
import { Book, ChevronLeft, ChevronRight, Home, Search, BookOpen, Calculator, Play } from 'lucide-react';
import { useNavigation } from '../contexts/NavigationContext';
import { SESSIONS } from '../constants/sessions';

interface SidebarProps {
  activeView: 'sessions' | 'definitions' | 'formulas' | 'search' | 'visualizations';
  onViewChange: (view: 'sessions' | 'definitions' | 'formulas' | 'search' | 'visualizations') => void;
}

export default function Sidebar({ activeView, onViewChange }: SidebarProps) {
  const { 
    currentSession, 
    sidebarOpen, 
    toggleSidebar, 
    goToSession 
  } = useNavigation();

  return (
    <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
      <button 
        className="sidebar-toggle"
        onClick={toggleSidebar}
        aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
      >
        {sidebarOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
      </button>

      {sidebarOpen && (
        <div className="sidebar-content">
          <div className="sidebar-header">
            <Book className="text-primary-400" size={24} />
            <h1 className="sidebar-title">EC Course</h1>
          </div>

          <nav className="sidebar-nav">
            <button
              className={`nav-link ${activeView === 'sessions' ? 'active' : ''}`}
              onClick={() => onViewChange('sessions')}
            >
              <Home size={18} />
              <span>Sessions</span>
            </button>
            <button
              className={`nav-link ${activeView === 'definitions' ? 'active' : ''}`}
              onClick={() => onViewChange('definitions')}
            >
              <BookOpen size={18} />
              <span>Definitions</span>
            </button>
            <button
              className={`nav-link ${activeView === 'formulas' ? 'active' : ''}`}
              onClick={() => onViewChange('formulas')}
            >
              <Calculator size={18} />
              <span>Formulas</span>
            </button>
            <button
              className={`nav-link ${activeView === 'search' ? 'active' : ''}`}
              onClick={() => onViewChange('search')}
            >
              <Search size={18} />
              <span>Search</span>
            </button>
            <button
              className={`nav-link ${activeView === 'visualizations' ? 'active' : ''}`}
              onClick={() => onViewChange('visualizations')}
            >
              <Play size={18} />
              <span>Visualizations</span>
            </button>
          </nav>

          <div className="session-list">
            <h3 className="session-list-title">Sessions</h3>
            <ul className="session-items">
              {SESSIONS.map((session) => (
                <li key={session.id}>
                  <button
                    className={`session-item ${currentSession === session.id ? 'active' : ''}`}
                    onClick={() => goToSession(session.id)}
                  >
                    <span className="session-number">{session.id}</span>
                    <span className="session-name">{session.title}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </aside>
  );
}
