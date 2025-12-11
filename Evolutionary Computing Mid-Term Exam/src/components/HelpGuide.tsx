import React, { useState } from 'react';
import { HelpCircle, X, ChevronRight, Book, Layers, Calculator, Search, Play, MousePointer } from 'lucide-react';

interface GuideSection {
  id: string;
  title: string;
  icon: React.ReactNode;
  content: React.ReactNode;
}

export default function HelpGuide() {
  const [isOpen, setIsOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('overview');

  const sections: GuideSection[] = [
    {
      id: 'overview',
      title: 'Overview',
      icon: <Book size={18} />,
      content: (
        <div className="guide-content">
          <h3>Welcome to the Evolutionary Computing Course</h3>
          <p>
            This interactive application helps you learn Evolutionary Computing concepts 
            through detailed explanations, definitions, formulas, and interactive visualizations.
          </p>
          
          <h4>Main Features:</h4>
          <ul>
            <li><strong>Sessions:</strong> Browse through 14 course sessions with detailed slide explanations</li>
            <li><strong>Definitions:</strong> Search and explore key EC terminology</li>
            <li><strong>Formulas:</strong> View mathematical formulas with LaTeX rendering and derivations</li>
            <li><strong>Search:</strong> Find any topic across all content</li>
            <li><strong>Visualizations:</strong> Interactive simulations of GA concepts</li>
          </ul>

          <div className="tip-box">
            <strong>💡 Quick Tip:</strong> Use the sidebar to navigate between sessions. 
            Click any session to see its overview, then switch to "Slide Viewer" to see individual slides.
          </div>
        </div>
      )
    },
    {
      id: 'sessions',
      title: 'Sessions & Slides',
      icon: <Layers size={18} />,
      content: (
        <div className="guide-content">
          <h3>How to Use Sessions</h3>
          
          <h4>Session Overview</h4>
          <p>When you click on a session, you'll see:</p>
          <ul>
            <li><strong>Summary:</strong> Brief overview of the session's content</li>
            <li><strong>Key Takeaway:</strong> The most important point to remember</li>
            <li><strong>Topics Covered:</strong> List of subjects in this session</li>
            <li><strong>Slide Cards:</strong> Click any slide to view it in detail</li>
          </ul>

          <h4>Slide Viewer</h4>
          <p>Toggle to "Slide Viewer" mode to see:</p>
          <ul>
            <li><strong>Original Content:</strong> Exact text from the lecture slides</li>
            <li><strong>Detailed Explanation:</strong> Comprehensive explanation with examples</li>
            <li><strong>Key Points:</strong> Bullet summary of main ideas</li>
            <li><strong>Professor's Notes:</strong> Critical insights emphasized by the instructor</li>
            <li><strong>Related Definitions & Formulas:</strong> Linked references</li>
          </ul>

          <h4>Navigation</h4>
          <ul>
            <li>Use <strong>Previous/Next</strong> buttons to move between slides</li>
            <li>Click the <strong>progress bar</strong> to jump to specific positions</li>
            <li>Use the <strong>sidebar</strong> to jump to any session</li>
          </ul>

          <div className="tip-box">
            <strong>💡 Tip:</strong> Look for the yellow "Professor's Note" boxes—these contain 
            the most critical exam-worthy insights!
          </div>
        </div>
      )
    },
    {
      id: 'definitions',
      title: 'Definitions',
      icon: <Book size={18} />,
      content: (
        <div className="guide-content">
          <h3>How to Use Definitions</h3>
          
          <p>The Definitions section contains all key terminology from the course.</p>

          <h4>Features:</h4>
          <ul>
            <li><strong>Search:</strong> Type in the search box to filter definitions</li>
            <li><strong>Click to Expand:</strong> Click any term to see full details</li>
            <li><strong>Professor's Emphasis:</strong> Special notes on what the instructor stressed</li>
            <li><strong>Examples:</strong> Concrete examples to understand the concept</li>
            <li><strong>Common Misconceptions:</strong> Mistakes to avoid</li>
            <li><strong>Related Terms:</strong> Click to navigate to related definitions</li>
          </ul>

          <h4>Key Definitions to Know:</h4>
          <ul>
            <li><strong>Fitness:</strong> We know CORRECT (ranking), not EXACT (values)</li>
            <li><strong>Crossover:</strong> Explores (large steps), cannot create new alleles</li>
            <li><strong>Mutation:</strong> Exploits (small steps), ONLY creates new alleles</li>
            <li><strong>Schema:</strong> Template over {'{0,1,#}'} representing a hyperplane</li>
            <li><strong>Building Block:</strong> Short, low-order, above-average schemata</li>
          </ul>

          <div className="tip-box">
            <strong>💡 Tip:</strong> The "Professor's Emphasis" sections often contain 
            exactly what will be asked on exams!
          </div>
        </div>
      )
    },
    {
      id: 'formulas',
      title: 'Formulas',
      icon: <Calculator size={18} />,
      content: (
        <div className="guide-content">
          <h3>How to Use Formulas</h3>
          
          <p>Mathematical formulas are rendered with LaTeX for clear display.</p>

          <h4>Each Formula Includes:</h4>
          <ul>
            <li><strong>LaTeX Display:</strong> Beautifully rendered mathematical notation</li>
            <li><strong>Plain English:</strong> What the formula means in words</li>
            <li><strong>Variables:</strong> Explanation of each symbol</li>
            <li><strong>Derivation:</strong> Step-by-step derivation (when applicable)</li>
            <li><strong>When to Use:</strong> Practical application guidance</li>
            <li><strong>Example:</strong> Worked example with numbers</li>
          </ul>

          <h4>Key Formulas to Master:</h4>
          <ul>
            <li><strong>Expected Copies (FPS):</strong> E(n_i) = μ × f(i)/⟨f⟩</li>
            <li><strong>Schema Theorem:</strong> The three-factor growth equation</li>
            <li><strong>Tournament Selection:</strong> ŝ(f_i) formula using cumulative distribution</li>
            <li><strong>Linear Ranking:</strong> Probability based on rank, not fitness</li>
          </ul>

          <div className="tip-box">
            <strong>💡 Tip:</strong> Click "Related Formulas" to see connections between 
            different mathematical concepts.
          </div>
        </div>
      )
    },
    {
      id: 'search',
      title: 'Search',
      icon: <Search size={18} />,
      content: (
        <div className="guide-content">
          <h3>How to Use Search</h3>
          
          <p>Search across all course content—slides, definitions, and formulas.</p>

          <h4>Search Tips:</h4>
          <ul>
            <li>Use <strong>specific terms</strong> like "tournament selection" or "schema theorem"</li>
            <li>Search by <strong>concept</strong>: "exploration", "exploitation", "diversity"</li>
            <li>Find <strong>formulas</strong> by searching their names or variables</li>
            <li>Results show which <strong>type</strong> (slide/definition/formula) they are</li>
          </ul>

          <h4>Result Types:</h4>
          <ul>
            <li><strong>📄 Slides:</strong> Click to go directly to that slide</li>
            <li><strong>📖 Definitions:</strong> Click to see the full definition</li>
            <li><strong>🔢 Formulas:</strong> Click to see the formula with derivation</li>
          </ul>

          <div className="tip-box">
            <strong>💡 Tip:</strong> Try searching for "professor" to find all instructor 
            emphasis points, or "CRITICAL" for key insights!
          </div>
        </div>
      )
    },
    {
      id: 'visualizations',
      title: 'Visualizations',
      icon: <Play size={18} />,
      content: (
        <div className="guide-content">
          <h3>Interactive Visualizations</h3>
          
          <p>Learn by experimenting with these interactive simulations!</p>

          <h4>GA Simulation (OneMax)</h4>
          <ul>
            <li>Watch a genetic algorithm evolve a population of binary strings</li>
            <li>Goal: Find a string of all 1s (maximum fitness)</li>
            <li>Adjust population size, mutation rate, crossover rate</li>
            <li>See the fitness graph rise over generations</li>
          </ul>

          <h4>Selection Comparison</h4>
          <ul>
            <li>Compare FPS, Rank-based, and Tournament selection</li>
            <li>See why FPS is problematic (uses EXACT fitness)</li>
            <li>Understand why Tournament is robust (uses only comparisons)</li>
            <li>Adjust tournament size and see the effect on selection pressure</li>
          </ul>

          <h4>Crossover Visualization</h4>
          <ul>
            <li>See how different crossover operators work</li>
            <li>Compare One-Point, Two-Point, Uniform, and PMX</li>
            <li>Understand positional and distributional bias</li>
          </ul>

          <h4>Fitness Landscape</h4>
          <ul>
            <li>Visualize populations evolving on different landscapes</li>
            <li>Try: Unimodal (easy), Multimodal, Deceptive (hard), Rugged</li>
            <li>See how GA handles local optima</li>
          </ul>

          <h4>Schema Theorem</h4>
          <ul>
            <li>Enter any schema and see its properties calculated</li>
            <li>Understand order, defining length, and growth rate</li>
            <li>Check if a schema qualifies as a building block</li>
          </ul>

          <div className="tip-box">
            <strong>💡 Tip:</strong> The Deceptive landscape visualization perfectly 
            demonstrates why "more domain knowledge can hurt GA performance"—watch 
            the population get trapped!
          </div>
        </div>
      )
    },
    {
      id: 'shortcuts',
      title: 'Keyboard Shortcuts',
      icon: <MousePointer size={18} />,
      content: (
        <div className="guide-content">
          <h3>Keyboard Shortcuts</h3>
          
          <table className="shortcuts-table">
            <thead>
              <tr>
                <th>Key</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><kbd>←</kbd> / <kbd>→</kbd></td>
                <td>Previous / Next slide</td>
              </tr>
              <tr>
                <td><kbd>1</kbd> - <kbd>9</kbd></td>
                <td>Jump to session 1-9</td>
              </tr>
              <tr>
                <td><kbd>/</kbd> or <kbd>Ctrl+K</kbd></td>
                <td>Focus search</td>
              </tr>
              <tr>
                <td><kbd>Esc</kbd></td>
                <td>Close help / Clear search</td>
              </tr>
              <tr>
                <td><kbd>?</kbd></td>
                <td>Toggle this help guide</td>
              </tr>
            </tbody>
          </table>

          <div className="tip-box">
            <strong>💡 Tip:</strong> Use arrow keys to quickly navigate through slides 
            while studying!
          </div>
        </div>
      )
    }
  ];

  const activeContent = sections.find(s => s.id === activeSection)?.content;

  return (
    <>
      <button className="help-trigger" onClick={() => setIsOpen(true)} title="Help">
        <HelpCircle size={24} />
      </button>

      {isOpen && (
        <div className="help-overlay" onClick={() => setIsOpen(false)}>
          <div className="help-modal" onClick={e => e.stopPropagation()}>
            <div className="help-header">
              <h2>
                <HelpCircle size={24} />
                How to Use This App
              </h2>
              <button className="close-btn" onClick={() => setIsOpen(false)}>
                <X size={24} />
              </button>
            </div>

            <div className="help-body">
              <nav className="help-nav">
                {sections.map(section => (
                  <button
                    key={section.id}
                    className={`help-nav-item ${activeSection === section.id ? 'active' : ''}`}
                    onClick={() => setActiveSection(section.id)}
                  >
                    {section.icon}
                    <span>{section.title}</span>
                    <ChevronRight size={16} className="nav-arrow" />
                  </button>
                ))}
              </nav>

              <div className="help-content">
                {activeContent}
              </div>
            </div>

            <div className="help-footer">
              <p>Press <kbd>?</kbd> anytime to toggle this help guide</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
