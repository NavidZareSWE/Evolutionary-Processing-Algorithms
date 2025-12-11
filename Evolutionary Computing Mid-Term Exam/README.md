# 🧬 Evolutionary Computing Course - Interactive Learning Platform

An interactive React application for studying Evolutionary Computing concepts, featuring detailed slide explanations, searchable definitions, mathematical formulas with LaTeX rendering, and hands-on visualizations.

---

## 📖 Overview

This application transforms a 248-slide Evolutionary Computing course into an interactive learning experience. Based on lectures by Dr. Ali Hamzeh at Shiraz University, it provides comprehensive explanations, professor insights, and interactive simulations to help students master EC concepts.

### Why This App?

- 📚 **Deep Understanding**: Go beyond slides with detailed explanations for every concept
- 🎯 **Exam Preparation**: Professor's notes highlight what's truly important
- 🔬 **Learn by Doing**: Interactive visualizations let you experiment with GA parameters
- 🔍 **Quick Reference**: Search across all content instantly
- 📐 **Math Made Clear**: LaTeX rendering with step-by-step derivations

---

## ✨ Features

### 📑 Session-Based Learning

- **14 Complete Sessions** covering the full EC curriculum
- **248 Detailed Slides** with original content + comprehensive explanations
- **Key Points** summarizing each slide's main ideas
- **Professor's Notes** highlighting exam-critical insights

### 📖 Definitions Library

- **20+ Key Terms** with full definitions
- **Professor's Emphasis** on what matters most
- **Examples** for concrete understanding
- **Common Misconceptions** to avoid
- **Related Terms** for connected learning

### 📐 Formula Reference

- **12+ Core Formulas** with LaTeX rendering
- **Plain English** explanations
- **Variable Definitions** for each symbol
- **Step-by-Step Derivations**
- **When to Use** guidance
- **Worked Examples** with numbers

### 🔬 Interactive Visualizations

| Visualization            | Description                                                                   |
| ------------------------ | ----------------------------------------------------------------------------- |
| **GA Simulation**        | Watch a genetic algorithm evolve to solve OneMax in real-time                 |
| **Selection Comparison** | Compare FPS, Rank, and Tournament selection methods                           |
| **Crossover Operators**  | Visualize One-Point, Two-Point, Uniform, and PMX crossover                    |
| **Fitness Landscape**    | Explore GA behavior on Unimodal, Multimodal, Deceptive, and Rugged landscapes |
| **Schema Theorem**       | Calculate schema properties and building block analysis                       |

### 🔍 Global Search

- Search across slides, definitions, and formulas
- Instant results with type indicators
- Click to navigate directly to content

### ⌨️ Keyboard Shortcuts

| Key       | Action                |
| --------- | --------------------- |
| `←` / `→` | Previous / Next slide |
| `1` - `9` | Jump to session 1-9   |
| `/`       | Focus search          |
| `?`       | Toggle help guide     |

---

## 🚀 Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

```bash
# Clone or extract the project
cd ec-course-complete

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build for Production

```bash
npm run build
```

Output will be in the `dist/` folder.

---

## 📁 Project Structure

```
ec-course-complete/
├── index.html                 # Entry HTML with MathJax CDN
├── package.json               # Dependencies and scripts
├── vite.config.ts            # Vite configuration with path aliases
├── tsconfig.json             # TypeScript configuration
├── tailwind.config.js        # Tailwind theme customization
├── postcss.config.js         # PostCSS plugins
│
└── src/
    ├── main.tsx              # React entry point
    ├── App.tsx               # Main app with routing
    ├── index.css             # Global styles (2000+ lines)
    │
    ├── types/
    │   └── index.ts          # TypeScript interfaces
    │
    ├── contexts/
    │   └── NavigationContext.tsx  # Global navigation state
    │
    ├── constants/
    │   ├── sessions.ts       # 14 session definitions
    │   ├── slides.ts         # 248 slide contents
    │   ├── definitions.ts    # 20+ term definitions
    │   └── formulas.ts       # 12+ mathematical formulas
    │
    └── components/
        ├── Sidebar.tsx           # Navigation sidebar
        ├── SessionOverview.tsx   # Session summary view
        ├── SlideViewer.tsx       # Individual slide display
        ├── DefinitionsView.tsx   # Searchable definitions
        ├── FormulasView.tsx      # Formula reference
        ├── SearchView.tsx        # Global search
        ├── HelpGuide.tsx         # Interactive help modal
        ├── MathBlock.tsx         # LaTeX rendering component
        ├── VisualizationsView.tsx # Visualization hub
        │
        └── visualizations/
            ├── GASimulation.tsx          # OneMax GA simulator
            ├── SelectionComparison.tsx   # Selection methods demo
            ├── CrossoverVisualization.tsx # Crossover operators
            ├── FitnessLandscape.tsx      # 2D landscape explorer
            └── SchemaVisualization.tsx   # Schema theorem calculator
```

---

## 📚 Course Content

### Session Overview

| Session | Title                       | Slides  | Key Topics                                       |
| ------- | --------------------------- | ------- | ------------------------------------------------ |
| 1       | Introduction                | 1-10    | Intelligence, AI approaches, Optimization        |
| 2       | Bio-Inspired Computing      | 11-19   | Natural selection, Three operators               |
| 3       | Four EA Brands              | 20-31   | GA, GP, ES, EP frameworks                        |
| 4       | Fitness & Population        | 32-47   | Fitness functions, Diversity hierarchy           |
| 5       | 8-Queens Example            | 48-55   | Problem formulation, Representations             |
| 6       | Simple GA & Binary          | 56-70   | Binary encoding, FPS, Gray coding                |
| 7       | Permutation Representations | 71-99   | Order-1, PMX, Cycle, Edge crossover              |
| 8       | Mutation Operators          | 100-114 | Mutation types, Exploration vs Exploitation      |
| 9       | Multi-Parent & Population   | 115-131 | k-parent crossover, Generational vs Steady-state |
| 10      | Selection Mechanisms        | 132-155 | Tournament, Ranking, FPS problems                |
| 11      | Schema Theory               | 156-177 | Schema Theorem, Building Block Hypothesis        |
| 12      | Royal Road Functions        | 178-201 | BBH testing, Counter-examples                    |
| 13      | GA Difficulty & Deception   | 202-216 | Deceptive problems, GA-hard                      |
| 14      | Selection Analysis          | 217-248 | Selection pressure, Loss of diversity            |

### Key Concepts Explained

#### 🎯 CORRECT vs EXACT Fitness

> "We know CORRECT fitness (rankings/comparisons), not EXACT fitness (numerical values). This is why FPS is problematic and Tournament selection is robust."

#### 🔄 Crossover vs Mutation Roles

| Aspect              | Crossover               | Mutation             |
| ------------------- | ----------------------- | -------------------- |
| Role                | **EXPLORATIVE**         | **EXPLOITATIVE**     |
| Step Size           | Large (between parents) | Small (local)        |
| Creates New Alleles | ❌ No                   | ✅ Yes (ONLY source) |
| Search Character    | Global combination      | Local fine-tuning    |

#### 🧱 Building Block Hypothesis

Building blocks are schemata that are:

- **Short** (small defining length δ)
- **Low-order** (few defined positions)
- **Above-average** fitness

The GA discovers and combines these into complete solutions.

#### 📊 Diversity Hierarchy

```
Fitness Diversity → Phenotype Diversity → Genotype Diversity
        ↓                    ↓                    ↓
   (implies)            (implies)           (does NOT
                                            imply reverse)
```

---

## 🛠️ Technologies

| Technology       | Purpose                       |
| ---------------- | ----------------------------- |
| **React 18**     | UI framework with hooks       |
| **TypeScript**   | Type safety                   |
| **Vite**         | Fast development and building |
| **Tailwind CSS** | Utility-first styling         |
| **Lucide React** | Modern icon set               |
| **MathJax 3**    | LaTeX formula rendering       |

---

## 🎨 Design System

### Color Palette

| Color       | Usage                       | Hex                   |
| ----------- | --------------------------- | --------------------- |
| **Gold**    | Primary actions, highlights | `#fbbf24`             |
| **Blue**    | Links, secondary actions    | `#3b82f6`             |
| **Purple**  | Accents, tags               | `#8b5cf6`             |
| **Green**   | Success, positive           | `#22c55e`             |
| **Surface** | Backgrounds                 | `#0f0f1a` - `#1a1a2e` |

### Animations

- Slide-up entrance animations
- Scale-in for cards
- Gold shimmer on progress bar
- Glow effect on Professor's Notes
- Smooth hover transitions

---

## 📱 Responsive Design

The application is fully responsive:

- **Desktop**: Full sidebar + content layout
- **Tablet**: Collapsible sidebar
- **Mobile**: Stacked layout with hamburger menu

---

## 🧪 Testing the Visualizations

### GA Simulation (OneMax)

1. Click "Run" to start evolution
2. Watch the population converge to all 1s
3. Adjust mutation rate to see its effect on diversity
4. Try different tournament sizes

### Selection Comparison

1. Click "Run 100 Selections"
2. Compare how each method distributes selections
3. Notice FPS over-selects the fittest individual
4. See how Tournament only uses comparisons

### Fitness Landscape

1. Select "Deceptive" landscape
2. Click "Run" and watch the population
3. Notice how it gets trapped in the center
4. This demonstrates why some problems are "GA-hard"

---

## 📝 Usage Tips

### For Studying

1. Start with **Session Overview** to understand the topic
2. Switch to **Slide Viewer** for detailed explanations
3. Look for **Professor's Notes** (yellow boxes) - these are exam gold
4. Use **Key Points** as quick summaries

### For Quick Reference

1. Use **Search** to find any topic instantly
2. Browse **Definitions** for terminology
3. Check **Formulas** for mathematical details

### For Deep Understanding

1. Experiment with **Visualizations**
2. Try the **Schema Calculator** with different schemata
3. Watch GA behavior on different **Fitness Landscapes**

---

## 🔧 Customization

### Adding New Slides

Edit `src/constants/slides.ts`:

```typescript
{
  number: 249,
  title: "Your New Slide",
  session: 14,
  content: "Original slide content...",
  explanation: "Detailed explanation...",
  keyPoints: [
    "Point 1 as a complete, meaningful sentence",
    "Point 2 explaining another key concept"
  ],
  definitions: ["relatedTerm"],
  formulas: ["relatedFormula"],
  professorNote: "Critical insight for exams"
}
```

### Adding New Definitions

Edit `src/constants/definitions.ts`:

```typescript
newTerm: {
  term: "New Term",
  definition: "Clear definition...",
  professorEmphasis: "What the professor stressed...",
  relatedTerms: ["related1", "related2"],
  examples: ["Example 1", "Example 2"],
  commonMisconceptions: ["Misconception to avoid"]
}
```

### Adding New Formulas

Edit `src/constants/formulas.ts`:

```typescript
{
  id: "newFormula",
  name: "Formula Name",
  latex: "E = mc^2",
  latexDisplay: "E = mc^2",
  plainEnglish: "Energy equals mass times speed of light squared",
  variables: [
    { symbol: "E", meaning: "Energy" },
    { symbol: "m", meaning: "Mass" }
  ],
  derivation: ["Step 1...", "Step 2..."],
  whenToUse: "When calculating energy...",
  example: "For m=1kg: E = 1 × (3×10⁸)² = 9×10¹⁶ J"
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional visualizations (ES, GP simulations)
- More worked examples
- Practice problems / quizzes
- Additional course sessions

---

## 📄 License

This project is for educational purposes. Course content is based on lectures by Dr. Ali Hamzeh at Shiraz University.

---

## 🙏 Acknowledgments

- **Dr. Ali Hamzeh** - Original course lectures (Shiraz University)
- **Holland, J.H.** - Schema Theorem and GA foundations
- **Eiben & Smith** - "Introduction to Evolutionary Computing" textbook
- **Goldberg, D.E.** - Building Block Hypothesis research

---

## 📞 Support

If you encounter issues:

1. Check the **Help Guide** (click `?` button or press `?` key)
2. Ensure all dependencies are installed (`npm install`)
3. Try clearing browser cache and reloading

---

<div align="center">

**Happy Learning! 🧬**

_"Evolution is cleverer than you are."_ - Orgel's Second Rule

</div>
