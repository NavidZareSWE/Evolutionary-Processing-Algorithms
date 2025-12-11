import React, { useEffect, useRef } from 'react';

interface MathBlockProps {
  latex: string;
  display?: boolean;
  className?: string;
}

declare global {
  interface Window {
    MathJax: {
      typesetPromise: (elements?: HTMLElement[]) => Promise<void>;
      startup: {
        promise: Promise<void>;
      };
    };
  }
}

export default function MathBlock({ latex, display = false, className = '' }: MathBlockProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current && window.MathJax) {
      window.MathJax.startup.promise.then(() => {
        if (containerRef.current) {
          window.MathJax.typesetPromise([containerRef.current]);
        }
      });
    }
  }, [latex]);

  const mathContent = display ? `\\[${latex}\\]` : `\\(${latex}\\)`;

  return (
    <div 
      ref={containerRef} 
      className={`math-block ${display ? 'math-display' : 'math-inline'} ${className}`}
      dangerouslySetInnerHTML={{ __html: mathContent }}
    />
  );
}
