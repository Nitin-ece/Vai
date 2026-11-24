"use client";

import React, { useState, useEffect } from "react";

export interface PanelOverlay {
  id: string;
  panel_index: number;
  reading_order: number;
  bbox: [number, number, number, number]; // [x0, y0, x1, y1]
  source_png: string;
}

export interface PageItem {
  page_number: number;
  title: string;
  image_url: string;
  width: number;
  height: number;
  panels: PanelOverlay[];
}

interface PageBrowserProps {
  projectId: string;
  onBack: () => void;
  onSelectPanel: (panelId: string) => void;
}

export function PageBrowser({ projectId, onBack, onSelectPanel }: PageBrowserProps) {
  const [pages, setPages] = useState<PageItem[]>([]);
  const [selectedPageIndex, setSelectedPageIndex] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [hoveredPanelId, setHoveredPanelId] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetch(`http://localhost:8000/api/v1/projects/${projectId}/pages`)
      .then((res) => res.json())
      .then((data) => {
        setPages(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load pages:", err);
        setLoading(false);
      });
  }, [projectId]);

  const currentPage = pages[selectedPageIndex];

  return (
    <div className="flex flex-col flex-1 h-full bg-slate-950 overflow-hidden">
      {/* Top Header */}
      <header className="h-14 border-b border-slate-800 bg-slate-900/60 backdrop-blur-md px-6 flex items-center justify-between z-20">
        <div className="flex items-center space-x-4">
          <button
            onClick={onBack}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 font-medium transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            <span>Projects</span>
          </button>

          <div className="h-4 w-px bg-slate-800" />

          <div>
            <h2 className="text-sm font-bold text-slate-200 capitalize">
              {projectId.replace("_", " ")} &bull; Page Browser
            </h2>
          </div>
        </div>

        {/* Page Switcher */}
        {pages.length > 0 && (
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400 font-mono">Page:</span>
            <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
              {pages.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedPageIndex(idx)}
                  className={`px-3 py-1 text-xs font-mono rounded ${
                    selectedPageIndex === idx
                      ? "bg-indigo-600 text-white font-bold"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                  }`}
                >
                  {p.page_number}
                </button>
              ))}
            </div>
          </div>
        )}
      </header>

      {/* Main Page Area */}
      <div className="flex flex-1 overflow-hidden">
        {loading ? (
          <div className="flex-1 flex flex-col items-center justify-center">
            <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            <p className="mt-4 text-xs font-mono uppercase tracking-widest text-slate-400">Loading Manga Page...</p>
          </div>
        ) : currentPage ? (
          <div className="flex-1 flex overflow-hidden">
            {/* Page View with Overlays */}
            <div className="flex-1 relative flex items-center justify-center p-8 bg-slate-950 overflow-auto">
              <div
                className="relative shadow-2xl rounded border border-slate-800 bg-white"
                style={{
                  maxWidth: "90%",
                  maxHeight: "90%",
                  aspectRatio: `${currentPage.width} / ${currentPage.height}`,
                }}
              >
                {/* Base Page Image */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={currentPage.image_url.startsWith("http") ? currentPage.image_url : `http://localhost:8000${currentPage.image_url}`}
                  alt={currentPage.title}
                  className="w-full h-full object-contain block"
                />

                {/* SVG Panel Overlays */}
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox={`0 0 ${currentPage.width} ${currentPage.height}`}
                >
                  {currentPage.panels.map((p) => {
                    const [x0, y0, x1, y1] = p.bbox;
                    const pw = x1 - x0;
                    const ph = y1 - y0;
                    const isHovered = hoveredPanelId === p.id;

                    return (
                      <g key={p.id}>
                        {/* Bounding box rect */}
                        <rect
                          x={x0}
                          y={y0}
                          width={pw}
                          height={ph}
                          fill={isHovered ? "rgba(99, 102, 241, 0.25)" : "rgba(99, 102, 241, 0.08)"}
                          stroke={isHovered ? "#818cf8" : "#6366f1"}
                          strokeWidth={isHovered ? 6 : 3}
                          strokeDasharray={isHovered ? "none" : "8,6"}
                          className="transition-all duration-200"
                        />

                        {/* Reading order badge (RTL) */}
                        <circle
                          cx={x0 + 36}
                          cy={y0 + 36}
                          r={24}
                          fill="#4f46e5"
                          stroke="#ffffff"
                          strokeWidth={3}
                        />
                        <text
                          x={x0 + 36}
                          y={y0 + 44}
                          textAnchor="middle"
                          fill="#ffffff"
                          fontSize="22"
                          fontWeight="bold"
                          fontFamily="monospace"
                        >
                          {p.reading_order}
                        </text>
                      </g>
                    );
                  })}
                </svg>

                {/* Clickable Hover Targets */}
                {currentPage.panels.map((p) => {
                  const [x0, y0, x1, y1] = p.bbox;
                  const leftPct = (x0 / currentPage.width) * 100;
                  const topPct = (y0 / currentPage.height) * 100;
                  const widthPct = ((x1 - x0) / currentPage.width) * 100;
                  const heightPct = ((y1 - y0) / currentPage.height) * 100;

                  return (
                    <div
                      key={p.id}
                      onClick={() => onSelectPanel(p.id)}
                      onMouseEnter={() => setHoveredPanelId(p.id)}
                      onMouseLeave={() => setHoveredPanelId(null)}
                      style={{
                        left: `${leftPct}%`,
                        top: `${topPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                      }}
                      className="absolute cursor-pointer flex items-center justify-center group"
                    >
                      <button className="opacity-0 group-hover:opacity-100 transition-opacity px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg shadow-xl text-xs font-bold tracking-wide flex items-center space-x-2 pointer-events-auto transform group-hover:scale-105 duration-200">
                        <span>Reconstruct Panel</span>
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right Panel Index Sidebar */}
            <aside className="w-72 border-l border-slate-800 bg-slate-900/40 p-5 flex flex-col overflow-y-auto">
              <h3 className="text-xs uppercase font-mono font-bold tracking-wider text-slate-400 mb-4">
                Detected Panels (RTL Reading Order)
              </h3>

              <div className="space-y-3">
                {currentPage.panels.map((p) => (
                  <div
                    key={p.id}
                    onClick={() => onSelectPanel(p.id)}
                    onMouseEnter={() => setHoveredPanelId(p.id)}
                    onMouseLeave={() => setHoveredPanelId(null)}
                    className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                      hoveredPanelId === p.id
                        ? "bg-indigo-950/60 border-indigo-500/80 shadow-lg"
                        : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-7 h-7 rounded-full bg-indigo-600 text-white font-mono text-xs font-bold flex items-center justify-center">
                        {p.reading_order}
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-slate-200">{p.id}</div>
                        <div className="text-[11px] font-mono text-slate-500">
                          {p.bbox[2] - p.bbox[0]} &times; {p.bbox[3] - p.bbox[1]} px
                        </div>
                      </div>
                    </div>

                    <button className="text-indigo-400 hover:text-indigo-300 text-xs font-medium">
                      Open &rarr;
                    </button>
                  </div>
                ))}
              </div>
            </aside>
          </div>
        ) : null}
      </div>
    </div>
  );
}
