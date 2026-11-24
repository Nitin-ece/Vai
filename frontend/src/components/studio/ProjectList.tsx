"use client";

import React, { useState, useEffect, useRef } from "react";

export interface ProjectItem {
  id: string;
  name: string;
  type: "benchmark" | "project";
  pages_count: number;
  panels_count: number;
  description: string;
  thumbnail_url: string;
}

interface ProjectListProps {
  onSelectProject: (projectId: string) => void;
  onOpenViewer: (projectId: string, panelId: string) => void;
  gpuStatus: {
    device_name: string;
    cuda_available: boolean;
    vram_gb: number;
  } | null;
}

export function ProjectList({ onSelectProject, onOpenViewer, gpuStatus }: ProjectListProps) {
  const [projects, setProjects] = useState<ProjectItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [creatingDemo, setCreatingDemo] = useState<boolean>(false);

  // Drag and drop / upload state
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [fastMode, setFastMode] = useState<boolean>(true);
  const [uploadState, setUploadState] = useState<{
    active: boolean;
    stageText: string;
    progress: number;
    fileName: string;
    error: string | null;
  }>({
    active: false,
    stageText: "",
    progress: 0,
    fileName: "",
    error: null,
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchProjects = () => {
    setLoading(true);
    fetch("http://localhost:8000/api/v1/projects")
      .then((res) => res.json())
      .then((data) => {
        setProjects(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load projects:", err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  const handleCreateDemo = async () => {
    setCreatingDemo(true);
    try {
      const res = await fetch("http://localhost:8000/api/v1/projects/demo", { method: "POST" });
      if (res.ok) {
        fetchProjects();
      }
    } catch (e) {
      console.error("Failed to create demo:", e);
    } finally {
      setCreatingDemo(false);
    }
  };

  const handleProcessFile = async (file: File) => {
    if (!file) return;

    setUploadState({
      active: true,
      stageText: "Uploading manga file...",
      progress: 15,
      fileName: file.name,
      error: null,
    });

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", file.name.substring(0, file.name.lastIndexOf(".")) || file.name);
    formData.append("auto_process", "true");
    formData.append("fast_mode", fastMode ? "true" : "false");

    const stages = [
      { text: "Ingesting pages & detecting panels (RTL)...", progress: 30, delay: 1500 },
      { text: "Enhancing resolution & denoising...", progress: 50, delay: 3500 },
      { text: "Decomposing semantic layers & inpainting plate...", progress: 70, delay: 6500 },
      { text: "Colorizing vector layers & estimating depth...", progress: 85, delay: 9500 },
      { text: "Assembling 2.5D Three.js Scene...", progress: 95, delay: 12000 },
    ];

    const timeouts = stages.map((s) =>
      setTimeout(() => {
        setUploadState((prev) => {
          if (!prev.active || prev.error) return prev;
          return { ...prev, stageText: s.text, progress: s.progress };
        });
      }, s.delay)
    );

    try {
      const res = await fetch("http://localhost:8000/api/v1/projects/upload", {
        method: "POST",
        body: formData,
      });

      timeouts.forEach(clearTimeout);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(errData.detail || "Server error processing manga");
      }

      const data = await res.json();

      setUploadState((prev) => ({
        ...prev,
        stageText: "2.5D Scene Reconstructed! Launching Viewer...",
        progress: 100,
      }));

      setTimeout(() => {
        setUploadState((prev) => ({ ...prev, active: false }));
        onOpenViewer(data.project_id, data.scene_id || data.panel_id);
      }, 700);
    } catch (err: any) {
      timeouts.forEach(clearTimeout);
      console.error("Upload & Reconstruction error:", err);
      setUploadState((prev) => ({
        ...prev,
        error: err.message || "Failed to process manga file. Please try again.",
      }));
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleProcessFile(e.target.files[0]);
    }
  };

  return (
    <div className="flex flex-col flex-1 h-full overflow-y-auto bg-slate-950 p-8">
      {/* Hero Header */}
      <div className="max-w-6xl mx-auto w-full mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center space-x-3">
              <span>VAI — Volumetric Art Intelligence Studio</span>
              <span className="text-xs uppercase font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                v1.0
              </span>
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              Drag & drop any manga PDF, scan, or panel to instantly reconstruct and explore in 2.5D.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            {gpuStatus && (
              <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-mono">
                <span className={`w-2 h-2 rounded-full ${gpuStatus.cuda_available ? "bg-emerald-400" : "bg-amber-400"}`} />
                <span className="text-slate-300">
                  {gpuStatus.cuda_available ? `${gpuStatus.device_name} (${gpuStatus.vram_gb} GB)` : "CPU Fallback (Torch)"}
                </span>
              </div>
            )}

            <button
              onClick={handleCreateDemo}
              disabled={creatingDemo}
              className="px-4 py-2 bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-semibold rounded-lg transition-all flex items-center space-x-2 disabled:opacity-50"
            >
              {creatingDemo ? (
                <span>Generating Demo...</span>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  <span>Load Demo Manga</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Drag & Drop Manga Upload Zone */}
      <div className="max-w-6xl mx-auto w-full mb-8">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp"
          className="hidden"
        />

        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !uploadState.active && fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-2xl p-8 transition-all duration-300 cursor-pointer flex flex-col items-center justify-center text-center overflow-hidden group ${
            isDragOver
              ? "border-indigo-400 bg-indigo-950/40 shadow-2xl shadow-indigo-500/20 scale-[1.01]"
              : "border-slate-800 hover:border-indigo-500/50 bg-slate-900/40 hover:bg-slate-900/70"
          }`}
        >
          {/* Subtle Ambient Glow */}
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/5 via-purple-500/5 to-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />

          {uploadState.active ? (
            /* Upload / Processing Progress State */
            <div className="w-full max-w-md py-4 z-10 flex flex-col items-center">
              <div className="w-12 h-12 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin mb-4" />
              <h3 className="text-base font-bold text-white mb-1">
                {uploadState.fileName}
              </h3>
              <p className="text-xs font-mono text-indigo-400 mb-4 animate-pulse">
                {uploadState.stageText}
              </p>

              {/* Progress Bar */}
              <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                <div
                  className="bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-400 h-2.5 rounded-full transition-all duration-500"
                  style={{ width: `${uploadState.progress}%` }}
                />
              </div>
              <div className="w-full flex justify-between mt-2 text-[10px] font-mono text-slate-500">
                <span>Ingest & Detect</span>
                <span>Layers & Inpaint</span>
                <span>Depth & 2.5D</span>
              </div>

              {uploadState.error && (
                <div className="mt-4 p-3 rounded-lg bg-red-950/80 border border-red-800/80 text-red-200 text-xs">
                  <p className="font-semibold">Error:</p>
                  <p>{uploadState.error}</p>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setUploadState((prev) => ({ ...prev, active: false, error: null }));
                    }}
                    className="mt-2 px-3 py-1 bg-red-800 hover:bg-red-700 rounded text-[11px] font-mono"
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </div>
          ) : (
            /* Idle Drag & Drop Call to Action */
            <div className="z-10 flex flex-col items-center">
              <div
                className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 transition-transform group-hover:scale-110 shadow-lg ${
                  isDragOver
                    ? "bg-indigo-500 text-white shadow-indigo-500/50"
                    : "bg-indigo-600/10 border border-indigo-500/20 text-indigo-400 group-hover:bg-indigo-600/20"
                }`}
              >
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>

              <h2 className="text-lg font-bold text-white tracking-tight">
                Drag & drop your Manga here
              </h2>
              <p className="text-xs text-slate-400 mt-1 max-w-md">
                Drop any Manga PDF chapter, raw page scan, or cropped panel image (PNG, JPG, WEBP).
                The engine will automatically ingest, layer, colorize, and launch the 2.5D viewer.
              </p>

              <div className="flex items-center space-x-4 mt-5">
                <button
                  type="button"
                  className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white text-xs font-semibold rounded-lg shadow-lg shadow-indigo-600/20 transition-all flex items-center space-x-2"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  <span>Select Manga File</span>
                </button>

                <div
                  onClick={(e) => e.stopPropagation()}
                  className="flex items-center space-x-2 bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-300"
                >
                  <label htmlFor="fastModeToggle" className="cursor-pointer select-none">Fast 2x Mode</label>
                  <input
                    id="fastModeToggle"
                    type="checkbox"
                    checked={fastMode}
                    onChange={(e) => setFastMode(e.target.checked)}
                    className="rounded bg-slate-900 border-slate-700 text-indigo-600 focus:ring-0 cursor-pointer"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Projects Grid Section */}
      <div className="max-w-6xl mx-auto w-full">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 font-mono">
            Available Manga Projects
          </h2>
          <span className="text-xs text-slate-500 font-mono">
            {projects.length} loaded
          </span>
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-24">
            <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            <p className="mt-4 text-xs font-mono uppercase tracking-widest text-slate-400">Loading Projects...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((proj) => (
              <div
                key={proj.id}
                onClick={() => onSelectProject(proj.id)}
                className="group relative bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden hover:border-indigo-500/60 hover:shadow-xl hover:shadow-indigo-500/10 transition-all cursor-pointer flex flex-col"
              >
                {/* Thumbnail Header */}
                <div className="h-44 bg-slate-950 relative overflow-hidden flex items-center justify-center">
                  {proj.thumbnail_url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={proj.thumbnail_url.startsWith("http") ? proj.thumbnail_url : `http://localhost:8000${proj.thumbnail_url}`}
                      alt={proj.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500 opacity-80 group-hover:opacity-100"
                    />
                  ) : (
                    <div className="text-slate-600 font-mono text-xs">No Preview</div>
                  )}

                  <div className="absolute top-3 right-3">
                    <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full bg-slate-950/80 border border-slate-700/80 text-slate-300">
                      {proj.type}
                    </span>
                  </div>
                </div>

                {/* Details */}
                <div className="p-5 flex-1 flex flex-col justify-between">
                  <div>
                    <h3 className="font-bold text-base text-slate-100 group-hover:text-indigo-400 transition-colors">
                      {proj.name}
                    </h3>
                    <p className="mt-1 text-xs text-slate-400 line-clamp-2">
                      {proj.description}
                    </p>
                  </div>

                  <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
                    <span>{proj.pages_count} Pages</span>
                    <span>{proj.panels_count} Panels</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
