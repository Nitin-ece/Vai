"use client";

import React, { useState, useEffect } from "react";
import { ProjectList } from "../components/studio/ProjectList";
import { PageBrowser } from "../components/studio/PageBrowser";
import { PanelWorkspace, StageName } from "../components/studio/PanelWorkspace";

type StudioView = "projects" | "pages" | "workspace";

export default function Home() {
  const [currentView, setCurrentView] = useState<StudioView>("projects");
  const [selectedProjectId, setSelectedProjectId] = useState<string>("benchmark");
  const [selectedPanelId, setSelectedPanelId] = useState<string>("clean_lineart");
  const [initialStage, setInitialStage] = useState<StageName>("enhance");
  const [gpuStatus, setGpuStatus] = useState<{
    device_name: string;
    cuda_available: boolean;
    vram_gb: number;
  } | null>(null);

  // Global window drag & drop overlay state
  const [isWindowDragging, setIsWindowDragging] = useState<boolean>(false);
  const [globalUploading, setGlobalUploading] = useState<{
    active: boolean;
    status: string;
    progress: number;
  }>({
    active: false,
    status: "",
    progress: 0,
  });

  // Fetch GPU / System Diagnostics on mount
  useEffect(() => {
    fetch("http://localhost:8000/api/v1/system/gpu")
      .then((res) => res.json())
      .then((data) => setGpuStatus(data))
      .catch((err) => console.warn("Failed to fetch GPU status:", err));
  }, []);

  // Global window drag-and-drop listeners
  useEffect(() => {
    let dragCounter = 0;

    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      dragCounter++;
      if (e.dataTransfer && e.dataTransfer.types.includes("Files")) {
        setIsWindowDragging(true);
      }
    };

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        setIsWindowDragging(false);
      }
    };

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
    };

    const handleDrop = async (e: DragEvent) => {
      e.preventDefault();
      dragCounter = 0;
      setIsWindowDragging(false);

      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        await processDroppedManga(file);
      }
    };

    window.addEventListener("dragenter", handleDragEnter);
    window.addEventListener("dragleave", handleDragLeave);
    window.addEventListener("dragover", handleDragOver);
    window.addEventListener("drop", handleDrop);

    return () => {
      window.removeEventListener("dragenter", handleDragEnter);
      window.removeEventListener("dragleave", handleDragLeave);
      window.removeEventListener("dragover", handleDragOver);
      window.removeEventListener("drop", handleDrop);
    };
  }, []);

  const processDroppedManga = async (file: File) => {
    setGlobalUploading({
      active: true,
      status: `Uploading & Ingesting ${file.name}...`,
      progress: 20,
    });

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", file.name.replace(/\.[^/.]+$/, ""));
    formData.append("auto_process", "true");
    formData.append("fast_mode", "true");

    const t1 = setTimeout(() => {
      setGlobalUploading((prev) => ({ ...prev, status: "Enhancing & Segmenting Layers...", progress: 50 }));
    }, 2500);

    const t2 = setTimeout(() => {
      setGlobalUploading((prev) => ({ ...prev, status: "Colorizing & Estimating Depth...", progress: 75 }));
    }, 6000);

    const t3 = setTimeout(() => {
      setGlobalUploading((prev) => ({ ...prev, status: "Assembling 2.5D Scene...", progress: 92 }));
    }, 9500);

    try {
      const res = await fetch("http://localhost:8000/api/v1/projects/upload", {
        method: "POST",
        body: formData,
      });

      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);

      if (!res.ok) {
        throw new Error("Failed to process manga");
      }

      const data = await res.json();
      setGlobalUploading({ active: true, status: "Complete! Launching Viewer...", progress: 100 });

      setTimeout(() => {
        setGlobalUploading({ active: false, status: "", progress: 0 });
        setSelectedProjectId(data.project_id);
        setSelectedPanelId(data.scene_id || data.panel_id);
        setInitialStage("viewer");
        setCurrentView("workspace");
      }, 600);
    } catch (err) {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      console.error("Global drop upload error:", err);
      setGlobalUploading({ active: false, status: "", progress: 0 });
      alert("Failed to reconstruct manga. Please try again or check server.");
    }
  };

  const handleOpenViewer = (projId: string, panId: string) => {
    setSelectedProjectId(projId);
    setSelectedPanelId(panId);
    setInitialStage("viewer");
    setCurrentView("workspace");
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans relative">
      {/* Global Window Drag Overlay */}
      {isWindowDragging && (
        <div className="absolute inset-0 z-50 bg-indigo-950/80 backdrop-blur-md border-4 border-dashed border-indigo-400 flex flex-col items-center justify-center pointer-events-none animate-fadeIn">
          <div className="w-20 h-20 rounded-3xl bg-indigo-500 text-white flex items-center justify-center shadow-2xl shadow-indigo-500/50 mb-6 animate-bounce">
            <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            Drop your Manga anywhere
          </h2>
          <p className="text-sm font-mono text-indigo-300 mt-2">
            Release to ingest, layer, and launch the 2.5D interactive scene
          </p>
        </div>
      )}

      {/* Global Uploading / Processing Modal */}
      {globalUploading.active && (
        <div className="absolute inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex flex-col items-center justify-center">
          <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl shadow-2xl max-w-md w-full flex flex-col items-center text-center">
            <div className="w-12 h-12 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin mb-4" />
            <h3 className="text-base font-bold text-white mb-1">
              Reconstructing 2.5D Manga
            </h3>
            <p className="text-xs font-mono text-indigo-400 mb-5 animate-pulse">
              {globalUploading.status}
            </p>

            <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
              <div
                className="bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-400 h-2.5 rounded-full transition-all duration-500"
                style={{ width: `${globalUploading.progress}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {currentView === "projects" && (
        <ProjectList
          onSelectProject={(projId) => {
            setSelectedProjectId(projId);
            setInitialStage("enhance");
            setCurrentView("pages");
          }}
          onOpenViewer={handleOpenViewer}
          gpuStatus={gpuStatus}
        />
      )}

      {currentView === "pages" && (
        <PageBrowser
          projectId={selectedProjectId}
          onBack={() => setCurrentView("projects")}
          onSelectPanel={(panId) => {
            setSelectedPanelId(panId);
            setInitialStage("enhance");
            setCurrentView("workspace");
          }}
        />
      )}

      {currentView === "workspace" && (
        <PanelWorkspace
          projectId={selectedProjectId}
          panelId={selectedPanelId}
          initialStage={initialStage}
          onBack={() => setCurrentView("pages")}
          gpuStatus={gpuStatus}
        />
      )}
    </div>
  );
}
