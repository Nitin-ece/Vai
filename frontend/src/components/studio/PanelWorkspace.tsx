"use client";

import React, { useState, useEffect, useRef } from "react";
import { MangaViewer, ViewerControlsRef } from "../MangaViewer";
import { MangaScene, SceneLayer } from "../../types/scene";

export type StageName =
  | "enhance"
  | "vectorize"
  | "segment"
  | "layers"
  | "colorize"
  | "depth"
  | "viewer";

interface PanelWorkspaceProps {
  projectId: string;
  panelId: string;
  initialStage?: StageName;
  onBack: () => void;
  gpuStatus: {
    device_name: string;
    cuda_available: boolean;
    vram_gb: number;
  } | null;
}

export function PanelWorkspace({
  projectId,
  panelId,
  initialStage,
  onBack,
  gpuStatus,
}: PanelWorkspaceProps) {
  const [activeStage, setActiveStage] = useState<StageName>(initialStage || "enhance");
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (initialStage) {
      setActiveStage(initialStage);
    }
  }, [initialStage]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [jobStatus, setJobStatus] = useState<string>("Ready");
  const [cacheHit, setCacheHit] = useState<boolean>(false);

  // Stage outputs & artifacts data
  const [stageData, setStageData] = useState<Record<string, any>>({});
  const [scene, setScene] = useState<MangaScene | null>(null);

  // Interactive Tools States
  // 1. Click-to-segment tool
  const [promptPoints, setPromptPoints] = useState<Array<{ x: number; y: number; label: number }>>([]);
  const [promptMode, setPromptMode] = useState<"positive" | "negative">("positive");
  const [refinedMaskUrl, setRefinedMaskUrl] = useState<string | null>(null);

  // 2. Colorize overrides
  const [layerColors, setLayerColors] = useState<Record<string, string>>({});

  // 3. Depth overrides
  const [layerDepths, setLayerDepths] = useState<Record<string, number>>({});

  // 4. Layer Tree Visibility
  const [layerVisibility, setLayerVisibility] = useState<Record<string, boolean>>({});

  // Stage Tunable Parameters
  const [enhanceParams, setEnhanceParams] = useState({
    denoiseStrength: 3.0,
    deblurAmount: 0.6,
    upscaleFactor: 4,
  });

  const [vectorizeParams, setVectorizeParams] = useState({
    threshold: 128,
    smoothingTolerance: 1.0,
    minArea: 8,
  });

  const [segmentParams, setSegmentParams] = useState({
    pointsPerSide: 32,
    confThreshold: 0.85,
  });

  const [layersParams, setLayersParams] = useState({
    padding: 2,
    feather: 2.0,
    inpaintRadius: 5,
  });

  const [depthParams, setDepthParams] = useState({
    modelWeight: 0.5,
    bilateralSigma: 50.0,
  });

  const viewerRef = useRef<ViewerControlsRef | null>(null);

  // Load all stage outputs and scene for panel
  const fetchPanelData = async () => {
    setLoading(true);
    try {
      // 1. Fetch stage artifacts
      const stagesRes = await fetch(
        `http://localhost:8000/api/v1/projects/${projectId}/panels/${panelId}/stages`
      );
      if (stagesRes.ok) {
        const data = await stagesRes.json();
        setStageData(data.stages || {});
      }

      // 2. Fetch scene.json
      const sceneRes = await fetch(
        `http://localhost:8000/api/v1/scenes/${panelId}`
      );
      if (sceneRes.ok) {
        const scData: MangaScene = await sceneRes.json();
        setScene(scData);

        // Populate initial depths and colors
        const depths: Record<string, number> = {};
        const colors: Record<string, string> = {};
        const vis: Record<string, boolean> = {};

        scData.layers.forEach((l) => {
          depths[l.id] = l.depth;
          colors[l.id] = l.color || "#FFFFFF";
          vis[l.id] = true;
        });

        setLayerDepths(depths);
        setLayerColors(colors);
        setLayerVisibility(vis);
      }
    } catch (e) {
      console.error("Error loading panel workspace:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPanelData();
  }, [projectId, panelId]);

  // Helper to dynamically resolve artifact URLs for any project (benchmark or uploaded user chapters)
  const getArtifactUrl = (stage: string, artifactKey: string, fallbackBenchUrl: string): string => {
    const formatUrl = (u: string) => {
      if (!u) return "";
      if (u.startsWith("http://") || u.startsWith("https://") || u.startsWith("data:")) return u;
      return `http://localhost:8000${u.startsWith("/") ? u : "/" + u}`;
    };

    // 1. Check if current stage has the artifact
    const art = stageData[stage]?.artifacts?.[artifactKey];
    if (art) return formatUrl(art);
    // 2. Check if another stage has it
    for (const s of Object.keys(stageData)) {
      if (stageData[s]?.artifacts?.[artifactKey]) {
        return formatUrl(stageData[s].artifacts[artifactKey]);
      }
    }
    // 3. Fallback for custom uploaded project scenes
    if (projectId !== "benchmark") {
      const ext = artifactKey.includes("svg") ? "svg" : "png";
      return `http://localhost:8000/assets/scenes/${panelId}/${artifactKey}.${ext}`;
    }
    return formatUrl(fallbackBenchUrl);
  };

  const getPanelImagePath = (): string => {
    if (projectId === "benchmark") {
      return `data/output/benchmark_enhanced/${panelId}/enhanced.png`;
    }
    return `data/output/scenes/${panelId}/enhanced.png`;
  };

  const getPanelDir = (stage: string): string => {
    if (projectId === "benchmark") {
      return `data/output/benchmark_${stage}/${panelId}`;
    }
    return `data/output/scenes/${panelId}`;
  };

  // Re-run Stage Action
  const handleRunStage = async (stage: StageName) => {
    if (stage === "viewer") return;
    setIsProcessing(true);
    setJobStatus(`Running ${stage.toUpperCase()} Stage...`);

    try {
      const isBench = projectId === "benchmark";
      const inDir = isBench
        ? stage === "enhance"
          ? `data/benchmark_panels/${panelId}`
          : stage === "vectorize" || stage === "segment"
          ? `data/output/benchmark_enhanced/${panelId}`
          : stage === "layers"
          ? `data/output/benchmark_segmented/${panelId}`
          : stage === "colorize" || stage === "depth"
          ? `data/output/benchmark_layers/${panelId}`
          : `data/output/benchmark_depth/${panelId}`
        : `data/output/scenes/${panelId}`;

      const outDir = isBench
        ? `data/output/benchmark_${stage === "depth" ? "depth" : stage.endsWith("e") ? stage + "d" : stage + "ed"}/${panelId}`
        : `data/output/scenes/${panelId}`;

      const res = await fetch(`http://localhost:8000/api/v1/stages/${stage}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_dir: inDir,
          output_dir: outDir,
          config_overrides: {
            enhance: enhanceParams,
            vectorize: vectorizeParams,
            segment: segmentParams,
            layers: layersParams,
            depth: depthParams,
          },
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setCacheHit(data.cached);
        setJobStatus(data.cached ? "Completed (Cache Hit)" : "Completed Successfully");
        await fetchPanelData();
      } else {
        setJobStatus("Stage Execution Failed");
      }
    } catch (e) {
      console.error(`Failed to run stage ${stage}:`, e);
      setJobStatus("Execution Error");
    } finally {
      setIsProcessing(false);
    }
  };

  // 1. Interactive Tool: Click-to-segment prompt point
  const handleImageCanvasClick = async (e: React.MouseEvent<HTMLDivElement>) => {
    if (activeStage !== "segment") return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 1280);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 1280);

    const newPt = { x, y, label: promptMode === "positive" ? 1 : 0 };
    const updatedPts = [...promptPoints, newPt];
    setPromptPoints(updatedPts);

    // Call refine endpoint
    try {
      const res = await fetch("http://localhost:8000/api/v1/stages/segment/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: getPanelImagePath(),
          points: updatedPts,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.mask_base64) {
          setRefinedMaskUrl(`data:image/png;base64,${data.mask_base64}`);
        } else if (data.mask_url) {
          setRefinedMaskUrl(data.mask_url);
        }
      }
    } catch (err) {
      console.error("Refine mask error:", err);
    }
  };

  // 2. Interactive Tool: Override Layer Color
  const handleColorChange = async (layerId: string, newColor: string) => {
    setLayerColors((prev) => ({ ...prev, [layerId]: newColor }));
    try {
      await fetch("http://localhost:8000/api/v1/stages/colorize/override", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          panel_dir: getPanelDir("colorized"),
          layer_id: layerId,
          color: newColor,
        }),
      });
      fetchPanelData();
    } catch (e) {
      console.error("Color override failed:", e);
    }
  };

  // 3. Interactive Tool: Override Layer Depth
  const handleDepthChange = async (layerId: string, newDepth: number) => {
    setLayerDepths((prev) => ({ ...prev, [layerId]: newDepth }));
    try {
      await fetch("http://localhost:8000/api/v1/stages/depth/override", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          panel_dir: getPanelDir("depth"),
          layer_id: layerId,
          depth: newDepth,
        }),
      });
      fetchPanelData();
    } catch (e) {
      console.error("Depth override failed:", e);
    }
  };

  // 4. Export ZIP Package
  const handleExportZip = () => {
    window.location.href = `http://localhost:8000/api/v1/projects/${projectId}/panels/${panelId}/export`;
  };

  return (
    <div className="flex flex-col flex-1 h-full bg-slate-950 overflow-hidden font-sans">
      {/* Workspace Top Header */}
      <header className="h-14 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 flex items-center justify-between z-20">
        <div className="flex items-center space-x-3">
          <button
            onClick={onBack}
            className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 font-medium transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            <span>Back</span>
          </button>

          <div className="h-4 w-px bg-slate-800" />

          <div className="flex items-center space-x-2 text-xs font-mono">
            <span className="text-slate-400 capitalize">{projectId}</span>
            <span className="text-slate-600">/</span>
            <span className="text-indigo-400 font-bold">{panelId}</span>
          </div>
        </div>

        {/* Global Pipeline Stage Tabs */}
        <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
          {[
            { id: "enhance", label: "1. Enhance" },
            { id: "vectorize", label: "2. Vectorize" },
            { id: "segment", label: "3. Segment" },
            { id: "layers", label: "4. Layers" },
            { id: "colorize", label: "5. Colorize" },
            { id: "depth", label: "6. Depth" },
            { id: "viewer", label: "7. 2.5D Viewer" },
          ].map((st) => (
            <button
              key={st.id}
              onClick={() => setActiveStage(st.id as StageName)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                activeStage === st.id
                  ? "bg-indigo-600 text-white shadow-sm font-semibold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
              }`}
            >
              {st.label}
            </button>
          ))}
        </div>

        {/* Status Pills & Export */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-xs font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                isProcessing ? "bg-amber-400 animate-spin" : cacheHit ? "bg-cyan-400" : "bg-emerald-400"
              }`}
            />
            <span className="text-slate-300">{jobStatus}</span>
          </div>

          {gpuStatus && (
            <div className="hidden lg:flex items-center space-x-1.5 bg-slate-950 px-2.5 py-1 rounded border border-slate-800 text-[11px] font-mono text-slate-400">
              <span className={`w-1.5 h-1.5 rounded-full ${gpuStatus.cuda_available ? "bg-emerald-400" : "bg-amber-400"}`} />
              <span>{gpuStatus.cuda_available ? "CUDA GPU" : "CPU Engine"}</span>
            </div>
          )}

          <button
            onClick={handleExportZip}
            className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 active:scale-95 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 flex items-center space-x-1.5 transition-all"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Export Scene</span>
          </button>
        </div>
      </header>

      {/* Main Studio Viewport */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left/Center Stage Canvas */}
        <main className="flex-1 relative bg-slate-950 flex flex-col overflow-hidden">
          {loading ? (
            <div className="flex-1 flex flex-col items-center justify-center">
              <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <p className="mt-4 text-xs font-mono uppercase tracking-widest text-slate-400">Loading Stage Outputs...</p>
            </div>
          ) : activeStage === "viewer" ? (
            // Embedded 2.5D Three.js Viewer
            <div className="flex-1 relative w-full h-full">
              {scene ? (
                <MangaViewer
                  ref={viewerRef}
                  scene={scene}
                  panelId={panelId}
                  depthScale={1.6}
                  parallaxIntensity={1.0}
                />
              ) : null}

              <div className="absolute top-4 left-4 z-10 flex space-x-2">
                <button
                  onClick={() => viewerRef.current?.slowPushIn()}
                  className="px-3 py-1.5 bg-indigo-600/90 hover:bg-indigo-500 text-white rounded text-xs font-medium backdrop-blur-md shadow-lg"
                >
                  Slow Push-In
                </button>
                <button
                  onClick={() => viewerRef.current?.dramaticDrift()}
                  className="px-3 py-1.5 bg-slate-900/80 hover:bg-slate-800 text-slate-200 border border-slate-700 rounded text-xs font-medium backdrop-blur-md"
                >
                  Drift
                </button>
                <button
                  onClick={() => viewerRef.current?.resetCamera()}
                  className="px-3 py-1.5 bg-slate-900/80 hover:bg-slate-800 text-slate-200 border border-slate-700 rounded text-xs font-medium backdrop-blur-md"
                >
                  Reset
                </button>
              </div>
            </div>
          ) : (
            // 2D Stage Visualizer
            <div className="flex-1 p-8 flex items-center justify-center overflow-auto relative">
              {/* STAGE 1: ENHANCE */}
              {activeStage === "enhance" && (
                <div className="flex flex-col items-center space-y-4 max-w-4xl w-full">
                  <div className="grid grid-cols-2 gap-6 w-full">
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-slate-400 mb-2">Original Scan</span>
                      <div className="border border-slate-800 rounded-lg overflow-hidden bg-white shadow-xl">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getArtifactUrl("enhance", "original", `/assets/benchmark_panels/${panelId}/original.png`)}
                          alt="Original"
                          className="max-h-[500px] object-contain"
                        />
                      </div>
                    </div>
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-indigo-400 mb-2">Enhanced (4x Real-ESRGAN + Denoise)</span>
                      <div className="border border-indigo-500/50 rounded-lg overflow-hidden bg-white shadow-xl">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getArtifactUrl("enhance", "enhanced", `/assets/benchmark_enhanced/${panelId}/enhanced.png`)}
                          alt="Enhanced"
                          className="max-h-[500px] object-contain"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* STAGE 2: VECTORIZE */}
              {activeStage === "vectorize" && (
                <div className="flex flex-col items-center space-y-4 max-w-4xl w-full">
                  <div className="grid grid-cols-2 gap-6 w-full">
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-slate-400 mb-2">Vectorized SVG (Lines + Tone Defs)</span>
                      <div className="border border-slate-800 rounded-lg overflow-hidden bg-white shadow-xl p-2 flex items-center justify-center h-[500px] w-full">
                        <object
                          data={getArtifactUrl("vectorize", "lines", `/assets/benchmark_vectorized/${panelId}/lines.svg`)}
                          type="image/svg+xml"
                          className="max-h-[480px] max-w-full"
                        />
                      </div>
                    </div>
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-indigo-400 mb-2">4x Zoom Fidelity Render</span>
                      <div className="border border-slate-800 rounded-lg overflow-hidden bg-white shadow-xl flex items-center justify-center h-[500px] w-full">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getArtifactUrl("vectorize", "preview_4x", `/assets/benchmark_vectorized/${panelId}/preview_4x.png`)}
                          alt="4x Zoom"
                          className="max-h-[480px] object-contain"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* STAGE 3: SEGMENT (WITH CLICK-TO-SEGMENT TOOL) */}
              {activeStage === "segment" && (
                <div className="flex flex-col items-center space-y-4 max-w-4xl w-full">
                  <div className="flex items-center space-x-3 bg-slate-900 border border-slate-800 rounded-lg px-4 py-2">
                    <span className="text-xs text-slate-300 font-semibold">Click-to-Segment Tool:</span>
                    <button
                      onClick={() => setPromptMode("positive")}
                      className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
                        promptMode === "positive" ? "bg-emerald-600 text-white shadow-sm" : "text-slate-400 hover:text-white"
                      }`}
                    >
                      + Foreground Point
                    </button>
                    <button
                      onClick={() => setPromptMode("negative")}
                      className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
                        promptMode === "negative" ? "bg-rose-600 text-white shadow-sm" : "text-slate-400 hover:text-white"
                      }`}
                    >
                      - Background Point
                    </button>
                    {promptPoints.length > 0 && (
                      <button
                        onClick={() => {
                          setPromptPoints([]);
                          setRefinedMaskUrl(null);
                        }}
                        className="text-xs text-slate-400 hover:text-slate-200 underline ml-2"
                      >
                        Clear Points ({promptPoints.length})
                      </button>
                    )}
                  </div>

                  <div className="relative border border-slate-800 rounded-lg overflow-hidden bg-white shadow-2xl cursor-crosshair">
                    <div onClick={handleImageCanvasClick} className="relative">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={getArtifactUrl("segment", "overlay", `/assets/benchmark_segmented/${panelId}/overlay.png`)}
                        alt="Segment Overlay"
                        className="max-h-[500px] object-contain block"
                      />

                      {/* Render clicked prompt points */}
                      {promptPoints.map((pt, idx) => (
                        <div
                          key={idx}
                          style={{
                            left: `${(pt.x / 1280) * 100}%`,
                            top: `${(pt.y / 1280) * 100}%`,
                          }}
                          className={`absolute w-3.5 h-3.5 -ml-1.5 -mt-1.5 rounded-full border-2 border-white shadow-lg pointer-events-none ${
                            pt.label === 1 ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
                          }`}
                        />
                      ))}

                      {/* Render refined mask overlay if available */}
                      {refinedMaskUrl && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={refinedMaskUrl}
                          alt="Refined Mask"
                          className="absolute inset-0 w-full h-full object-contain pointer-events-none opacity-60 mix-blend-multiply filter drop-shadow-[0_0_8px_rgba(16,185,129,0.8)]"
                        />
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* STAGE 4: LAYERS */}
              {activeStage === "layers" && (
                <div className="flex flex-col items-center space-y-4 max-w-4xl w-full">
                  <div className="grid grid-cols-2 gap-6 w-full">
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-slate-400 mb-2">Inpainted Background Plate</span>
                      <div className="border border-slate-800 rounded-lg overflow-hidden bg-white shadow-xl">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getArtifactUrl("layers", "background_filled", `/assets/benchmark_layers/${panelId}/background_filled.png`)}
                          alt="Inpainted Background"
                          className="max-h-[500px] object-contain"
                        />
                      </div>
                    </div>
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-indigo-400 mb-2">Parallax Gap Diagnostic Test</span>
                      <div className="border border-slate-800 rounded-lg overflow-hidden bg-white shadow-xl">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getArtifactUrl("layers", "parallax_test", `/assets/benchmark_layers/${panelId}/parallax_test.png`)}
                          alt="Parallax Gap Test"
                          className="max-h-[500px] object-contain"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* STAGE 5: COLORIZE (WITH PER-LAYER COLOR PICKER) */}
              {activeStage === "colorize" && (
                <div className="flex flex-col items-center space-y-4 max-w-4xl w-full">
                  <div className="grid grid-cols-2 gap-6 w-full">
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-indigo-400 mb-2">Master Colored SVG Panel</span>
                      <div className="border border-slate-800 rounded-lg overflow-hidden bg-white shadow-xl p-2 flex items-center justify-center h-[500px] w-full">
                        <object
                          data={getArtifactUrl("colorize", "colored_panel", `/assets/benchmark_colorized/${panelId}/colored_panel.svg`)}
                          type="image/svg+xml"
                          className="max-h-[480px] max-w-full"
                        />
                      </div>
                    </div>
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-slate-400 mb-2">Colorize Before &amp; After</span>
                      <div className="border border-slate-800 rounded-lg overflow-hidden bg-white shadow-xl flex items-center justify-center h-[500px] w-full">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getArtifactUrl("colorize", "comparison", `/assets/benchmark_colorized/${panelId}/comparison.png`)}
                          alt="Comparison"
                          className="max-h-[480px] object-contain"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* STAGE 6: DEPTH (WITH DEPTH MAP & VISUALIZATION) */}
              {activeStage === "depth" && (
                <div className="flex flex-col items-center space-y-4 max-w-5xl w-full">
                  <div className="grid grid-cols-2 gap-6 w-full">
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-indigo-400 mb-2">Continuous Depth Map (Turbo)</span>
                      <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-900 shadow-xl">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getArtifactUrl("depth", "depth_map", `/assets/benchmark_depth/${panelId}/depth_map.png`)}
                          alt="Depth Map"
                          className="max-h-[500px] object-contain"
                        />
                      </div>
                    </div>
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-mono text-slate-400 mb-2">Spatial Depth Map &amp; Layer Ladder</span>
                      <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-900 shadow-xl">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getArtifactUrl("depth", "depth_visualization", `/assets/benchmark_depth/${panelId}/depth_visualization.png`)}
                          alt="Depth Visualization"
                          className="max-h-[500px] object-contain"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </main>

        {/* Right Sidebar: Stage Parameters & Layer Tree */}
        <aside className="w-88 border-l border-slate-800 bg-slate-900/60 backdrop-blur-md flex flex-col z-20 overflow-hidden">
          {/* Active Stage Parameters Drawer */}
          <div className="p-4 border-b border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs uppercase font-mono font-bold tracking-wider text-slate-300">
                {activeStage.toUpperCase()} Parameters
              </h3>
              {activeStage !== "viewer" && (
                <button
                  onClick={() => handleRunStage(activeStage)}
                  disabled={isProcessing}
                  className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white rounded text-xs font-semibold shadow transition-all disabled:opacity-50"
                >
                  {isProcessing ? "Processing..." : "Re-run"}
                </button>
              )}
            </div>

            {/* Stage Parameter Inputs */}
            {activeStage === "enhance" && (
              <div className="space-y-3 pt-1 text-xs">
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Denoise Strength (h)</span>
                    <span className="font-mono text-indigo-400">{enhanceParams.denoiseStrength}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="0.5"
                    value={enhanceParams.denoiseStrength}
                    onChange={(e) =>
                      setEnhanceParams((p) => ({ ...p, denoiseStrength: parseFloat(e.target.value) }))
                    }
                    className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded cursor-pointer"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Upscale Factor</span>
                    <span className="font-mono text-indigo-400">{enhanceParams.upscaleFactor}x</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {[2, 4].map((scale) => (
                      <button
                        key={scale}
                        onClick={() => setEnhanceParams((p) => ({ ...p, upscaleFactor: scale }))}
                        className={`py-1 text-xs rounded border ${
                          enhanceParams.upscaleFactor === scale
                            ? "bg-indigo-600 text-white border-indigo-500"
                            : "bg-slate-800 text-slate-300 border-slate-700"
                        }`}
                      >
                        {scale}x Real-ESRGAN
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeStage === "vectorize" && (
              <div className="space-y-3 pt-1 text-xs">
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Potrace Threshold</span>
                    <span className="font-mono text-indigo-400">{vectorizeParams.threshold}</span>
                  </div>
                  <input
                    type="range"
                    min="64"
                    max="192"
                    step="2"
                    value={vectorizeParams.threshold}
                    onChange={(e) =>
                      setVectorizeParams((p) => ({ ...p, threshold: parseInt(e.target.value) }))
                    }
                    className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded cursor-pointer"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Min Speck Area (px)</span>
                    <span className="font-mono text-indigo-400">{vectorizeParams.minArea}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="30"
                    step="1"
                    value={vectorizeParams.minArea}
                    onChange={(e) =>
                      setVectorizeParams((p) => ({ ...p, minArea: parseInt(e.target.value) }))
                    }
                    className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded cursor-pointer"
                  />
                </div>
              </div>
            )}

            {activeStage === "segment" && (
              <div className="space-y-3 pt-1 text-xs">
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Points Per Side</span>
                    <span className="font-mono text-indigo-400">{segmentParams.pointsPerSide}</span>
                  </div>
                  <input
                    type="range"
                    min="16"
                    max="64"
                    step="8"
                    value={segmentParams.pointsPerSide}
                    onChange={(e) =>
                      setSegmentParams((p) => ({ ...p, pointsPerSide: parseInt(e.target.value) }))
                    }
                    className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded cursor-pointer"
                  />
                </div>
              </div>
            )}

            {activeStage === "layers" && (
              <div className="space-y-3 pt-1 text-xs">
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Feather Radius (px)</span>
                    <span className="font-mono text-indigo-400">{layersParams.feather}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="5.0"
                    step="0.5"
                    value={layersParams.feather}
                    onChange={(e) =>
                      setLayersParams((p) => ({ ...p, feather: parseFloat(e.target.value) }))
                    }
                    className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded cursor-pointer"
                  />
                </div>
              </div>
            )}

            {activeStage === "depth" && (
              <div className="space-y-3 pt-1 text-xs">
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Model vs Prior Weight</span>
                    <span className="font-mono text-indigo-400">{(depthParams.modelWeight * 100).toFixed(0)}% Model</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={depthParams.modelWeight}
                    onChange={(e) =>
                      setDepthParams((p) => ({ ...p, modelWeight: parseFloat(e.target.value) }))
                    }
                    className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded cursor-pointer"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Layer Tree Inspector */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            <div className="flex items-center justify-between text-xs uppercase font-mono font-bold tracking-wider text-slate-400">
              <span>Layer Tree &amp; Overrides</span>
              <span>{scene?.layers.length || 0} Layers</span>
            </div>

            <div className="space-y-2">
              {scene?.layers
                .slice()
                .reverse()
                .map((layer) => {
                  const currentColor = layerColors[layer.id] || layer.color || "#FFFFFF";
                  const currentDepth = layerDepths[layer.id] ?? layer.depth;
                  const isVisible = layerVisibility[layer.id] !== false;

                  return (
                    <div
                      key={layer.id}
                      className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors"
                    >
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center space-x-2 truncate">
                          {/* Visibility Eye Toggle */}
                          <button
                            onClick={() =>
                              setLayerVisibility((prev) => ({ ...prev, [layer.id]: !isVisible }))
                            }
                            className="text-slate-400 hover:text-white"
                          >
                            {isVisible ? (
                              <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                              </svg>
                            ) : (
                              <svg className="w-4 h-4 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l18 18" />
                              </svg>
                            )}
                          </button>

                          <span className="font-semibold text-slate-200">
                            z{layer.zOrder}: {layer.id}
                          </span>
                        </div>

                        {/* Semantic Type Badge */}
                        <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-400">
                          {layer.type}
                        </span>
                      </div>

                      {/* Interactive Color Picker & Depth Slider */}
                      <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800/60 items-center">
                        {/* Color Picker */}
                        <div className="flex items-center space-x-1.5">
                          <input
                            type="color"
                            value={currentColor}
                            onChange={(e) => handleColorChange(layer.id, e.target.value)}
                            className="w-5 h-5 rounded cursor-pointer bg-transparent border-0"
                          />
                          <span className="font-mono text-[10px] text-slate-400 uppercase">
                            {currentColor}
                          </span>
                        </div>

                        {/* Depth Slider */}
                        <div className="flex items-center space-x-1.5">
                          <input
                            type="range"
                            min="0.0"
                            max="1.0"
                            step="0.01"
                            value={currentDepth}
                            onChange={(e) => handleDepthChange(layer.id, parseFloat(e.target.value))}
                            className="w-full accent-indigo-500 h-1 bg-slate-800 rounded cursor-pointer"
                          />
                          <span className="font-mono text-[10px] text-indigo-400 w-8 text-right">
                            {currentDepth.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
