"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import gsap from "gsap";
import { MangaScene, SceneLayer } from "../types/scene";

interface MangaViewerProps {
  scene: MangaScene;
  panelId: string;
  depthScale?: number;
  parallaxIntensity?: number;
  swayActive?: boolean;
  floatActive?: boolean;
  blinkActive?: boolean;
  onFpsUpdate?: (fps: number) => void;
}

export interface ViewerControlsRef {
  slowPushIn: () => void;
  dramaticDrift: () => void;
  resetCamera: () => void;
}

export const MangaViewer = React.forwardRef<ViewerControlsRef, MangaViewerProps>(
  (
    {
      scene,
      panelId,
      depthScale = 1.6,
      parallaxIntensity = 1.0,
      swayActive = false,
      floatActive = false,
      blinkActive = false,
      onFpsUpdate,
    },
    ref
  ) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const layerMeshesRef = useRef<Map<string, { mesh: THREE.Mesh; layer: SceneLayer; baseZ: number; baseY: number }>>(
      new Map()
    );

    // Mouse and camera motion states
    const mouseRef = useRef<{ x: number; y: number; targetX: number; targetY: number }>({
      x: 0,
      y: 0,
      targetX: 0,
      targetY: 0,
    });
    const isDraggingRef = useRef<boolean>(false);
    const previousMousePositionRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
    const orbitAnglesRef = useRef<{ yaw: number; pitch: number }>({ yaw: 0, pitch: 0 });

    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [loadedCount, setLoadedCount] = useState<number>(0);

    // Camera limits from scene
    const limits = scene.camera.limits || {
      minDistance: 3.2,
      maxDistance: 5.5,
      maxPitch: 0.045, // ~2.57 deg
      maxYaw: 0.060,   // ~3.44 deg
    };

    // Forward ref methods for GSAP cinematics
    React.useImperativeHandle(ref, () => ({
      slowPushIn: () => {
        if (!cameraRef.current) return;
        gsap.killTweensOf(cameraRef.current.position);
        gsap.to(cameraRef.current.position, {
          z: limits.minDistance ? limits.minDistance + 0.3 : 3.6,
          duration: 4.5,
          ease: "power2.inOut",
        });
      },
      dramaticDrift: () => {
        if (!cameraRef.current) return;
        gsap.killTweensOf(orbitAnglesRef.current);
        const tl = gsap.timeline({ repeat: -1, yoyo: true });
        tl.to(orbitAnglesRef.current, {
          yaw: limits.maxYaw ? limits.maxYaw * 0.85 : 0.05,
          pitch: limits.maxPitch ? -limits.maxPitch * 0.7 : -0.03,
          duration: 3.5,
          ease: "sine.inOut",
        }).to(orbitAnglesRef.current, {
          yaw: limits.maxYaw ? -limits.maxYaw * 0.85 : -0.05,
          pitch: limits.maxPitch ? limits.maxPitch * 0.7 : 0.03,
          duration: 3.5,
          ease: "sine.inOut",
        });
      },
      resetCamera: () => {
        if (!cameraRef.current) return;
        gsap.killTweensOf(cameraRef.current.position);
        gsap.killTweensOf(orbitAnglesRef.current);
        gsap.to(cameraRef.current.position, {
          x: scene.camera.position[0],
          y: scene.camera.position[1],
          z: scene.camera.position[2],
          duration: 1.2,
          ease: "power2.out",
        });
        gsap.to(orbitAnglesRef.current, {
          yaw: 0,
          pitch: 0,
          duration: 1.2,
          ease: "power2.out",
        });
      },
    }));

    // Mouse Parallax Event Handlers
    const handleMouseMove = useCallback((e: MouseEvent) => {
      if (isDraggingRef.current) {
        const deltaX = e.clientX - previousMousePositionRef.current.x;
        const deltaY = e.clientY - previousMousePositionRef.current.y;

        const maxPitch = limits.maxPitch ?? 0.045;
        const maxYaw = limits.maxYaw ?? 0.060;

        orbitAnglesRef.current.yaw = Math.max(
          -maxYaw,
          Math.min(maxYaw, orbitAnglesRef.current.yaw + deltaX * 0.001)
        );
        orbitAnglesRef.current.pitch = Math.max(
          -maxPitch,
          Math.min(maxPitch, orbitAnglesRef.current.pitch - deltaY * 0.001)
        );

        previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
      } else {
        const width = window.innerWidth;
        const height = window.innerHeight;
        // Normalized device coordinates [-1, 1]
        mouseRef.current.targetX = ((e.clientX / width) * 2 - 1) * parallaxIntensity;
        mouseRef.current.targetY = (-(e.clientY / height) * 2 + 1) * parallaxIntensity;
      }
    }, [limits, parallaxIntensity]);

    const handleMouseDown = useCallback((e: MouseEvent) => {
      isDraggingRef.current = true;
      previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
    }, []);

    const handleMouseUp = useCallback(() => {
      isDraggingRef.current = false;
    }, []);

    const handleWheel = useCallback((e: WheelEvent) => {
      if (!cameraRef.current) return;
      e.preventDefault();
      const zoomSpeed = 0.003;
      const minD = limits.minDistance ?? 3.2;
      const maxD = limits.maxDistance ?? 5.5;
      const newZ = cameraRef.current.position.z + e.deltaY * zoomSpeed;
      cameraRef.current.position.z = Math.max(minD, Math.min(maxD, newZ));
    }, [limits]);

    // Setup Three.js Scene
    useEffect(() => {
      if (!containerRef.current) return;
      const container = containerRef.current;
      const width = container.clientWidth || 800;
      const height = container.clientHeight || 800;

      // 1. Scene & Camera
      const threeScene = new THREE.Scene();
      threeScene.background = new THREE.Color("#0c0d10");
      sceneRef.current = threeScene;

      const camera = new THREE.PerspectiveCamera(
        scene.camera.fov || 45,
        width / height,
        scene.camera.near || 0.1,
        scene.camera.far || 100.0
      );
      camera.position.set(
        scene.camera.position[0],
        scene.camera.position[1],
        scene.camera.position[2]
      );
      cameraRef.current = camera;

      // 2. Renderer
      const renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: "high-performance",
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      container.innerHTML = "";
      container.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      // 3. Build Layer Meshes with Depth & Perspective Alignment
      const pw = scene.panel.width || 1280;
      const ph = scene.panel.height || 1280;
      const aspect = pw / ph;

      // World dimensions at Z = 0
      const worldHeight = 3.6;
      const worldWidth = worldHeight * aspect;

      const textureLoader = new THREE.TextureLoader();
      const layersMap = new Map<string, { mesh: THREE.Mesh; layer: SceneLayer; baseZ: number; baseY: number }>();
      layerMeshesRef.current = layersMap;

      const totalLayers = scene.layers.length;
      let loaded = 0;
      setIsLoading(true);

      const zCam = scene.camera.position[2] || 5.0;

      scene.layers.forEach((layer) => {
        // Compute world Z from normalized depth
        // 0.0 (far) -> -depthScale * 0.5
        // 1.0 (near) -> +depthScale * 0.5
        const worldZ = (layer.depth - 0.5) * depthScale;

        // Perspective compensation scale factor
        // S(Z) = (Z_cam - Z) / Z_cam ensures pixel-perfect 2D alignment when camera is at center
        const perspectiveScale = (zCam - worldZ) / zCam;

        const geometry = new THREE.PlaneGeometry(
          worldWidth * perspectiveScale,
          worldHeight * perspectiveScale
        );

        // Asset URI: benchmark scenes use static /scenes/, uploaded scenes load directly from /assets/
        const isBenchmark = [
          "clean_lineart",
          "dense_screentone",
          "lowres_scan",
          "speed_lines",
          "text_heavy",
        ].includes(panelId);

        const primaryUri = isBenchmark
          ? `/scenes/${panelId}/${layer.pngPath}`
          : `http://localhost:8000/assets/scenes/${panelId}/${layer.pngPath}`;
        const fallbackUri = isBenchmark
          ? `http://localhost:8000/assets/scenes/${panelId}/${layer.pngPath}`
          : `/scenes/${panelId}/${layer.pngPath}`;

        const loadTexture = (uri: string, isFallback: boolean = false) => {
          textureLoader.load(
            uri,
            (texture) => {
              texture.colorSpace = THREE.SRGBColorSpace;
              texture.minFilter = THREE.LinearFilter;
              texture.magFilter = THREE.LinearFilter;
              texture.generateMipmaps = false;

              const material = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                alphaTest: 0.005,
                depthWrite: false,
                depthTest: true,
                side: THREE.DoubleSide,
              });

              const mesh = new THREE.Mesh(geometry, material);
              mesh.position.set(0, 0, worldZ);
              mesh.renderOrder = layer.zOrder;

              threeScene.add(mesh);
              layersMap.set(layer.id, { mesh, layer, baseZ: worldZ, baseY: 0 });

              loaded++;
              setLoadedCount(loaded);
              if (loaded >= totalLayers) {
                setIsLoading(false);
              }
            },
            undefined,
            () => {
              if (!isFallback) {
                loadTexture(fallbackUri, true);
              } else {
                console.warn(`Failed to load texture for layer ${layer.id} at ${uri}`);
                loaded++;
                setLoadedCount(loaded);
                if (loaded >= totalLayers) {
                  setIsLoading(false);
                }
              }
            }
          );
        };

        loadTexture(primaryUri);
      });

      // 4. Render & Animation Loop (60 FPS)
      let animationFrameId: number;
      let lastTime = performance.now();
      let frameCount = 0;
      let fpsTime = lastTime;

      const animate = (currentTime: number) => {
        animationFrameId = requestAnimationFrame(animate);

        // Measure FPS
        frameCount++;
        if (currentTime - fpsTime >= 500) {
          const currentFps = Math.round((frameCount * 1000) / (currentTime - fpsTime));
          if (onFpsUpdate) onFpsUpdate(currentFps);
          frameCount = 0;
          fpsTime = currentTime;
        }

        const delta = (currentTime - lastTime) * 0.001;
        lastTime = currentTime;
        const timeSec = currentTime * 0.001;

        // Smooth mouse parallax damping (lerp)
        mouseRef.current.x += (mouseRef.current.targetX - mouseRef.current.x) * 0.08;
        mouseRef.current.y += (mouseRef.current.targetY - mouseRef.current.y) * 0.08;

        // Camera positioning with strict angle limits
        const maxYaw = limits.maxYaw ?? 0.060;
        const maxPitch = limits.maxPitch ?? 0.045;

        const effectiveYaw = Math.max(
          -maxYaw,
          Math.min(maxYaw, orbitAnglesRef.current.yaw + mouseRef.current.x * maxYaw)
        );
        const effectivePitch = Math.max(
          -maxPitch,
          Math.min(maxPitch, orbitAnglesRef.current.pitch + mouseRef.current.y * maxPitch)
        );

        // Gentle camera tilt to complement per-segment parallax
        camera.rotation.y = effectiveYaw * 0.35;
        camera.rotation.x = effectivePitch * 0.35;
        camera.position.x = Math.sin(effectiveYaw * 0.35) * camera.position.z;
        camera.position.y = -Math.sin(effectivePitch * 0.35) * camera.position.z;

        // Per-Segment Relative Parallax & Animation Hooks
        // Parallax is calculated individually per layer/segment relative to its depth
        layersMap.forEach(({ mesh, layer, baseZ, baseY }) => {
          const anims = layer.metadata?.animations;

          // Parallax relative to each individual layer's depth:
          // The base white canvas (depth <= 0.05) remains completely stationary.
          // Characters, props, and bubbles displace across X and Y based on their individual depth.
          const relDepth = Math.max(0, (layer.depth ?? 0.5) - 0.05);
          const layerShiftX = mouseRef.current.x * relDepth * 0.45 * parallaxIntensity;
          const layerShiftY = mouseRef.current.y * relDepth * 0.45 * parallaxIntensity;
          const layerTiltY = mouseRef.current.x * relDepth * 0.07 * parallaxIntensity;
          const layerTiltX = -mouseRef.current.y * relDepth * 0.07 * parallaxIntensity;

          mesh.position.x = layerShiftX;
          mesh.position.z = baseZ;
          mesh.rotation.y = layerTiltY;
          mesh.rotation.x = layerTiltX;

          // 1. Sway animation (hair, clothes, foliage)
          if (swayActive || anims?.sway?.enabled) {
            const speed = anims?.sway?.speed ?? 1.5;
            const amp = anims?.sway?.amplitude ?? 0.02;
            mesh.rotation.z = Math.sin(timeSec * speed * 2.0) * amp;
          } else {
            mesh.rotation.z = 0;
          }

          // 2. Float animation (speech bubbles, sfx)
          if (floatActive || anims?.float?.enabled) {
            const speed = anims?.float?.speed ?? 1.8;
            const amp = anims?.float?.amplitude ?? 0.035;
            mesh.position.y = baseY + layerShiftY + Math.sin(timeSec * speed * 2.5) * amp;
          } else {
            mesh.position.y = baseY + layerShiftY;
          }

          // 3. Blink animation (eyes)
          if (blinkActive || anims?.blink?.enabled) {
            const interval = anims?.blink?.interval ?? 3.5;
            const cycle = (timeSec % interval) / interval;
            // Short blink dip
            if (cycle > 0.94) {
              (mesh.material as THREE.MeshBasicMaterial).opacity = 0.2;
            } else {
              (mesh.material as THREE.MeshBasicMaterial).opacity = 1.0;
            }
          }
        });

        renderer.render(threeScene, camera);
      };

      animate(performance.now());

      // 5. Window Resize Handler
      const handleResize = () => {
        if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
        const newW = containerRef.current.clientWidth || 800;
        const newH = containerRef.current.clientHeight || 800;
        cameraRef.current.aspect = newW / newH;
        cameraRef.current.updateProjectionMatrix();
        rendererRef.current.setSize(newW, newH);
      };

      window.addEventListener("resize", handleResize);
      container.addEventListener("mousemove", handleMouseMove);
      container.addEventListener("mousedown", handleMouseDown);
      window.addEventListener("mouseup", handleMouseUp);
      container.addEventListener("wheel", handleWheel, { passive: false });

      return () => {
        cancelAnimationFrame(animationFrameId);
        window.removeEventListener("resize", handleResize);
        container.removeEventListener("mousemove", handleMouseMove);
        container.removeEventListener("mousedown", handleMouseDown);
        window.removeEventListener("mouseup", handleMouseUp);
        container.removeEventListener("wheel", handleWheel);
        renderer.dispose();
      };
    }, [
      scene,
      panelId,
      depthScale,
      limits,
      swayActive,
      floatActive,
      blinkActive,
      handleMouseMove,
      handleMouseDown,
      handleMouseUp,
      handleWheel,
      onFpsUpdate,
    ]);

    return (
      <div className="relative w-full h-full select-none overflow-hidden bg-slate-950">
        <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-sm z-30 transition-opacity">
            <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            <p className="mt-4 text-xs font-mono uppercase tracking-widest text-slate-300">
              Assembling 2.5D Planes ({loadedCount} / {scene.layers.length})
            </p>
          </div>
        )}

        {/* Parallax Clamping Safety Indicator */}
        <div className="absolute bottom-4 left-4 pointer-events-none z-10 flex items-center space-x-2 bg-slate-900/70 border border-slate-800 rounded-md px-3 py-1.5 text-[11px] font-mono text-slate-400 backdrop-blur-md">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>Gaps Hidden (Yaw: ±{((limits.maxYaw ?? 0.06) * 57.3).toFixed(1)}°, Pitch: ±{((limits.maxPitch ?? 0.045) * 57.3).toFixed(1)}°)</span>
        </div>
      </div>
    );
  }
);

MangaViewer.displayName = "MangaViewer";
