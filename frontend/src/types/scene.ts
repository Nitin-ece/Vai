/**
 * TypeScript types matching docs/scene_schema.json (v1.0)
 */

export interface ScenePanel {
  project: string;
  page: number;
  panel: number;
  source: string;
  width: number;
  height: number;
}

export interface SceneCameraLimits {
  minDistance?: number;
  maxDistance?: number;
  maxPitch?: number;
  maxYaw?: number;
}

export interface SceneCamera {
  fov: number;
  near: number;
  far: number;
  position: [number, number, number];
  limits?: SceneCameraLimits;
}

export interface LayerTransform {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
}

export interface SwayAnimation {
  enabled: boolean;
  speed: number;
  amplitude: number;
}

export interface FloatAnimation {
  enabled: boolean;
  speed: number;
  amplitude: number;
}

export interface BlinkAnimation {
  enabled: boolean;
  interval: number;
}

export interface LayerAnimations {
  sway?: SwayAnimation;
  float?: FloatAnimation;
  blink?: BlinkAnimation;
}

export interface LayerMetadata {
  part?: string | null;
  confidence?: number;
  model_depth?: number;
  prior_depth?: number;
  is_override?: boolean;
  bbox?: [number, number, number, number];
  animations?: LayerAnimations;
  [key: string]: unknown;
}

export type LayerType =
  | "background"
  | "character"
  | "character_part"
  | "speech_bubble"
  | "sfx"
  | "foreground"
  | "effect"
  | "prop"
  | "environment";

export interface SceneLayer {
  id: string;
  type: LayerType;
  parentId: string | null;
  svgPath: string;
  pngPath: string;
  color: string;
  depth: number;
  zOrder: number;
  transform: LayerTransform;
  metadata?: LayerMetadata;
}

export interface MangaScene {
  version: string;
  panel: ScenePanel;
  camera: SceneCamera;
  layers: SceneLayer[];
}
