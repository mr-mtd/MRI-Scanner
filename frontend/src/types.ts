export type Mode = "brain" | "spine";
export type Plane = "axial" | "coronal" | "sagittal";
export type ReviewState = "pending" | "accepted" | "rejected" | "uncertain";

export interface StudyMeta {
  id: string;
  description: string;
  source_format: string;
  shape: [number, number, number];
  spacing_mm: [number, number, number];
  affine: number[][];
  orientation: string;
  warnings: string[];
  metadata: Record<string, unknown>;
  suggested_mode?: Mode;
  demo_truth?: unknown;
}

export interface Finding {
  id: number;
  label_value: number;
  type: string;
  description?: string;
  location: Record<string, unknown>;
  measurements: Record<string, unknown> & {
    centroid_voxel: [number, number, number];
    centroid_ras_mm: [number, number, number];
  };
  severity?: string;
  score: number;
  slice_hint: { axial: number; coronal: number; sagittal: number };
  review: ReviewState;
}

export interface ProfilePoint {
  z_index: number;
  z_mm: number;
  ap_mm: number;
  level: string | null;
}

export interface Disc {
  level: string;
  z_index: number;
  z_mm: number;
  signal: number;
  signal_ratio_to_brightest_discs?: number;
  signal_ratio_to_csf?: number | null;
}

export interface AnalysisResult {
  study_id: string;
  disclaimer: string;
  mode: Mode;
  method: { name: string; kind: string; details: Record<string, unknown> };
  summary: Record<string, number | string>;
  findings: Finding[];
  warnings: string[];
  profile?: ProfilePoint[];
  discs?: Disc[];
  request: Record<string, unknown>;
}

export interface AnalyzeOptions {
  mode: Mode;
  method: "baseline" | "pretrained";
  sensitivity: number;
  min_volume_mm3: number;
  polarity: "hyper" | "hypo";
  use_asymmetry: boolean;
  already_skull_stripped: boolean;
  min_protrusion_mm: number;
}

export interface Health {
  status: string;
  disclaimer: string;
  pretrained_brain_model: string | null;
  pretrained_brain_model_error: string | null;
}
