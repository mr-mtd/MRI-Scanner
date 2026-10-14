export type Vec3 = [number, number, number];

export interface StudyInfo {
  shape: Vec3;
  spacing: Vec3;
  affine: number[][];
  sourceFormat: "nifti" | "dicom";
  description: string;
  modalityHint: string;
  meta: Record<string, unknown>;
}

export interface StudyResponse {
  id: string;
  info: StudyInfo;
  disclaimer: string;
}

export interface Flag {
  code: string;
  text: string;
}

export interface BrainFinding {
  id: string;
  kind: "lesion";
  label: string;
  volumeMl: number;
  voxels: number;
  centroidVoxel: Vec3;
  centroidWorldMm: Vec3;
  extentMm: Vec3;
  maxDiameterMm: number;
  location: { side: string; region: string; depth: string; text: string };
  meanIntensityZ: number;
  asymmetryZ: number;
  confidence: number;
  bbox: [number, number][];
}

export interface SpineFinding {
  id: string;
  kind: "disc_level";
  label: string;
  level: string;
  canalApMm: number;
  canalApRatio: number | null;
  posteriorDisplacementMm: number;
  discHeightMm: number;
  discHeightRatio: number;
  discSignalRatio: number;
  flags: Flag[];
  severity: number;
  centroidVoxel: Vec3;
  centroidWorldMm: Vec3;
}

export type Finding = BrainFinding | SpineFinding;

export interface CanalProfile {
  zMm: number[];
  apMm: number[];
  levels: { level: string; zMm: number }[];
}

export interface AnalysisResult {
  summary: {
    mode: "brain" | "spine";
    engine: string;
    labels: Record<string, string>;
    notes: string[];
    params: Record<string, unknown>;
    // brain
    brainVolumeMl?: number;
    totalLesionVolumeMl?: number;
    candidateCount?: number;
    // spine
    useCase?: string;
    sagittalSlice?: number;
    referenceCanalApMm?: number;
    discsFound?: number;
    flaggedLevels?: string[];
    canalProfile?: CanalProfile;
  };
  findings: Finding[];
  disclaimer: string;
  study: { id: string; description: string; shape: Vec3; spacing: Vec3 };
}

export type Mode = "brain" | "spine";
export type Plane = "axial" | "coronal" | "sagittal";
export type ReviewStatus = "pending" | "accepted" | "rejected";

export interface BrainOptions {
  sensitivity: number;
  polarity: "hyper" | "hypo" | "both";
  min_volume_ml: number;
  require_asymmetry: boolean;
  engine: "auto" | "heuristic" | "onnx";
}

export interface SpineOptions {
  lowest_level: string;
  absolute_stenosis_mm: number;
  relative_stenosis_mm: number;
  narrowing_ratio: number;
  displacement_mm: number;
  sagittal_slice: number | null;
}
