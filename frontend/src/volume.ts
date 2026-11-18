import type { Plane } from "./types";

/**
 * Volumes arrive as C-order (X, Y, Z) arrays in canonical RAS+ orientation:
 * x grows toward patient right, y toward anterior, z toward superior.
 * Views use radiological convention: patient right on screen left (axial,
 * coronal), anterior on the left (sagittal), superior at the top.
 */
export type Shape = [number, number, number];

export function planeSize(plane: Plane, shape: Shape): { cols: number; rows: number } {
  const [X, Y, Z] = shape;
  if (plane === "axial") return { cols: X, rows: Y };
  if (plane === "coronal") return { cols: X, rows: Z };
  return { cols: Y, rows: Z };
}

export function planeSpacing(plane: Plane, spacing: Shape): { sc: number; sr: number } {
  const [sx, sy, sz] = spacing;
  if (plane === "axial") return { sc: sx, sr: sy };
  if (plane === "coronal") return { sc: sx, sr: sz };
  return { sc: sy, sr: sz };
}

export function sliceCount(plane: Plane, shape: Shape): number {
  return plane === "axial" ? shape[2] : plane === "coronal" ? shape[1] : shape[0];
}

export function viewToVoxel(plane: Plane, shape: Shape, col: number, row: number, index: number): Shape {
  const [X, Y, Z] = shape;
  if (plane === "axial") return [X - 1 - col, Y - 1 - row, index];
  if (plane === "coronal") return [X - 1 - col, index, Z - 1 - row];
  return [index, Y - 1 - col, Z - 1 - row];
}

export function voxelToView(plane: Plane, shape: Shape, v: Shape): { col: number; row: number; index: number } {
  const [X, Y, Z] = shape;
  if (plane === "axial") return { col: X - 1 - v[0], row: Y - 1 - v[1], index: v[2] };
  if (plane === "coronal") return { col: X - 1 - v[0], row: Z - 1 - v[2], index: v[1] };
  return { col: Y - 1 - v[1], row: Z - 1 - v[2], index: v[0] };
}

/** Copy one plane of a (X, Y, Z) uint8 array into a row-major cols x rows image. */
export function extractSlice(data: Uint8Array, shape: Shape, plane: Plane, index: number): Uint8Array {
  const [, Y, Z] = shape;
  const { cols, rows } = planeSize(plane, shape);
  const out = new Uint8Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const [x, y, z] = viewToVoxel(plane, shape, c, r, index);
      out[r * cols + c] = data[x * Y * Z + y * Z + z];
    }
  }
  return out;
}

export const CANAL_LABEL = 250;
export const DISC_LABEL = 251;

const PALETTE: [number, number, number][] = [
  [255, 64, 64],
  [255, 170, 0],
  [80, 220, 100],
  [80, 160, 255],
  [220, 90, 255],
  [255, 105, 180],
  [0, 220, 220],
  [240, 230, 60],
];

export function labelColor(label: number): [number, number, number] {
  if (label === CANAL_LABEL) return [0, 200, 255];
  if (label === DISC_LABEL) return [255, 230, 120];
  return PALETTE[(label - 1) % PALETTE.length];
}

export function labelCss(label: number): string {
  const [r, g, b] = labelColor(label);
  return `rgb(${r}, ${g}, ${b})`;
}
