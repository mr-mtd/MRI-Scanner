import type { Plane, Vec3 } from "./types";

// Volumes arrive as uint8 in RAS+ canonical order, C-contiguous with shape
// (X, Y, Z): x = Left->Right, y = Posterior->Anterior, z = Inferior->Superior.
// Display follows radiological convention: patient right on screen left,
// anterior up on axial, superior up on coronal/sagittal, anterior left on sagittal.

export const LABEL_COLORS: Record<number, [number, number, number]> = {
  1: [255, 77, 79], // brain candidate lesion
  2: [56, 189, 248], // spinal canal
  3: [74, 222, 128], // disc
  4: [251, 191, 36], // disc with candidate abnormality
  5: [244, 63, 94], // candidate canal narrowing
};

export interface PlaneGeometry {
  width: number;
  height: number;
  pxW: number; // mm per screen pixel horizontally
  pxH: number;
  depth: number; // number of slices along the plane normal
  axis: 0 | 1 | 2;
  labels: { left: string; right: string; top: string; bottom: string };
}

export function planeGeometry(plane: Plane, shape: Vec3, spacing: Vec3): PlaneGeometry {
  const [X, Y, Z] = shape;
  const [sx, sy, sz] = spacing;
  switch (plane) {
    case "axial":
      return { width: X, height: Y, pxW: sx, pxH: sy, depth: Z, axis: 2, labels: { left: "R", right: "L", top: "A", bottom: "P" } };
    case "coronal":
      return { width: X, height: Z, pxW: sx, pxH: sz, depth: Y, axis: 1, labels: { left: "R", right: "L", top: "S", bottom: "I" } };
    case "sagittal":
      return { width: Y, height: Z, pxW: sy, pxH: sz, depth: X, axis: 0, labels: { left: "A", right: "P", top: "S", bottom: "I" } };
  }
}

/** Voxel coordinate for screen pixel (c, r) on a plane at the given slice. */
export function screenToVoxel(plane: Plane, shape: Vec3, slice: number, c: number, r: number): Vec3 {
  const [X, Y, Z] = shape;
  switch (plane) {
    case "axial":
      return [X - 1 - c, Y - 1 - r, slice];
    case "coronal":
      return [X - 1 - c, slice, Z - 1 - r];
    case "sagittal":
      return [slice, Y - 1 - c, Z - 1 - r];
  }
}

/** Screen pixel (c, r) of a voxel on a plane. */
export function voxelToScreen(plane: Plane, shape: Vec3, v: Vec3): [number, number] {
  const [X, Y, Z] = shape;
  switch (plane) {
    case "axial":
      return [X - 1 - v[0], Y - 1 - v[1]];
    case "coronal":
      return [X - 1 - v[0], Z - 1 - v[2]];
    case "sagittal":
      return [Y - 1 - v[1], Z - 1 - v[2]];
  }
}

export interface RenderOptions {
  windowLow: number; // 0-255
  windowHigh: number;
  overlayOpacity: number; // 0-1
  outline: boolean;
  showOverlay: boolean;
}

export function renderSlice(
  out: ImageData,
  plane: Plane,
  shape: Vec3,
  slice: number,
  volume: Uint8Array,
  mask: Uint8Array | null,
  opts: RenderOptions,
): void {
  const [, Y, Z] = shape;
  const { width, height } = out;
  const px = out.data;
  const range = Math.max(1, opts.windowHigh - opts.windowLow);
  const labelAt = (c: number, r: number): number => {
    if (c < 0 || r < 0 || c >= width || r >= height || !mask) return 0;
    const [x, y, z] = screenToVoxel(plane, shape, slice, c, r);
    return mask[(x * Y + y) * Z + z];
  };

  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const [x, y, z] = screenToVoxel(plane, shape, slice, c, r);
      const idx = (x * Y + y) * Z + z;
      let g = ((volume[idx] - opts.windowLow) / range) * 255;
      g = g < 0 ? 0 : g > 255 ? 255 : g;
      let R = g, G = g, B = g;
      if (mask && opts.showOverlay) {
        const lab = mask[idx];
        if (lab) {
          const draw =
            !opts.outline ||
            labelAt(c - 1, r) !== lab ||
            labelAt(c + 1, r) !== lab ||
            labelAt(c, r - 1) !== lab ||
            labelAt(c, r + 1) !== lab;
          if (draw) {
            const col = LABEL_COLORS[lab] ?? [255, 255, 255];
            const a = opts.outline ? Math.max(0.85, opts.overlayOpacity) : opts.overlayOpacity;
            R = R * (1 - a) + col[0] * a;
            G = G * (1 - a) + col[1] * a;
            B = B * (1 - a) + col[2] * a;
          }
        }
      }
      const o = (r * width + c) * 4;
      px[o] = R;
      px[o + 1] = G;
      px[o + 2] = B;
      px[o + 3] = 255;
    }
  }
}
