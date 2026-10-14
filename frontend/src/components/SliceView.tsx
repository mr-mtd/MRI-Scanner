import { useEffect, useMemo, useRef } from "react";
import type { Plane } from "../types";
import {
  CANAL_LABEL,
  DISC_LABEL,
  extractSlice,
  labelColor,
  planeSize,
  planeSpacing,
  sliceCount,
  viewToVoxel,
  voxelToView,
  type Shape,
} from "../volume";

interface Props {
  plane: Plane;
  shape: Shape;
  spacing: Shape;
  volume: Uint8Array;
  labels: Uint8Array | null;
  cursor: Shape;
  onCursor: (v: Shape) => void;
  windowCenter: number;
  windowWidth: number;
  opacity: number;
  showContext: boolean;
  selectedLabel: number | null;
  hiddenLabels: Set<number>;
  maxSize: number;
}

const TITLES: Record<Plane, string> = { axial: "Axial", coronal: "Coronal", sagittal: "Sagittal" };
const EDGES: Record<Plane, [string, string, string, string]> = {
  // top, right, bottom, left
  axial: ["A", "L", "P", "R"],
  coronal: ["S", "L", "I", "R"],
  sagittal: ["S", "P", "I", "A"],
};

export function SliceView(props: Props) {
  const { plane, shape, spacing, volume, labels, cursor, onCursor, maxSize } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { cols, rows } = planeSize(plane, shape);
  const { sc, sr } = planeSpacing(plane, spacing);
  const view = voxelToView(plane, shape, cursor);
  const count = sliceCount(plane, shape);

  const widthMm = cols * sc;
  const heightMm = rows * sr;
  const scale = maxSize / Math.max(widthMm, heightMm);
  const dispW = Math.round(widthMm * scale);
  const dispH = Math.round(heightMm * scale);

  const img = useMemo(() => extractSlice(volume, shape, plane, view.index), [volume, shape, plane, view.index]);
  const lab = useMemo(
    () => (labels ? extractSlice(labels, shape, plane, view.index) : null),
    [labels, shape, plane, view.index],
  );

  const selectedBox = useMemo(() => {
    if (!lab || props.selectedLabel == null) return null;
    let c0 = Infinity, r0 = Infinity, c1 = -1, r1 = -1;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        if (lab[r * cols + c] === props.selectedLabel) {
          c0 = Math.min(c0, c); c1 = Math.max(c1, c);
          r0 = Math.min(r0, r); r1 = Math.max(r1, r);
        }
    return c1 < 0 ? null : { c0, r0, c1, r1 };
  }, [lab, props.selectedLabel, rows, cols]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const out = ctx.createImageData(cols, rows);
    const lo = props.windowCenter - props.windowWidth / 2;
    const w = Math.max(1, props.windowWidth);
    const a = props.opacity;
    for (let i = 0; i < cols * rows; i++) {
      let g = ((img[i] - lo) / w) * 255;
      g = g < 0 ? 0 : g > 255 ? 255 : g;
      let r = g, gg = g, b = g;
      const l = lab ? lab[i] : 0;
      const isContext = l === CANAL_LABEL || l === DISC_LABEL;
      if (l && !props.hiddenLabels.has(l) && (!isContext || props.showContext)) {
        const [cr, cg, cb] = labelColor(l);
        const sel = props.selectedLabel === l;
        const alpha = isContext ? a * 0.35 : sel ? Math.min(1, a * 1.3) : a;
        // Outline pixels (label boundary) are drawn solid so small regions stay visible.
        const c = i % cols;
        const edge =
          (c > 0 && lab![i - 1] !== l) || (c < cols - 1 && lab![i + 1] !== l) ||
          (i >= cols && lab![i - cols] !== l) || (i + cols < lab!.length && lab![i + cols] !== l);
        const al = edge && !isContext ? 1 : alpha;
        r = r * (1 - al) + cr * al;
        gg = gg * (1 - al) + cg * al;
        b = b * (1 - al) + cb * al;
      }
      out.data[i * 4] = r;
      out.data[i * 4 + 1] = gg;
      out.data[i * 4 + 2] = b;
      out.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  }, [img, lab, cols, rows, props.windowCenter, props.windowWidth, props.opacity, props.showContext,
      props.selectedLabel, props.hiddenLabels]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * cols);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * rows);
    onCursor(viewToVoxel(plane, shape, Math.min(cols - 1, Math.max(0, col)), Math.min(rows - 1, Math.max(0, row)), view.index));
  }

  function setIndex(index: number) {
    const i = Math.min(count - 1, Math.max(0, index));
    const next = [...cursor] as Shape;
    next[plane === "axial" ? 2 : plane === "coronal" ? 1 : 0] = i;
    onCursor(next);
  }

  const edges = EDGES[plane];
  return (
    <div className="slice-view">
      <div className="slice-head">
        <strong>{TITLES[plane]}</strong>
        <span className="muted">
          {view.index + 1} / {count}
        </span>
      </div>
      <div
        className="slice-canvas-wrap"
        style={{ width: dispW, height: dispH }}
        onWheel={(e) => setIndex(view.index + (e.deltaY > 0 ? 1 : -1))}
      >
        <canvas ref={canvasRef} style={{ width: dispW, height: dispH }} onClick={handleClick} />
        <div className="crosshair-v" style={{ left: ((view.col + 0.5) / cols) * dispW }} />
        <div className="crosshair-h" style={{ top: ((view.row + 0.5) / rows) * dispH }} />
        {selectedBox && (
          <div
            className="select-box"
            style={{
              left: (selectedBox.c0 / cols) * dispW - 4,
              top: (selectedBox.r0 / rows) * dispH - 4,
              width: ((selectedBox.c1 - selectedBox.c0 + 1) / cols) * dispW + 8,
              height: ((selectedBox.r1 - selectedBox.r0 + 1) / rows) * dispH + 8,
            }}
          />
        )}
        <span className="edge edge-top">{edges[0]}</span>
        <span className="edge edge-right">{edges[1]}</span>
        <span className="edge edge-bottom">{edges[2]}</span>
        <span className="edge edge-left">{edges[3]}</span>
      </div>
      <input
        type="range"
        min={0}
        max={count - 1}
        value={view.index}
        onChange={(e) => setIndex(Number(e.target.value))}
        aria-label={`${TITLES[plane]} slice`}
        style={{ width: dispW }}
      />
    </div>
  );
}
