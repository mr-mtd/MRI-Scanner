import { useEffect, useMemo, useRef, useState } from "react";
import type { Plane, Vec3 } from "./types";
import { planeGeometry, renderSlice, screenToVoxel, voxelToScreen, type RenderOptions } from "./slicing";

export interface Marker {
  id: string;
  voxel: Vec3;
  color: string;
  selected: boolean;
}

interface Props {
  plane: Plane;
  shape: Vec3;
  spacing: Vec3;
  volume: Uint8Array;
  mask: Uint8Array | null;
  cursor: Vec3;
  onCursor: (v: Vec3) => void;
  render: RenderOptions;
  markers: Marker[];
  onMarker: (id: string) => void;
  measuring: boolean;
  maxHeightVh?: number;
}

type Line = { a: [number, number]; b: [number, number] } | null;

export default function PlaneView(p: Props) {
  const geo = useMemo(() => planeGeometry(p.plane, p.shape, p.spacing), [p.plane, p.shape, p.spacing]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const slice = Math.round(p.cursor[geo.axis]);
  const [line, setLine] = useState<Line>(null);
  const dragging = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(geo.width, geo.height);
    renderSlice(img, p.plane, p.shape, slice, p.volume, p.mask, p.render);
    ctx.putImageData(img, 0, 0);
  }, [geo, p.plane, p.shape, slice, p.volume, p.mask, p.render]);

  useEffect(() => setLine(null), [slice]);

  const toScreen = (e: React.MouseEvent): [number, number] => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const c = ((e.clientX - rect.left) / rect.width) * geo.width;
    const r = ((e.clientY - rect.top) / rect.height) * geo.height;
    return [Math.min(geo.width - 1, Math.max(0, c)), Math.min(geo.height - 1, Math.max(0, r))];
  };

  const setCursorFrom = (e: React.MouseEvent) => {
    const [c, r] = toScreen(e);
    const v = screenToVoxel(p.plane, p.shape, slice, Math.floor(c), Math.floor(r));
    p.onCursor(v);
  };

  const onDown = (e: React.MouseEvent) => {
    if (p.measuring || e.shiftKey) {
      const pt = toScreen(e);
      setLine({ a: pt, b: pt });
    } else {
      setCursorFrom(e);
    }
    dragging.current = true;
  };
  const onMove = (e: React.MouseEvent) => {
    if (!dragging.current) return;
    if (line && (p.measuring || e.shiftKey)) setLine({ a: line.a, b: toScreen(e) });
    else if (!p.measuring) setCursorFrom(e);
  };
  const onUp = () => {
    dragging.current = false;
  };

  const onWheel = (e: React.WheelEvent) => {
    const next = [...p.cursor] as Vec3;
    next[geo.axis] = Math.min(geo.depth - 1, Math.max(0, slice + (e.deltaY > 0 ? -1 : 1)));
    p.onCursor(next);
  };

  const [cx, cy] = voxelToScreen(p.plane, p.shape, p.cursor);
  const lengthMm = line
    ? Math.hypot((line.b[0] - line.a[0]) * geo.pxW, (line.b[1] - line.a[1]) * geo.pxH)
    : 0;
  // SVG uses mm units so lines and text are not stretched by anisotropic voxels.
  const W = geo.width * geo.pxW;
  const H = geo.height * geo.pxH;
  const sx = (c: number) => (c + 0.5) * geo.pxW;
  const sy = (r: number) => (r + 0.5) * geo.pxH;
  const fs = Math.max(W, H) / 32;

  const visibleMarkers = p.markers.filter((m) => Math.abs(m.voxel[geo.axis] - slice) <= 3 || m.selected);

  return (
    <div className="plane">
      <div className="plane-header">
        <span className="plane-name">{p.plane}</span>
        <span className="plane-slice">
          {slice + 1}/{geo.depth}
        </span>
      </div>
      <div className="plane-stage" style={{ aspectRatio: `${W} / ${H}`, width: `min(100%, ${(((p.maxHeightVh ?? 72) * W) / H).toFixed(2)}vh)` }}>
        <canvas ref={canvasRef} width={geo.width} height={geo.height} />
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={onUp}
          onMouseLeave={onUp}
          onWheel={onWheel}
          className={p.measuring ? "measuring" : ""}
        >
          <line x1={sx(cx)} x2={sx(cx)} y1={0} y2={H} className="crosshair" />
          <line x1={0} x2={W} y1={sy(cy)} y2={sy(cy)} className="crosshair" />
          {visibleMarkers.map((m) => {
            const [mc, mr] = voxelToScreen(p.plane, p.shape, m.voxel);
            const rad = fs * (m.selected ? 1.3 : 0.9);
            return (
              <g
                key={m.id}
                className="marker"
                onMouseDown={(e) => {
                  e.stopPropagation();
                  p.onMarker(m.id);
                }}
              >
                <circle cx={sx(mc)} cy={sy(mr)} r={rad} stroke={m.color} strokeWidth={fs / 6} fill="none" opacity={m.selected ? 1 : 0.7} />
                <text
                  x={sx(mc) + rad * 1.1 + fs * 3 > W ? sx(mc) - rad * 1.1 : sx(mc) + rad * 1.1}
                  y={sy(mr) - rad * 0.6}
                  textAnchor={sx(mc) + rad * 1.1 + fs * 3 > W ? "end" : "start"}
                  fontSize={fs * 0.9}
                  fill={m.color}
                >
                  {m.id}
                </text>
              </g>
            );
          })}
          {line && (
            <g className="ruler">
              <line x1={sx(line.a[0])} y1={sy(line.a[1])} x2={sx(line.b[0])} y2={sy(line.b[1])} strokeWidth={fs / 8} />
              <text x={sx(line.b[0]) + fs / 2} y={sy(line.b[1])} fontSize={fs}>
                {lengthMm.toFixed(1)} mm
              </text>
            </g>
          )}
          <text x={fs * 0.4} y={H / 2} fontSize={fs} className="orient">{geo.labels.left}</text>
          <text x={W - fs * 1.1} y={H / 2} fontSize={fs} className="orient">{geo.labels.right}</text>
          <text x={W / 2} y={fs * 1.1} fontSize={fs} className="orient">{geo.labels.top}</text>
          <text x={W / 2} y={H - fs * 0.4} fontSize={fs} className="orient">{geo.labels.bottom}</text>
        </svg>
      </div>
      <input
        type="range"
        min={0}
        max={geo.depth - 1}
        value={slice}
        onChange={(e) => {
          const next = [...p.cursor] as Vec3;
          next[geo.axis] = Number(e.target.value);
          p.onCursor(next);
        }}
        aria-label={`${p.plane} slice`}
      />
    </div>
  );
}
