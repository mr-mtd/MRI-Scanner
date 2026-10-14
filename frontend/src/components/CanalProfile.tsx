import type { Disc, Finding, ProfilePoint } from "../types";

interface Props {
  profile: ProfilePoint[];
  discs: Disc[];
  findings: Finding[];
  onPick: (zIndex: number) => void;
}

/** AP canal diameter along the craniocaudal axis, superior at the top like the sagittal view. */
export function CanalProfile({ profile, discs, findings, onPick }: Props) {
  if (!profile.length) return null;
  const W = 300, H = 320, pad = { l: 64, r: 16, t: 12, b: 32 };
  const zMin = Math.min(...profile.map((p) => p.z_mm));
  const zMax = Math.max(...profile.map((p) => p.z_mm));
  const apMax = Math.max(20, ...profile.map((p) => p.ap_mm));
  const x = (ap: number) => pad.l + (ap / apMax) * (W - pad.l - pad.r);
  const y = (z: number) => pad.t + ((zMax - z) / Math.max(1, zMax - zMin)) * (H - pad.t - pad.b);
  const path = profile.map((p, i) => `${i ? "L" : "M"}${x(p.ap_mm).toFixed(1)},${y(p.z_mm).toFixed(1)}`).join(" ");
  const narrow = findings.filter((f) => f.type === "central canal narrowing");
  const dz = profile.length > 1 ? (profile[1].z_mm - profile[0].z_mm) / (profile[1].z_index - profile[0].z_index) : 1;

  return (
    <svg width={W} height={H} className="profile" role="img" aria-label="Canal AP diameter profile">
      {[10, 12].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={pad.t} y2={H - pad.b} className={t === 10 ? "thr-abs" : "thr-rel"} />
          <text x={x(t)} y={H - pad.b + 12} textAnchor="middle" className="axis-text">{t}</text>
        </g>
      ))}
      {narrow.map((f) => {
        const z = Number(f.measurements.centroid_voxel[2]) * dz;
        return <rect key={f.id} x={pad.l} width={W - pad.l - pad.r} y={y(z) - 6} height={12} className="narrow-band" />;
      })}
      {discs.map((d) => (
        <g key={d.level} onClick={() => onPick(d.z_index)} style={{ cursor: "pointer" }}>
          <line x1={pad.l - 4} x2={W - pad.r} y1={y(d.z_mm)} y2={y(d.z_mm)} className="disc-line" />
          <text x={pad.l - 8} y={y(d.z_mm) + 4} textAnchor="end" className="axis-text">{d.level}</text>
        </g>
      ))}
      <path d={path} className="ap-line" />
      <text x={(W + pad.l) / 2} y={H - 4} textAnchor="middle" className="axis-text">AP diameter (mm)</text>
    </svg>
  );
}
