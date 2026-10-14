import type { AnalysisResult, BrainFinding, CanalProfile, ReviewStatus, SpineFinding } from "./types";

interface Props {
  result: AnalysisResult;
  selected: string | null;
  onSelect: (id: string) => void;
  review: Record<string, ReviewStatus>;
  onReview: (id: string, s: ReviewStatus) => void;
}

function ReviewButtons({ id, status, onReview }: { id: string; status: ReviewStatus; onReview: Props["onReview"] }) {
  return (
    <span className="review" onClick={(e) => e.stopPropagation()}>
      <button className={status === "accepted" ? "on accept" : ""} title="Agree with candidate" onClick={() => onReview(id, status === "accepted" ? "pending" : "accepted")}>
        ✓
      </button>
      <button className={status === "rejected" ? "on reject" : ""} title="Reject (false positive)" onClick={() => onReview(id, status === "rejected" ? "pending" : "rejected")}>
        ✕
      </button>
    </span>
  );
}

export default function Findings(p: Props) {
  const { summary, findings } = p.result;
  if (summary.mode === "brain") {
    const list = findings as BrainFinding[];
    return (
      <section className="findings">
        <div className="stats">
          <Stat label="Candidates" value={String(summary.candidateCount ?? 0)} />
          <Stat label="Total volume" value={`${(summary.totalLesionVolumeMl ?? 0).toFixed(2)} mL`} />
          <Stat label="Brain mask" value={`${Math.round(summary.brainVolumeMl ?? 0)} mL`} />
          <Stat label="Engine" value={summary.engine} />
        </div>
        {list.length === 0 ? (
          <p className="muted">No candidate regions above the current sensitivity. Try lowering the threshold.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Location</th>
                <th className="num">Volume</th>
                <th className="num">Max ⌀</th>
                <th className="num">Extent (RL×AP×SI)</th>
                <th className="num">Score</th>
                <th>Review</th>
              </tr>
            </thead>
            <tbody>
              {list.map((f) => (
                <tr key={f.id} className={rowClass(f.id, p)} onClick={() => p.onSelect(f.id)}>
                  <td><b>{f.id}</b></td>
                  <td>{f.location.text}</td>
                  <td className="num">{f.volumeMl.toFixed(2)} mL</td>
                  <td className="num">{f.maxDiameterMm.toFixed(1)} mm</td>
                  <td className="num">{f.extentMm.map((e) => e.toFixed(0)).join("×")} mm</td>
                  <td className="num">{f.confidence.toFixed(2)}</td>
                  <td><ReviewButtons id={f.id} status={p.review[f.id] ?? "pending"} onReview={p.onReview} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    );
  }

  const list = findings as SpineFinding[];
  return (
    <section className="findings">
      <div className="stats">
        <Stat label="Discs found" value={String(summary.discsFound ?? 0)} />
        <Stat label="Flagged levels" value={(summary.flaggedLevels ?? []).join(", ") || "none"} />
        <Stat label="Reference canal AP" value={`${summary.referenceCanalApMm?.toFixed(1)} mm`} />
        <Stat label="Sagittal slice" value={String((summary.sagittalSlice ?? 0) + 1)} />
      </div>
      {summary.canalProfile && <CanalChart profile={summary.canalProfile} params={summary.params} flagged={summary.flaggedLevels ?? []} />}
      <table>
        <thead>
          <tr>
            <th>Level</th>
            <th className="num">Canal AP</th>
            <th className="num">Disc displ.</th>
            <th className="num">Disc height</th>
            <th className="num">T2 signal</th>
            <th>Candidate findings</th>
            <th>Review</th>
          </tr>
        </thead>
        <tbody>
          {list.map((f) => (
            <tr key={f.id} className={`${rowClass(f.id, p)} sev${f.severity}`} onClick={() => p.onSelect(f.id)}>
              <td><b>{f.level}</b></td>
              <td className="num">{f.canalApMm.toFixed(1)} mm</td>
              <td className="num">{f.posteriorDisplacementMm.toFixed(1)} mm</td>
              <td className="num">{f.discHeightMm.toFixed(1)} mm</td>
              <td className="num">{Math.round(f.discSignalRatio * 100)}%</td>
              <td>{f.flags.length ? <ul className="flags">{f.flags.map((fl) => <li key={fl.code}>{fl.text}</li>)}</ul> : <span className="muted">none</span>}</td>
              <td><ReviewButtons id={f.id} status={p.review[f.id] ?? "pending"} onReview={p.onReview} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function rowClass(id: string, p: Props) {
  return [id === p.selected ? "selected" : "", p.review[id] ?? ""].join(" ");
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function CanalChart({ profile, params, flagged }: { profile: CanalProfile; params: Record<string, unknown>; flagged: string[] }) {
  const W = 560, H = 170, L = 40, R = 10, T = 10, B = 26;
  const zs = profile.zMm;
  if (zs.length < 2) return null;
  const zMin = Math.min(...zs), zMax = Math.max(...zs);
  const apMax = Math.max(20, ...profile.apMm) * 1.05;
  // Cranial on the left so the chart reads like the level table (top to bottom).
  const x = (z: number) => L + ((zMax - z) / (zMax - zMin || 1)) * (W - L - R);
  const y = (ap: number) => T + (1 - ap / apMax) * (H - T - B);
  const path = zs.map((z, i) => `${i ? "L" : "M"}${x(z).toFixed(1)},${y(profile.apMm[i]).toFixed(1)}`).join("");
  const abs = Number(params.absolute_stenosis_mm ?? 10);
  const rel = Number(params.relative_stenosis_mm ?? 12);
  return (
    <figure className="chart">
      <figcaption>Central canal AP diameter along the lumbar spine (mid-sagittal)</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Canal AP diameter chart">
        <rect x={L} y={y(abs)} width={W - L - R} height={H - B - y(abs)} className="zone-severe" />
        <rect x={L} y={y(rel)} width={W - L - R} height={y(abs) - y(rel)} className="zone-relative" />
        {[0, 5, 10, 15, 20].filter((v) => v < apMax).map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="grid" />
            <text x={L - 6} y={y(v) + 4} textAnchor="end" className="tick">{v}</text>
          </g>
        ))}
        <text x={12} y={T + (H - T - B) / 2} className="tick" transform={`rotate(-90 12 ${T + (H - T - B) / 2})`} textAnchor="middle">mm</text>
        {profile.levels.map((lv) => (
          <g key={lv.level}>
            <line x1={x(lv.zMm)} x2={x(lv.zMm)} y1={T} y2={H - B} className={flagged.includes(lv.level) ? "level flagged" : "level"} />
            <text x={x(lv.zMm)} y={H - 8} textAnchor="middle" className={flagged.includes(lv.level) ? "tick flagged" : "tick"}>{lv.level}</text>
          </g>
        ))}
        <path d={path} className="series" />
      </svg>
    </figure>
  );
}
