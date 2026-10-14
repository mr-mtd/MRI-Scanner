import type { Finding, Mode, ReviewState } from "../types";
import { labelCss } from "../volume";

interface Props {
  mode: Mode;
  findings: Finding[];
  selected: number | null;
  onSelect: (f: Finding) => void;
  onReview: (id: number, state: ReviewState) => void;
  notes: Record<number, string>;
  onNote: (id: number, note: string) => void;
}

const REVIEW_LABELS: [ReviewState, string][] = [
  ["accepted", "Confirm"],
  ["uncertain", "Unsure"],
  ["rejected", "Reject"],
];

function num(v: unknown, digits = 1): string {
  return typeof v === "number" ? v.toFixed(digits) : "–";
}

function locationText(mode: Mode, f: Finding): string {
  const loc = f.location as Record<string, string>;
  if (mode === "spine") return loc.level ?? "–";
  return `${loc.hemisphere ?? ""} ${loc.region_estimate ?? ""}`.trim();
}

function keyMeasure(mode: Mode, f: Finding): string {
  const m = f.measurements;
  if (mode === "brain") {
    return `${num(m.volume_ml, 2)} mL · ⌀max ${num(m.max_diameter_mm)} mm`;
  }
  if (f.type === "central canal narrowing")
    return `AP ${num(m.min_ap_diameter_mm)} mm (ref ${num(m.reference_ap_diameter_mm)}, ratio ${num(m.ap_ratio, 2)})`;
  if (f.type === "posterior disc protrusion") return `${num(m.protrusion_depth_mm)} mm into canal`;
  if (f.type === "disc T2 signal loss") return `signal ${Math.round(Number(m.signal_ratio_to_brightest_discs) * 100)}% of brighter discs`;
  return "";
}

export function FindingsTable({ mode, findings, selected, onSelect, onReview, notes, onNote }: Props) {
  if (!findings.length) return <p className="muted">No candidate regions met the current thresholds.</p>;
  return (
    <table className="findings">
      <thead>
        <tr>
          <th>#</th>
          <th>Candidate</th>
          <th>{mode === "spine" ? "Level" : "Location (approx.)"}</th>
          <th>Measurements</th>
          <th title="Heuristic detector score, not a probability">Score</th>
          <th>Radiologist review</th>
        </tr>
      </thead>
      <tbody>
        {findings.map((f) => (
          <tr key={f.id} className={selected === f.label_value ? "selected" : ""} onClick={() => onSelect(f)}>
            <td>
              <span className="swatch" style={{ background: labelCss(f.label_value) }} />
              {f.id}
            </td>
            <td>
              <div>{f.type}</div>
              {f.severity && <div className="muted small">{f.severity}</div>}
            </td>
            <td>{locationText(mode, f)}</td>
            <td>
              <div>{keyMeasure(mode, f)}</div>
              <div className="muted small">
                RAS ({f.measurements.centroid_ras_mm.map((v) => v.toFixed(0)).join(", ")}) mm
              </div>
            </td>
            <td>{f.score.toFixed(2)}</td>
            <td onClick={(e) => e.stopPropagation()}>
              <div className="review-buttons">
                {REVIEW_LABELS.map(([state, text]) => (
                  <button
                    key={state}
                    className={`review ${state} ${f.review === state ? "active" : ""}`}
                    onClick={() => onReview(f.id, f.review === state ? "pending" : state)}
                  >
                    {text}
                  </button>
                ))}
              </div>
              <input
                className="note"
                placeholder="Note"
                value={notes[f.id] ?? ""}
                onChange={(e) => onNote(f.id, e.target.value)}
              />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
