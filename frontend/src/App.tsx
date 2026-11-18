import { useEffect, useMemo, useState } from "react";
import * as api from "./api";
import { CanalProfile } from "./components/CanalProfile";
import { FindingsTable } from "./components/FindingsTable";
import { SliceView } from "./components/SliceView";
import type { AnalysisResult, AnalyzeOptions, Finding, Health, Mode, Plane, ReviewState, StudyMeta } from "./types";
import type { Shape } from "./volume";

const DEFAULT_OPTS: AnalyzeOptions = {
  mode: "brain",
  method: "baseline",
  sensitivity: 3,
  min_volume_mm3: 30,
  polarity: "hyper",
  use_asymmetry: true,
  already_skull_stripped: false,
  min_protrusion_mm: 3,
};

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function csvCell(v: unknown): string {
  const s = v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [study, setStudy] = useState<StudyMeta | null>(null);
  const [volume, setVolume] = useState<Uint8Array | null>(null);
  const [labels, setLabels] = useState<Uint8Array | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [notes, setNotes] = useState<Record<number, string>>({});
  const [opts, setOpts] = useState<AnalyzeOptions>(DEFAULT_OPTS);
  const [cursor, setCursor] = useState<Shape>([0, 0, 0]);
  const [selected, setSelected] = useState<number | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [wc, setWc] = useState(128);
  const [ww, setWw] = useState(256);
  const [opacity, setOpacity] = useState(0.45);
  const [showContext, setShowContext] = useState(true);
  const [hideRejected, setHideRejected] = useState(true);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    api.getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  async function run<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(label);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function openStudy(meta: StudyMeta) {
    const vol = await api.getVolume(meta.id);
    setStudy(meta);
    setVolume(vol);
    setLabels(null);
    setResult(null);
    setFindings([]);
    setNotes({});
    setSelected(null);
    setCursor(meta.shape.map((n) => Math.floor(n / 2)) as Shape);
    if (meta.suggested_mode) setOpts((o) => ({ ...o, mode: meta.suggested_mode! }));
  }

  const onFiles = (files: File[]) =>
    files.length && run("Loading study…", async () => openStudy(await api.uploadStudy(files)));

  const onDemo = (kind: "brain" | "brain-dicom" | "spine") =>
    run("Generating demo volume…", async () => openStudy(await api.loadDemo(kind)));

  const onAnalyze = () =>
    study &&
    run(opts.mode === "brain" ? "Scanning brain for candidate regions…" : "Measuring lumbar canal and discs…", async () => {
      const res = await api.analyze(study.id, opts);
      const lab = await api.getLabels(study.id);
      setResult(res);
      setFindings(res.findings);
      setLabels(lab);
      setNotes({});
      setSelected(null);
      if (res.findings[0]) focus(res.findings[0]);
      else if (res.mode === "spine" && res.summary.midsagittal_index != null)
        setCursor((c) => [Number(res.summary.midsagittal_index), c[1], c[2]]);
    });

  function focus(f: Finding) {
    setSelected(f.label_value);
    setCursor([f.slice_hint.sagittal, f.slice_hint.coronal, f.slice_hint.axial]);
  }

  function setReview(id: number, state: ReviewState) {
    setFindings((fs) => fs.map((f) => (f.id === id ? { ...f, review: state } : f)));
  }

  const hidden = useMemo(
    () => new Set(hideRejected ? findings.filter((f) => f.review === "rejected").map((f) => f.label_value) : []),
    [findings, hideRejected],
  );

  function report() {
    return {
      generated_at: new Date().toISOString(),
      disclaimer: result?.disclaimer,
      study,
      method: result?.method,
      request: result?.request,
      summary: result?.summary,
      warnings: result?.warnings,
      findings: findings.map((f) => ({ ...f, reviewer_note: notes[f.id] ?? "" })),
      discs: result?.discs,
    };
  }

  function exportCsv() {
    const cols = ["id", "type", "severity", "location", "measurements", "score", "review", "reviewer_note"];
    const rows = findings.map((f) =>
      [f.id, f.type, f.severity ?? "", f.location, f.measurements, f.score, f.review, notes[f.id] ?? ""].map(csvCell).join(","),
    );
    download(`mri-findings-${study?.id}.csv`, [cols.join(","), ...rows].join("\n"), "text/csv");
  }

  const planes: Plane[] = opts.mode === "spine" ? ["sagittal", "axial", "coronal"] : ["axial", "coronal", "sagittal"];
  const reviewed = findings.filter((f) => f.review !== "pending").length;

  return (
    <div className="app">
      <header>
        <h1>MRI Scanner</h1>
        <span className="muted">Candidate lesion detection and lumbar spine measurements for radiologist review</span>
      </header>
      <div className="disclaimer" role="note">
        Research prototype. Highlighted regions are <strong>candidate findings for radiologist review, not diagnoses</strong>.
        Not validated for clinical use.
      </div>

      <div className="layout">
        <aside>
          <section>
            <h2>1. Load MRI</h2>
            <label
              className={`dropzone ${dragOver ? "over" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                onFiles(Array.from(e.dataTransfer.files));
              }}
            >
              <input
                type="file"
                multiple
                accept=".nii,.gz,.dcm,.zip,application/dicom"
                onChange={(e) => onFiles(Array.from(e.target.files ?? []))}
              />
              <span>Drop a NIfTI (.nii / .nii.gz), DICOM slices, or a .zip of one DICOM series</span>
            </label>
            <div className="demo-buttons">
              <span className="muted small">Or load a synthetic demo phantom:</span>
              <button onClick={() => onDemo("brain")}>Demo brain (NIfTI)</button>
              <button onClick={() => onDemo("brain-dicom")}>Demo brain (DICOM)</button>
              <button onClick={() => onDemo("spine")}>Demo lumbar spine</button>
            </div>
            {study && (
              <div className="study-meta small">
                <div><strong>{study.description}</strong> ({study.source_format})</div>
                <div>{study.shape.join(" × ")} voxels · {study.spacing_mm.map((s) => s.toFixed(2)).join(" × ")} mm</div>
                {study.warnings.map((w) => <div key={w} className="warn">{w}</div>)}
              </div>
            )}
          </section>

          <section>
            <h2>2. Analyse</h2>
            <div className="segmented">
              {(["brain", "spine"] as Mode[]).map((m) => (
                <button key={m} className={opts.mode === m ? "active" : ""} onClick={() => setOpts({ ...opts, mode: m })}>
                  {m === "brain" ? "Brain lesions" : "Lumbar spine"}
                </button>
              ))}
            </div>
            {opts.mode === "brain" ? (
              <div className="form">
                <label>
                  Detector
                  <select value={opts.method} onChange={(e) => setOpts({ ...opts, method: e.target.value as AnalyzeOptions["method"] })}>
                    <option value="baseline">Baseline (intensity + asymmetry)</option>
                    <option value="pretrained" disabled={!health?.pretrained_brain_model}>
                      Pretrained model {health?.pretrained_brain_model ? `(${health.pretrained_brain_model})` : "(not configured)"}
                    </option>
                  </select>
                </label>
                <label>
                  Lesion appearance
                  <select value={opts.polarity} onChange={(e) => setOpts({ ...opts, polarity: e.target.value as "hyper" | "hypo" })}>
                    <option value="hyper">Hyperintense (FLAIR / T2 / T1 post-contrast)</option>
                    <option value="hypo">Hypointense (e.g. T1 black holes)</option>
                  </select>
                </label>
                <label>
                  Threshold (robust z): {opts.sensitivity.toFixed(1)}
                  <input type="range" min={1.5} max={8} step={0.1} value={opts.sensitivity}
                    onChange={(e) => setOpts({ ...opts, sensitivity: Number(e.target.value) })} />
                  <span className="muted small">Lower finds more, with more false positives.</span>
                </label>
                <label>
                  Minimum volume (mm³)
                  <input type="number" min={0} value={opts.min_volume_mm3}
                    onChange={(e) => setOpts({ ...opts, min_volume_mm3: Number(e.target.value) })} />
                </label>
                <label className="check">
                  <input type="checkbox" checked={opts.use_asymmetry}
                    onChange={(e) => setOpts({ ...opts, use_asymmetry: e.target.checked })} />
                  Require left/right asymmetry
                </label>
                <label className="check">
                  <input type="checkbox" checked={opts.already_skull_stripped}
                    onChange={(e) => setOpts({ ...opts, already_skull_stripped: e.target.checked })} />
                  Input is already skull-stripped
                </label>
              </div>
            ) : (
              <div className="form">
                <p className="muted small">
                  Focused use case: central canal stenosis and disc abnormalities on a <strong>sagittal T2</strong> lumbar series,
                  measured on the midsagittal slice.
                </p>
                <label>
                  Minimum disc protrusion (mm)
                  <input type="number" min={0.5} step={0.5} value={opts.min_protrusion_mm}
                    onChange={(e) => setOpts({ ...opts, min_protrusion_mm: Number(e.target.value) })} />
                </label>
              </div>
            )}
            <button className="primary" disabled={!study || !!busy} onClick={onAnalyze}>
              Run analysis
            </button>
          </section>

          <section>
            <h2>Display</h2>
            <div className="form">
              <label>Window level {wc}
                <input type="range" min={0} max={255} value={wc} onChange={(e) => setWc(Number(e.target.value))} />
              </label>
              <label>Window width {ww}
                <input type="range" min={10} max={400} value={ww} onChange={(e) => setWw(Number(e.target.value))} />
              </label>
              <label>Overlay opacity {Math.round(opacity * 100)}%
                <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
              </label>
              <label className="check">
                <input type="checkbox" checked={showContext} onChange={(e) => setShowContext(e.target.checked)} />
                Show canal / disc outlines (spine)
              </label>
              <label className="check">
                <input type="checkbox" checked={hideRejected} onChange={(e) => setHideRejected(e.target.checked)} />
                Hide rejected candidates
              </label>
            </div>
          </section>
        </aside>

        <main>
          {busy && <div className="busy">{busy}</div>}
          {error && <div className="error">{error}</div>}
          {!study && !busy && (
            <div className="empty">Load a study or a demo phantom to begin.</div>
          )}
          {study && volume && (
            <>
              <div className="views">
                {planes.map((p, i) => (
                  <SliceView
                    key={p}
                    plane={p}
                    shape={study.shape}
                    spacing={study.spacing_mm}
                    volume={volume}
                    labels={labels}
                    cursor={cursor}
                    onCursor={setCursor}
                    windowCenter={wc}
                    windowWidth={ww}
                    opacity={opacity}
                    showContext={showContext}
                    selectedLabel={selected}
                    hiddenLabels={hidden}
                    maxSize={opts.mode === "spine" && i === 0 ? 440 : 300}
                  />
                ))}
                {result?.mode === "spine" && result.profile && (
                  <div className="slice-view">
                    <div className="slice-head"><strong>Canal AP diameter</strong></div>
                    <CanalProfile
                      profile={result.profile}
                      discs={result.discs ?? []}
                      findings={findings}
                      onPick={(z) => setCursor((c) => [c[0], c[1], z])}
                    />
                    <span className="muted small">Dashed lines: 12 mm relative, 10 mm absolute stenosis.</span>
                  </div>
                )}
              </div>
              <div className="cursor-readout small muted">
                Cursor voxel ({cursor.join(", ")}) — click any view to move, scroll to change slice
              </div>
            </>
          )}

          {result && (
            <section className="results">
              <div className="results-head">
                <h2>Candidate findings ({findings.length})</h2>
                <span className="muted small">
                  Reviewed {reviewed}/{findings.length} · method: {result.method.name}
                </span>
                <div className="spacer" />
                <button onClick={() => download(`mri-report-${study?.id}.json`, JSON.stringify(report(), null, 2), "application/json")}>
                  Export JSON
                </button>
                <button onClick={exportCsv}>Export CSV</button>
              </div>
              <div className="summary small">
                {Object.entries(result.summary).map(([k, v]) => (
                  <span key={k}><span className="muted">{k.replace(/_/g, " ")}:</span> {String(v)}</span>
                ))}
              </div>
              {result.warnings.map((w) => <div key={w} className="warn small">{w}</div>)}
              <FindingsTable
                mode={result.mode}
                findings={findings}
                selected={selected}
                onSelect={focus}
                onReview={setReview}
                notes={notes}
                onNote={(id, note) => setNotes((n) => ({ ...n, [id]: note }))}
              />
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
