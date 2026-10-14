import { useCallback, useMemo, useState } from "react";
import { analyze, fetchBytes, loadDemo, uploadStudy } from "./api";
import Findings from "./Findings";
import PlaneView, { type Marker } from "./PlaneView";
import { LABEL_COLORS, type RenderOptions } from "./slicing";
import type {
  AnalysisResult,
  BrainFinding,
  BrainOptions,
  Mode,
  Plane,
  ReviewStatus,
  SpineFinding,
  SpineOptions,
  StudyResponse,
  Vec3,
} from "./types";

const DEFAULT_BRAIN: BrainOptions = { sensitivity: 3, polarity: "hyper", min_volume_ml: 0.05, require_asymmetry: true, engine: "auto" };
const DEFAULT_SPINE: SpineOptions = {
  lowest_level: "L5-S1",
  absolute_stenosis_mm: 10,
  relative_stenosis_mm: 12,
  narrowing_ratio: 0.67,
  displacement_mm: 3,
  sagittal_slice: null,
};
const LEVELS = ["T10-T11", "T11-T12", "T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"];
const SEVERITY_COLORS = ["#4ade80", "#fbbf24", "#fb923c", "#f43f5e"];

interface Loaded {
  study: StudyResponse;
  volume: Uint8Array;
}

function guessMode(study: StudyResponse): Mode {
  const text = `${study.info.description} ${study.info.modalityHint} ${String(study.info.meta.body_part ?? "")}`.toLowerCase();
  return /spine|lumbar|l-?spine|lspine|sag|disc|cord/.test(text) ? "spine" : "brain";
}

function center(shape: Vec3): Vec3 {
  return shape.map((n) => Math.floor(n / 2)) as Vec3;
}

export default function App() {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [mask, setMask] = useState<Uint8Array | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [mode, setMode] = useState<Mode>("brain");
  const [brainOpts, setBrainOpts] = useState(DEFAULT_BRAIN);
  const [spineOpts, setSpineOpts] = useState(DEFAULT_SPINE);
  const [cursor, setCursor] = useState<Vec3>([0, 0, 0]);
  const [primary, setPrimary] = useState<Plane>("axial");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [review, setReview] = useState<Record<string, ReviewStatus>>({});
  const [dragOver, setDragOver] = useState(false);
  const [measuring, setMeasuring] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [outline, setOutline] = useState(false);
  const [opacity, setOpacity] = useState(0.45);
  const [windowLow, setWindowLow] = useState(0);
  const [windowHigh, setWindowHigh] = useState(255);

  const render: RenderOptions = useMemo(
    () => ({ windowLow, windowHigh, overlayOpacity: opacity, outline, showOverlay }),
    [windowLow, windowHigh, opacity, outline, showOverlay],
  );

  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const openStudy = async (study: StudyResponse, forceMode?: Mode) => {
    const volume = await fetchBytes(`/api/studies/${study.id}/volume`);
    const m = forceMode ?? guessMode(study);
    setLoaded({ study, volume });
    setMask(null);
    setResult(null);
    setSelected(null);
    setReview({});
    setMode(m);
    setPrimary(m === "spine" ? "sagittal" : "axial");
    setCursor(center(study.info.shape));
    setSpineOpts((o) => ({ ...o, sagittal_slice: null }));
  };

  const onFiles = (files: File[]) => {
    if (!files.length) return;
    run("Loading study…", async () => openStudy(await uploadStudy(files)));
  };

  const onDemo = (kind: Mode) => run("Loading demo…", async () => openStudy(await loadDemo(kind), kind));

  const onAnalyze = () => {
    if (!loaded) return;
    run(mode === "brain" ? "Scanning for candidate lesions…" : "Measuring discs and canal…", async () => {
      const res = await analyze(loaded.study.id, mode, brainOpts, spineOpts);
      const m = await fetchBytes(`/api/studies/${loaded.study.id}/mask`);
      setResult(res);
      setMask(m);
      setReview({});
      setShowOverlay(true);
      const first = res.findings[0];
      if (res.summary.mode === "spine") {
        setPrimary("sagittal");
        const worst = [...(res.findings as SpineFinding[])].sort((a, b) => b.severity - a.severity)[0];
        if (worst) jumpTo(worst.id, res);
        else setCursor((c) => [res.summary.sagittalSlice ?? c[0], c[1], c[2]]);
      } else if (first) {
        jumpTo(first.id, res);
      } else {
        setSelected(null);
      }
    });
  };

  const jumpTo = useCallback(
    (id: string, res: AnalysisResult | null = result) => {
      const f = res?.findings.find((x) => x.id === id);
      if (!f) return;
      setSelected(id);
      setCursor(f.centroidVoxel.map((v) => Math.round(v)) as Vec3);
    },
    [result],
  );

  const markers: Marker[] = useMemo(() => {
    if (!result) return [];
    return result.findings
      .filter((f) => result.summary.mode === "brain" || (f as SpineFinding).severity > 0 || f.id === selected)
      .map((f) => ({
        id: f.id,
        voxel: f.centroidVoxel,
        color: f.kind === "lesion" ? "#ff4d4f" : SEVERITY_COLORS[(f as SpineFinding).severity],
        selected: f.id === selected,
      }));
  }, [result, selected]);

  const exportReport = (format: "json" | "csv") => {
    if (!result || !loaded) return;
    const findings = result.findings.map((f) => ({ ...f, reviewStatus: review[f.id] ?? "pending" }));
    let blob: Blob;
    if (format === "json") {
      const report = { generatedAt: new Date().toISOString(), disclaimer: result.disclaimer, study: result.study, summary: result.summary, findings };
      blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    } else {
      const rows =
        result.summary.mode === "brain"
          ? [
              ["id", "location", "volume_ml", "max_diameter_mm", "extent_rl_mm", "extent_ap_mm", "extent_si_mm", "centroid_x_mm", "centroid_y_mm", "centroid_z_mm", "score", "review"],
              ...(findings as (BrainFinding & { reviewStatus: string })[]).map((f) => [
                f.id, f.location.text, f.volumeMl, f.maxDiameterMm, ...f.extentMm, ...f.centroidWorldMm, f.confidence, f.reviewStatus,
              ]),
            ]
          : [
              ["level", "canal_ap_mm", "canal_ap_ratio", "posterior_displacement_mm", "disc_height_mm", "disc_signal_ratio", "flags", "review"],
              ...(findings as (SpineFinding & { reviewStatus: string })[]).map((f) => [
                f.level, f.canalApMm, f.canalApRatio ?? "", f.posteriorDisplacementMm, f.discHeightMm, f.discSignalRatio, f.flags.map((x) => x.text).join("; "), f.reviewStatus,
              ]),
            ];
      const csv = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
      blob = new Blob([`# ${result.disclaimer}\n${csv}\n`], { type: "text/csv" });
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `mri_candidates_${result.summary.mode}_${loaded.study.id}.${format}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  const planes: Plane[] = ["axial", "coronal", "sagittal"];
  const secondary = planes.filter((pl) => pl !== primary);
  const info = loaded?.study.info;
  const labels = result?.summary.labels ?? {};

  return (
    <div
      className={`app ${dragOver ? "drag" : ""}`}
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
      <header>
        <h1>MRI Scanner</h1>
        <p className="disclaimer">
          Research prototype. Highlights <b>candidate</b> findings for radiologist review. Not a diagnosis and not for clinical use.
        </p>
      </header>

      <aside>
        <section>
          <h2>Study</h2>
          <label className="dropzone">
            <input type="file" multiple accept=".nii,.gz,.dcm,.zip,application/dicom" onChange={(e) => onFiles(Array.from(e.target.files ?? []))} />
            <span>Drop or choose NIfTI (.nii, .nii.gz), DICOM files, or a .zip of a DICOM series</span>
          </label>
          <label className="linkish">
            <input
              type="file"
              // @ts-expect-error non-standard but widely supported folder picker
              webkitdirectory=""
              onChange={(e) => onFiles(Array.from(e.target.files ?? []))}
            />
            Choose a DICOM folder…
          </label>
          <div className="row">
            <button onClick={() => onDemo("brain")}>Demo brain</button>
            <button onClick={() => onDemo("spine")}>Demo lumbar spine</button>
          </div>
          <p className="small muted">
            Demo files to test upload: <a href="/api/demo/brain/download">brain .nii</a> · <a href="/api/demo/spine/download">spine DICOM .zip</a>
          </p>
          {info && (
            <dl className="info">
              <dt>Series</dt>
              <dd>{info.description || "—"}</dd>
              <dt>Format</dt>
              <dd>{info.sourceFormat.toUpperCase()}</dd>
              <dt>Matrix</dt>
              <dd>{info.shape.join(" × ")}</dd>
              <dt>Voxel</dt>
              <dd>{info.spacing.map((s) => s.toFixed(2)).join(" × ")} mm</dd>
              {typeof info.meta.note === "string" && (
                <>
                  <dt>Note</dt>
                  <dd>{info.meta.note}</dd>
                </>
              )}
            </dl>
          )}
        </section>

        {loaded && (
          <section>
            <h2>Analysis</h2>
            <div className="segmented">
              <button className={mode === "brain" ? "on" : ""} onClick={() => setMode("brain")}>Brain lesions</button>
              <button className={mode === "spine" ? "on" : ""} onClick={() => setMode("spine")}>Lumbar spine</button>
            </div>
            {mode === "brain" ? (
              <div className="form">
                <p className="small muted">Best on FLAIR, T2 or post-contrast T1.</p>
                <label>
                  Sensitivity threshold (z) <b>{brainOpts.sensitivity.toFixed(1)}</b>
                  <input type="range" min={1.5} max={8} step={0.1} value={brainOpts.sensitivity} onChange={(e) => setBrainOpts({ ...brainOpts, sensitivity: Number(e.target.value) })} />
                </label>
                <label>
                  Lesion signal
                  <select value={brainOpts.polarity} onChange={(e) => setBrainOpts({ ...brainOpts, polarity: e.target.value as BrainOptions["polarity"] })}>
                    <option value="hyper">Hyperintense (FLAIR / T2 / T1+C)</option>
                    <option value="hypo">Hypointense</option>
                    <option value="both">Both</option>
                  </select>
                </label>
                <label>
                  Minimum volume (mL)
                  <input type="number" min={0} step={0.01} value={brainOpts.min_volume_ml} onChange={(e) => setBrainOpts({ ...brainOpts, min_volume_ml: Number(e.target.value) })} />
                </label>
                <label className="check">
                  <input type="checkbox" checked={brainOpts.require_asymmetry} onChange={(e) => setBrainOpts({ ...brainOpts, require_asymmetry: e.target.checked })} />
                  Suppress left/right symmetric signal (ventricles, normal anatomy)
                </label>
                <label>
                  Engine
                  <select value={brainOpts.engine} onChange={(e) => setBrainOpts({ ...brainOpts, engine: e.target.value as BrainOptions["engine"] })}>
                    <option value="auto">Auto (model if configured)</option>
                    <option value="heuristic">Heuristic</option>
                    <option value="onnx">ONNX model</option>
                  </select>
                </label>
              </div>
            ) : (
              <div className="form">
                <p className="small muted">Sagittal T2 lumbar spine: disc levels and central canal stenosis.</p>
                <label>
                  Lowest disc is
                  <select value={spineOpts.lowest_level} onChange={(e) => setSpineOpts({ ...spineOpts, lowest_level: e.target.value })}>
                    {LEVELS.map((l) => <option key={l}>{l}</option>)}
                  </select>
                </label>
                <label>
                  Absolute stenosis below (mm)
                  <input type="number" step={0.5} value={spineOpts.absolute_stenosis_mm} onChange={(e) => setSpineOpts({ ...spineOpts, absolute_stenosis_mm: Number(e.target.value) })} />
                </label>
                <label>
                  Relative stenosis below (mm)
                  <input type="number" step={0.5} value={spineOpts.relative_stenosis_mm} onChange={(e) => setSpineOpts({ ...spineOpts, relative_stenosis_mm: Number(e.target.value) })} />
                </label>
                <label>
                  Flag disc displacement from (mm)
                  <input type="number" step={0.5} value={spineOpts.displacement_mm} onChange={(e) => setSpineOpts({ ...spineOpts, displacement_mm: Number(e.target.value) })} />
                </label>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={spineOpts.sagittal_slice !== null}
                    onChange={(e) => setSpineOpts({ ...spineOpts, sagittal_slice: e.target.checked ? cursor[0] : null })}
                  />
                  Use current sagittal slice ({cursor[0] + 1}) instead of auto
                </label>
              </div>
            )}
            <button className="primary" disabled={!!busy} onClick={onAnalyze}>
              {mode === "brain" ? "Scan for candidate lesions" : "Analyse lumbar spine"}
            </button>
          </section>
        )}

        {loaded && (
          <section>
            <h2>Display</h2>
            <div className="form">
              <label className="check">
                <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} disabled={!mask} />
                Show candidate overlay
              </label>
              <label className="check">
                <input type="checkbox" checked={outline} onChange={(e) => setOutline(e.target.checked)} />
                Outline only
              </label>
              <label>
                Overlay opacity
                <input type="range" min={0.1} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
              </label>
              <label>
                Window low / high
                <div className="row">
                  <input type="range" min={0} max={254} value={windowLow} onChange={(e) => setWindowLow(Math.min(Number(e.target.value), windowHigh - 1))} />
                  <input type="range" min={1} max={255} value={windowHigh} onChange={(e) => setWindowHigh(Math.max(Number(e.target.value), windowLow + 1))} />
                </div>
              </label>
              <label className="check">
                <input type="checkbox" checked={measuring} onChange={(e) => setMeasuring(e.target.checked)} />
                Ruler (or hold Shift and drag)
              </label>
            </div>
            {Object.keys(labels).length > 0 && (
              <ul className="legend">
                {Object.entries(labels).map(([k, v]) => {
                  const c = LABEL_COLORS[Number(k)] ?? [255, 255, 255];
                  return (
                    <li key={k}>
                      <span className="swatch" style={{ background: `rgb(${c.join(",")})` }} />
                      {v}
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        )}

        {result && (
          <section>
            <h2>Export</h2>
            <div className="row">
              <button onClick={() => exportReport("json")}>Report JSON</button>
              <button onClick={() => exportReport("csv")}>Table CSV</button>
              <a className="button" href={`/api/studies/${loaded?.study.id}/mask.nii.gz`}>Mask NIfTI</a>
            </div>
          </section>
        )}
      </aside>

      <main>
        {busy && <div className="banner busy">{busy}</div>}
        {error && (
          <div className="banner error" onClick={() => setError(null)}>
            {error}
          </div>
        )}
        {!loaded ? (
          <div className="empty">
            <h2>Load a brain or spine MRI to begin</h2>
            <p>
              Upload a NIfTI volume or a DICOM series, or try a synthetic demo. Brain mode segments candidate lesions and reports their location,
              volume and size. Spine mode focuses on one question: lumbar disc abnormalities and central canal stenosis on sagittal T2.
            </p>
          </div>
        ) : (
          <>
            <div className="viewer">
              <div className="primary-plane">
                <PlaneView
                  plane={primary}
                  shape={loaded.study.info.shape}
                  spacing={loaded.study.info.spacing}
                  volume={loaded.volume}
                  mask={mask}
                  cursor={cursor}
                  onCursor={setCursor}
                  render={render}
                  markers={markers}
                  onMarker={(id) => jumpTo(id)}
                  measuring={measuring}
                />
              </div>
              <div className="secondary-planes">
                {secondary.map((pl) => (
                  <div key={pl} className="secondary" onDoubleClick={() => setPrimary(pl)} title="Double-click to enlarge">
                    <PlaneView
                      plane={pl}
                      shape={loaded.study.info.shape}
                      spacing={loaded.study.info.spacing}
                      volume={loaded.volume}
                      mask={mask}
                      cursor={cursor}
                      onCursor={setCursor}
                      render={render}
                      markers={markers}
                      onMarker={(id) => jumpTo(id)}
                      measuring={measuring}
                      maxHeightVh={32}
                    />
                  </div>
                ))}
              </div>
            </div>
            {result ? (
              <>
                <Findings result={result} selected={selected} onSelect={(id) => jumpTo(id)} review={review} onReview={(id, s) => setReview((r) => ({ ...r, [id]: s }))} />
                <ul className="notes">
                  {result.summary.notes.map((n) => <li key={n}>{n}</li>)}
                </ul>
              </>
            ) : (
              <p className="muted">Choose an analysis on the left and run it to highlight candidate regions.</p>
            )}
          </>
        )}
      </main>
    </div>
  );
}
