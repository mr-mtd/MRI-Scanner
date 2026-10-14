import type { AnalysisResult, AnalyzeOptions, Health, StudyMeta } from "./types";

async function check(res: Response): Promise<Response> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return res;
}

export async function getHealth(): Promise<Health> {
  return (await check(await fetch("/api/health"))).json();
}

export async function uploadStudy(files: File[]): Promise<StudyMeta> {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  return (await check(await fetch("/api/studies", { method: "POST", body: form }))).json();
}

export async function loadDemo(kind: "brain" | "brain-dicom" | "spine"): Promise<StudyMeta> {
  return (await check(await fetch(`/api/demo/${kind}`, { method: "POST" }))).json();
}

export async function getVolume(id: string): Promise<Uint8Array> {
  return new Uint8Array(await (await check(await fetch(`/api/studies/${id}/volume`))).arrayBuffer());
}

export async function getLabels(id: string): Promise<Uint8Array> {
  return new Uint8Array(await (await check(await fetch(`/api/studies/${id}/labels`))).arrayBuffer());
}

export async function analyze(id: string, opts: AnalyzeOptions): Promise<AnalysisResult> {
  const res = await fetch(`/api/studies/${id}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  return (await check(res)).json();
}
