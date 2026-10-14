import type { AnalysisResult, BrainOptions, Mode, SpineOptions, StudyResponse } from "./types";

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

export async function uploadStudy(files: File[]): Promise<StudyResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f, (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name);
  const res = await check(await fetch("/api/studies", { method: "POST", body: form }));
  return res.json();
}

export async function loadDemo(kind: Mode): Promise<StudyResponse> {
  const res = await check(await fetch(`/api/demo/${kind}`, { method: "POST" }));
  return res.json();
}

export async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await check(await fetch(url));
  return new Uint8Array(await res.arrayBuffer());
}

export async function analyze(
  id: string,
  mode: Mode,
  brain: BrainOptions,
  spine: SpineOptions,
): Promise<AnalysisResult> {
  const res = await check(
    await fetch(`/api/studies/${id}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, brain, spine }),
    }),
  );
  return res.json();
}
