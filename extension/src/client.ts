import { SummarizeRequest } from "./config";

export interface CoreBlock { text: string; block_type: string; prob: number; }
export interface Example {
  code: string; core_blocks: CoreBlock[]; summary: string; score: number;
}
export interface Trace {
  translation: {
    pivot_code: string; candidates: string[]; selected_score: number;
    repaired: boolean; fell_back: boolean;
  };
  retrieved: Example[];
  core_blocks: CoreBlock[];
  prompt: string;
}
export interface SummarizeResponse {
  summary: string; trace: Trace | null;
  error: string | null; failed_stage: string | null;
}

export async function getHealth(backendUrl: string): Promise<boolean> {
  try {
    const r = await fetch(`${backendUrl}/health`);
    return r.ok;
  } catch {
    return false;
  }
}

export async function postSummarize(
  backendUrl: string, body: SummarizeRequest,
): Promise<SummarizeResponse> {
  const r = await fetch(`${backendUrl}/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    throw new Error(`Backend returned ${r.status}: ${await r.text()}`);
  }
  return (await r.json()) as SummarizeResponse;
}
