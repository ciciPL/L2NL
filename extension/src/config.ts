export interface RawSettings {
  mode: "online" | "offline";
  online: { baseUrl: string; apiKey: string; model: string };
  offline: { baseUrl: string; model: string };
  params: {
    k: number; temperatures: number[]; lambda: number;
    threshold: number; maxRepairIters: number;
  };
}

export interface SummarizeRequest {
  code: string;
  language: string;
  model: { base_url: string; api_key: string; model: string };
  params: {
    k: number; temperatures: number[]; lambda: number;
    threshold: number; max_repair_iters: number;
  };
  trace: boolean;
}

export function activeModel(s: RawSettings) {
  return s.mode === "offline"
    ? { base_url: s.offline.baseUrl, api_key: "sk-no-key", model: s.offline.model }
    : { base_url: s.online.baseUrl, api_key: s.online.apiKey, model: s.online.model };
}

export function buildRequest(
  code: string, language: string, s: RawSettings,
): SummarizeRequest {
  return {
    code,
    language,
    model: activeModel(s),
    params: {
      k: s.params.k,
      temperatures: s.params.temperatures,
      lambda: s.params.lambda,
      threshold: s.params.threshold,
      max_repair_iters: s.params.maxRepairIters,
    },
    trace: true,
  };
}
