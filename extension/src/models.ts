export interface ModelConfig { base_url: string; api_key: string; model: string; }

export interface VendorPreset {
  id: string;
  label: string;
  baseUrl: string;      // "" for custom
  defaultModel: string;
  needsKey: boolean;
}

export const CLOUD_VENDORS: VendorPreset[] = [
  { id: "deepseek", label: "DeepSeek", baseUrl: "https://api.deepseek.com/v1", defaultModel: "deepseek-chat", needsKey: true },
  { id: "openai", label: "OpenAI", baseUrl: "https://api.openai.com/v1", defaultModel: "gpt-4o-mini", needsKey: true },
  { id: "custom", label: "Custom (OpenAI-compatible)", baseUrl: "", defaultModel: "", needsKey: true },
];

export const LOCAL_PRESETS: VendorPreset[] = [
  { id: "ollama", label: "Ollama", baseUrl: "http://localhost:11434/v1", defaultModel: "qwen2.5-coder", needsKey: false },
  { id: "llamacpp", label: "llama.cpp server", baseUrl: "http://localhost:8080/v1", defaultModel: "local-model", needsKey: false },
  { id: "vllm", label: "vLLM / SGLang", baseUrl: "http://localhost:8000/v1", defaultModel: "local-model", needsKey: false },
  { id: "custom", label: "Custom", baseUrl: "", defaultModel: "", needsKey: false },
];

export function buildTestRequest(
  m: ModelConfig,
): { url: string; headers: Record<string, string>; body: string } {
  const base = m.base_url.replace(/\/+$/, "");
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (m.api_key) headers.Authorization = `Bearer ${m.api_key}`;
  return {
    url: `${base}/chat/completions`,
    headers,
    body: JSON.stringify({ model: m.model, messages: [{ role: "user", content: "ping" }], max_tokens: 1 }),
  };
}

export interface SavedModel {
  mode: string;
  online: { baseUrl: string; apiKey: string; model: string };
  offline: { baseUrl: string; model: string };
}
export interface ModelFormInput { mode: string; base_url: string; api_key: string; model: string; }

// Merge a wizard form submission into the saved config. Never blanks a key:
// if the online key field is empty, the existing key is preserved.
export function mergeModelConfig(existing: SavedModel, form: ModelFormInput): SavedModel {
  const next: SavedModel = {
    mode: form.mode,
    online: { ...existing.online },
    offline: { ...existing.offline },
  };
  if (form.mode === "online") {
    next.online.baseUrl = form.base_url;
    next.online.model = form.model;
    if (form.api_key) next.online.apiKey = form.api_key;
  } else {
    next.offline.baseUrl = form.base_url;
    next.offline.model = form.model;
  }
  return next;
}
