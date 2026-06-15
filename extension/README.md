# Low-Resource Code Summary (VS Code)

Thin client for the pivot-retrieval code summarization pipeline (paper artifact).
Select code → right-click → **Code Summary: Summarize Selection**. Results show
the summary plus a collapsible trace (pivot translation, retrieved examples,
core statement blocks, final prompt).

## Setup

The extension talks to a backend over HTTP. Point it at your backend and
configure the LLM in settings:

```jsonc
{
  "codeSummary.backend.url": "http://100.122.192.123:8000", // your backend
  "codeSummary.backend.autoStart": false,                   // remote backend
  "codeSummary.mode": "online",
  "codeSummary.online.baseUrl": "https://api.deepseek.com/v1",
  "codeSummary.online.model": "deepseek-v4-flash",
  "codeSummary.online.apiKey": "<your-key>"
}
```

The status-bar item shows the active model and toggles online/offline.
