import { describe, it, expect } from "vitest";
import { loadOnlineApiKey, saveOnlineApiKey, ONLINE_API_KEY_SECRET } from "../src/secrets";

class FakeSecrets {
  values = new Map<string, string>();
  async get(k: string) { return this.values.get(k); }
  async store(k: string, v: string) { this.values.set(k, v); }
  async delete(k: string) { this.values.delete(k); }
}

describe("online API key secret storage", () => {
  it("prefers SecretStorage over the deprecated settings value", async () => {
    const secrets = new FakeSecrets();
    await secrets.store(ONLINE_API_KEY_SECRET, "SECRET");
    expect(await loadOnlineApiKey(secrets, "LEGACY")).toBe("SECRET");
  });

  it("falls back to the deprecated settings value during migration", async () => {
    expect(await loadOnlineApiKey(new FakeSecrets(), "LEGACY")).toBe("LEGACY");
  });

  it("does not persist blank form submissions over an existing key", async () => {
    const secrets = new FakeSecrets();
    await secrets.store(ONLINE_API_KEY_SECRET, "OLD");
    await saveOnlineApiKey(secrets, "");
    expect(await loadOnlineApiKey(secrets, "")).toBe("OLD");
  });

  it("stores a non-empty key in SecretStorage", async () => {
    const secrets = new FakeSecrets();
    await saveOnlineApiKey(secrets, "NEW");
    expect(secrets.values.get(ONLINE_API_KEY_SECRET)).toBe("NEW");
  });
});
