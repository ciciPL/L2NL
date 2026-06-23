export const ONLINE_API_KEY_SECRET = "codeSummary.online.apiKey";

export interface SecretStore {
  get(key: string): Thenable<string | undefined> | Promise<string | undefined>;
  store(key: string, value: string): Thenable<void> | Promise<void>;
  delete(key: string): Thenable<void> | Promise<void>;
}

export async function loadOnlineApiKey(secrets: SecretStore, legacyValue = ""): Promise<string> {
  return (await secrets.get(ONLINE_API_KEY_SECRET)) ?? legacyValue;
}

export async function saveOnlineApiKey(secrets: SecretStore, value: string): Promise<void> {
  if (!value) return;
  await secrets.store(ONLINE_API_KEY_SECRET, value);
}
