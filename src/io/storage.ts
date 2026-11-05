/**
 * All persistence goes through this small async port: JSON values by key plus
 * append-only logs. Implementations: `MemoryStorage` (below) and
 * `FileStorage` (src/adapters/file-storage.ts). Values must be JSON-serializable.
 * Stored data is trusted (written by this process), so reads are not re-validated.
 */
export interface Storage {
  get<T>(key: string): Promise<T | undefined>;
  put(key: string, value: unknown): Promise<void>;
  /** Keys starting with `prefix`, sorted. */
  keys(prefix: string): Promise<string[]>;
  append(log: string, entry: unknown): Promise<void>;
  /** All entries of a log in append order ([] if it does not exist). */
  read<T>(log: string): Promise<T[]>;
}

/** Process-local storage. Values are JSON round-tripped, like the file adapter. */
export class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();
  private readonly logs = new Map<string, string[]>();

  async get<T>(key: string): Promise<T | undefined> {
    const raw = this.values.get(key);
    return raw === undefined ? undefined : (JSON.parse(raw) as T);
  }
  async put(key: string, value: unknown): Promise<void> {
    this.values.set(key, JSON.stringify(value));
  }
  async keys(prefix: string): Promise<string[]> {
    return [...this.values.keys()].filter((k) => k.startsWith(prefix)).sort();
  }
  async append(log: string, entry: unknown): Promise<void> {
    this.logs.set(log, [...(this.logs.get(log) ?? []), JSON.stringify(entry)]);
  }
  async read<T>(log: string): Promise<T[]> {
    return (this.logs.get(log) ?? []).map((raw) => JSON.parse(raw) as T);
  }
}
