/**
 * All persistence goes through this tiny key-value interface. Keys are "/"-separated paths
 * ("episodes/000123"), and list() is a prefix scan in key order.
 * Swap in SQLite, Redis, S3, a filesystem... (see src/adapters/file-storage.ts).
 */
export interface Storage {
  get<T>(key: string): Promise<T | undefined>;
  put<T>(key: string, value: T): Promise<void>;
  delete(key: string): Promise<void>;
  list<T>(prefix: string): Promise<{ key: string; value: T }[]>;
}

export class MemoryStorage implements Storage {
  private data = new Map<string, unknown>();

  async get<T>(key: string) {
    return structuredClone(this.data.get(key)) as T | undefined;
  }

  async put<T>(key: string, value: T) {
    this.data.set(key, structuredClone(value));
  }

  async delete(key: string) {
    this.data.delete(key);
  }

  async list<T>(prefix: string) {
    return [...this.data.keys()]
      .filter((key) => key.startsWith(prefix))
      .sort()
      .map((key) => ({ key, value: structuredClone(this.data.get(key)) as T }));
  }
}
