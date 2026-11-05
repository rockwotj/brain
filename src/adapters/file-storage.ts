import { appendFile, mkdir, readdir, readFile, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { Storage } from "../io/storage.ts";

/**
 * Storage on the local file system: one JSON file per key (`kv/`) and one
 * JSON-lines file per log (`logs/`). Inspectable and simple; single process,
 * not built for scale or concurrent writers.
 */
export class FileStorage implements Storage {
  constructor(readonly dir: string) {}

  async get<T>(key: string): Promise<T | undefined> {
    const raw = await readIfExists(this.kvPath(key));
    return raw === undefined ? undefined : (JSON.parse(raw) as T);
  }

  async put(key: string, value: unknown): Promise<void> {
    await mkdir(join(this.dir, "kv"), { recursive: true });
    const path = this.kvPath(key);
    await writeFile(`${path}.tmp`, JSON.stringify(value));
    await rename(`${path}.tmp`, path); // atomic replace: readers never see half a file
  }

  async keys(prefix: string): Promise<string[]> {
    const names = await readdir(join(this.dir, "kv")).catch(notFound([] as string[]));
    return names
      .filter((n) => n.endsWith(".json"))
      .map((n) => decodeURIComponent(n.slice(0, -".json".length)))
      .filter((k) => k.startsWith(prefix))
      .sort();
  }

  async append(log: string, entry: unknown): Promise<void> {
    await mkdir(join(this.dir, "logs"), { recursive: true });
    await appendFile(this.logPath(log), `${JSON.stringify(entry)}\n`);
  }

  async read<T>(log: string): Promise<T[]> {
    const raw = await readIfExists(this.logPath(log));
    if (!raw) return [];
    return raw.split("\n").filter((line) => line.trim()).map((line) => JSON.parse(line) as T);
  }

  private kvPath(key: string): string {
    return join(this.dir, "kv", `${encodeURIComponent(key)}.json`);
  }
  private logPath(log: string): string {
    return join(this.dir, "logs", `${encodeURIComponent(log)}.jsonl`);
  }
}

async function readIfExists(path: string): Promise<string | undefined> {
  return readFile(path, "utf8").catch(notFound(undefined));
}

/** Map "file not found" to a default; every other error (permissions, corruption) propagates. */
function notFound<T>(fallback: T) {
  return (error: unknown): T => {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") return fallback;
    throw error;
  };
}
