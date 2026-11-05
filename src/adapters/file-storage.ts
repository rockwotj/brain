import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { Storage } from "../io/storage";

/** One JSON file per key in a directory. Simple and inspectable; not built for scale. */
export class FileStorage implements Storage {
  constructor(private readonly dir: string) {}

  async get<T>(key: string): Promise<T | undefined> {
    try {
      return JSON.parse(await readFile(this.path(key), "utf8")) as T;
    } catch {
      return undefined;
    }
  }

  async put<T>(key: string, value: T) {
    await mkdir(this.dir, { recursive: true });
    await writeFile(this.path(key), JSON.stringify(value));
  }

  async delete(key: string) {
    await rm(this.path(key), { force: true });
  }

  async list<T>(prefix: string) {
    const names = await readdir(this.dir).catch(() => [] as string[]);
    const keys = names.map((n) => decodeURIComponent(n.replace(/\.json$/, ""))).filter((k) => k.startsWith(prefix)).sort();
    return Promise.all(keys.map(async (key) => ({ key, value: (await this.get<T>(key))! })));
  }

  private path(key: string) {
    return join(this.dir, `${encodeURIComponent(key)}.json`);
  }
}
