/**
 * TEST FIXTURE ONLY. A tiny deterministic embedder (hashed bag of words) so
 * recall tests can exercise real cosine ranking without a provider. It has no
 * semantic understanding beyond shared words; never use it outside tests.
 */
import type { Embedder } from "../../src/index.ts";

export class WordHashEmbedder implements Embedder {
  readonly calls: string[][] = [];
  constructor(private readonly dims = 512) {}

  async embed(texts: readonly string[]): Promise<number[][]> {
    this.calls.push([...texts]);
    return texts.map((t) => {
      const v = new Array<number>(this.dims).fill(0);
      for (const word of t.toLowerCase().match(/[a-z0-9]{3,}/g) ?? []) {
        let h = 0;
        for (const ch of word) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
        v[h % this.dims] = (v[h % this.dims] ?? 0) + 1;
      }
      return v;
    });
  }
}
