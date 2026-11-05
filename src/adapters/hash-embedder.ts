import type { Embedder } from "../io/embeddings";

/**
 * Deterministic bag-of-words embeddings for tests and offline demos. Not semantic!
 * Numbers are collapsed to "#" so "what is 2 * 3" ≈ "what is 17 * 23".
 */
export class HashEmbedder implements Embedder {
  constructor(private readonly dims = 256) {}

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.vector(t));
  }

  private vector(text: string): number[] {
    const v = new Array<number>(this.dims).fill(0);
    const words = text.toLowerCase().replace(/\d+/g, "#").match(/[a-z#]+/g) ?? [];
    for (const word of words) v[hash(word) % this.dims]! += 1;
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map((x) => x / norm);
  }
}

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
