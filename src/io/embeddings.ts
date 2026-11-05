/** Provider-neutral embeddings port. Used for memory recall (cosine similarity). */
export interface Embedder {
  embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]>;
}

export async function embedOne(embedder: Embedder, text: string, signal?: AbortSignal): Promise<number[]> {
  const [vector] = await embedder.embed([text], signal);
  if (!vector) throw new Error("embedder returned no vector");
  return vector;
}

export function cosine(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) throw new Error(`embedding size mismatch: ${a.length} vs ${b.length}`);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}
