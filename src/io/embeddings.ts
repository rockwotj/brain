export interface Embedder {
  embed(texts: string[]): Promise<number[][]>;
}

export async function embedOne(embedder: Embedder, text: string): Promise<number[]> {
  const [vector] = await embedder.embed([text]);
  return vector!;
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}
