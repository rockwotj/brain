import type { Clock } from "../io/clock";
import { cosine, embedOne, type Embedder } from "../io/embeddings";
import type { Storage } from "../io/storage";

export type Fact = {
  id: string;
  text: string;
  embedding: number[];
  uses: number;
  version: number;
  /** Earlier versions, oldest first (rewrites never destroy the original). */
  history: string[];
  updatedAt: number;
};

/**
 * Slowly learned general knowledge: facts, schemas, the user model.
 * Written mostly by consolidation ("sleep"), not during the task.
 *
 * Brain analogue: neocortex. Recall is reconstructive and recalling a memory touches it
 * (reconsolidation); near-duplicate facts revise the existing fact instead of piling up.
 */
export class SemanticMemory {
  constructor(
    private readonly storage: Storage,
    private readonly embedder: Embedder,
    private readonly clock: Clock,
  ) {}

  async learn(text: string): Promise<Fact> {
    const embedding = await embedOne(this.embedder, text);
    const existing = await this.nearest(embedding, 0.9);

    if (existing) {
      const revised: Fact = {
        ...existing,
        text,
        embedding,
        version: existing.version + 1,
        history: [...existing.history, existing.text],
        updatedAt: this.clock(),
      };
      await this.storage.put(`facts/${existing.id}`, revised);
      return revised;
    }

    const fact: Fact = { id: crypto.randomUUID(), text, embedding, uses: 0, version: 1, history: [], updatedAt: this.clock() };
    await this.storage.put(`facts/${fact.id}`, fact);
    return fact;
  }

  async recall(cue: string, k = 3, minScore = 0.3): Promise<Fact[]> {
    const query = await embedOne(this.embedder, cue);
    const facts = (await this.storage.list<Fact>("facts/")).map((f) => f.value);
    const hits = facts
      .map((fact) => ({ fact, score: cosine(query, fact.embedding) }))
      .filter((r) => r.score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((r) => r.fact);

    // Reconsolidation (lite): using a memory strengthens it.
    for (const fact of hits) await this.storage.put(`facts/${fact.id}`, { ...fact, uses: fact.uses + 1 });
    return hits;
  }

  private async nearest(embedding: number[], minScore: number): Promise<Fact | undefined> {
    const facts = (await this.storage.list<Fact>("facts/")).map((f) => f.value);
    let best: { fact: Fact; score: number } | undefined;
    for (const fact of facts) {
      const score = cosine(embedding, fact.embedding);
      if (score >= minScore && (!best || score > best.score)) best = { fact, score };
    }
    return best?.fact;
  }
}
