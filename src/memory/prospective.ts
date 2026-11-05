import { cosine, embedOne, type Embedder } from "../io/embeddings";
import type { Storage } from "../io/storage";

export type Intention = {
  id: string;
  /** A situation cue ("when the user mentions the deploy") — optional if time-based. */
  when?: string;
  /** Fire at this time (ms since epoch) — optional if cue-based. */
  dueAt?: number;
  then: string;
  embedding?: number[];
};

/**
 * Remembering to do something later: "when X happens, do Y" or "at time T, do Y".
 *
 * Brain analogue: prospective memory (prefrontal cortex + hippocampus).
 */
export class ProspectiveMemory {
  constructor(
    private readonly storage: Storage,
    private readonly embedder: Embedder,
  ) {}

  async add(intention: Omit<Intention, "id" | "embedding">): Promise<Intention> {
    const embedding = intention.when ? await embedOne(this.embedder, intention.when) : undefined;
    const full = { ...intention, id: crypto.randomUUID(), embedding };
    await this.storage.put(`intentions/${full.id}`, full);
    return full;
  }

  /** Intentions whose cue matches this event. Fired intentions are removed. */
  async triggeredBy(eventText: string, minScore = 0.75): Promise<Intention[]> {
    const all = await this.storage.list<Intention>("intentions/");
    if (!all.some((i) => i.value.embedding)) return [];
    const query = await embedOne(this.embedder, eventText);
    return this.fire(all.map((i) => i.value).filter((i) => i.embedding && cosine(query, i.embedding) >= minScore));
  }

  /** Time-based intentions that are due. Fired intentions are removed. */
  async due(now: number): Promise<Intention[]> {
    const all = await this.storage.list<Intention>("intentions/");
    return this.fire(all.map((i) => i.value).filter((i) => i.dueAt !== undefined && i.dueAt <= now));
  }

  private async fire(intentions: Intention[]): Promise<Intention[]> {
    for (const i of intentions) await this.storage.delete(`intentions/${i.id}`);
    return intentions;
  }
}
