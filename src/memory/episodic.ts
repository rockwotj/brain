import type { Clock } from "../io/clock";
import { cosine, embedOne, type Embedder } from "../io/embeddings";
import type { Storage } from "../io/storage";

export type Episode = {
  at: number;
  /** What happened, in one line. */
  what: string;
  /** The goal that was active at the time. */
  goal?: string;
  /** Who or what caused it. */
  source: string;
  surprise: number;
  /** -1..1, how it turned out. */
  valence: number;
  embedding: number[];
};

/**
 * Fast, append-only record of what happened. Never edited.
 *
 * Brain analogue: hippocampus (one-shot episodic encoding, indexed by context).
 */
export class EpisodicMemory {
  private seq = 0;

  constructor(
    private readonly storage: Storage,
    private readonly embedder: Embedder,
    private readonly clock: Clock,
  ) {}

  async record(episode: Omit<Episode, "at" | "embedding">) {
    const at = this.clock();
    const embedding = await embedOne(this.embedder, episode.what);
    // Zero-padded time + sequence → keys sort chronologically.
    const key = `episodes/${String(at).padStart(15, "0")}-${String(this.seq++).padStart(6, "0")}`;
    await this.storage.put<Episode>(key, { ...episode, at, embedding });
  }

  async since(time: number): Promise<Episode[]> {
    const all = await this.storage.list<Episode>("episodes/");
    return all.map((e) => e.value).filter((e) => e.at > time);
  }

  /** Pattern completion: the episodes most similar to a partial cue. */
  async recall(cue: string, k = 3, minScore = 0.4): Promise<Episode[]> {
    const query = await embedOne(this.embedder, cue);
    const all = await this.storage.list<Episode>("episodes/");
    return all
      .map(({ value }) => ({ episode: value, score: cosine(query, value.embedding) }))
      .filter((r) => r.score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((r) => r.episode);
  }
}
