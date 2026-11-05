/**
 * Two-speed memory on the async Storage port:
 *  - EpisodicStore: fast, append-only record of whole turns (written live).
 *  - KnowledgeStore: slow, consolidated statements with provenance (written only by consolidation).
 *  - HabitStore: situation -> action shortcuts (registered pending, approved by a human).
 *
 * Recall uses an injected Embedder: items are embedded when written and ranked
 * by cosine similarity to the cue (brute force; fine for a PoC). Without an
 * embedder there is no recall: no keyword fallback pretends to be semantic.
 */
import { cosine, embedOne, type Embedder } from "./io/embeddings.ts";
import type { Storage } from "./io/storage.ts";
import type { Episode, Habit, KnowledgeItem, Prediction, ToolCall } from "./types.ts";

type Stored<T> = { item: T; embedding?: number[] };
export type RecallOptions = { limit: number; minSimilarity: number };

/** Items with ids like `ep-3`, stored under keys like `episodes/000003`. */
class Collection<T extends { id: string }> {
  constructor(
    private readonly storage: Storage,
    private readonly prefix: string,
    private readonly idPrefix: string,
  ) {}

  async nextId(): Promise<string> {
    return `${this.idPrefix}-${(await this.storage.keys(this.prefix)).length + 1}`;
  }
  async get(id: string): Promise<Stored<T> | undefined> {
    return this.storage.get<Stored<T>>(this.key(id));
  }
  async put(entry: Stored<T>): Promise<void> {
    await this.storage.put(this.key(entry.item.id), entry);
  }
  async all(): Promise<Stored<T>[]> {
    const keys = await this.storage.keys(this.prefix);
    const entries = await Promise.all(keys.map((k) => this.storage.get<Stored<T>>(k)));
    return entries.filter((e): e is Stored<T> => !!e);
  }
  private key(id: string): string {
    const n = Number(id.slice(this.idPrefix.length + 1));
    if (!id.startsWith(`${this.idPrefix}-`) || !Number.isInteger(n)) throw new Error(`not a ${this.idPrefix} id: ${id}`);
    return `${this.prefix}${String(n).padStart(6, "0")}`;
  }
}

/** Top matches by cosine similarity. Items without a comparable embedding are skipped. */
function rank<T>(entries: readonly Stored<T>[], cue: readonly number[], opts: RecallOptions): T[] {
  return entries
    .flatMap((e) => (e.embedding && e.embedding.length === cue.length ? [{ item: e.item, score: cosine(cue, e.embedding) }] : []))
    .filter((x) => x.score >= opts.minSimilarity)
    .sort((a, b) => b.score - a.score)
    .slice(0, opts.limit)
    .map((x) => x.item);
}

export class EpisodicStore {
  private readonly items: Collection<Episode>;
  constructor(storage: Storage, private readonly embedder?: Embedder) {
    this.items = new Collection(storage, "episodes/", "ep");
  }
  async append(draft: Omit<Episode, "id" | "consolidated">): Promise<Episode> {
    const episode: Episode = { ...draft, id: await this.items.nextId(), consolidated: false };
    const embedding = this.embedder ? await embedOne(this.embedder, `${episode.goal}\n${episode.answer ?? ""}`) : undefined;
    await this.items.put({ item: episode, ...(embedding ? { embedding } : {}) });
    return episode;
  }
  async get(id: string): Promise<Episode | undefined> {
    return (await this.items.get(id))?.item;
  }
  async list(): Promise<Episode[]> {
    return (await this.items.all()).map((e) => e.item);
  }
  async recall(cue: readonly number[], opts: RecallOptions): Promise<Episode[]> {
    return rank(await this.items.all(), cue, opts);
  }
  async markConsolidated(ids: readonly string[]): Promise<void> {
    for (const id of ids) {
      const entry = await this.items.get(id);
      if (entry) await this.items.put({ ...entry, item: { ...entry.item, consolidated: true } });
    }
  }
}

export class KnowledgeStore {
  private readonly items: Collection<KnowledgeItem>;
  constructor(storage: Storage, private readonly embedder?: Embedder) {
    this.items = new Collection(storage, "knowledge/", "kn");
  }
  /** Adds a statement, or merges provenance into an identical existing one. */
  async add(statement: string, sources: readonly string[]): Promise<KnowledgeItem> {
    if (!sources.length) throw new Error("knowledge requires provenance");
    const existing = (await this.items.all()).find((e) => e.item.statement.toLowerCase() === statement.toLowerCase());
    if (existing) {
      const item = { ...existing.item, sources: [...new Set([...existing.item.sources, ...sources])], version: existing.item.version + 1 };
      await this.items.put({ ...existing, item });
      return item;
    }
    const item: KnowledgeItem = { id: await this.items.nextId(), statement, sources: [...sources], version: 1 };
    const embedding = this.embedder ? await embedOne(this.embedder, statement) : undefined;
    await this.items.put({ item, ...(embedding ? { embedding } : {}) });
    return item;
  }
  async list(): Promise<KnowledgeItem[]> {
    return (await this.items.all()).map((e) => e.item);
  }
  async recall(cue: readonly number[], opts: RecallOptions): Promise<KnowledgeItem[]> {
    return rank(await this.items.all(), cue, opts);
  }
}

export class HabitStore {
  private readonly items: Collection<Habit>;
  constructor(storage: Storage) {
    this.items = new Collection(storage, "habits/", "hb");
  }
  async register(draft: { trigger: string; call: ToolCall; expect: Prediction; sources: string[] }, status: Habit["status"] = "pending"): Promise<Habit> {
    const habit: Habit = { ...draft, id: await this.items.nextId(), status, successes: 0, failures: 0 };
    await this.items.put({ item: habit });
    return habit;
  }
  async approve(id: string): Promise<void> {
    await this.update(id, (h) => ({ ...h, status: "approved" }));
  }
  /** Approved habits whose trigger phrase appears in the goal (case-insensitive). */
  async match(goal: string): Promise<Habit[]> {
    const g = goal.toLowerCase();
    return (await this.list()).filter((h) => h.status === "approved" && g.includes(h.trigger.toLowerCase()));
  }
  /** A surprising outcome suspends the habit until a human re-approves it. */
  async recordOutcome(id: string, ok: boolean): Promise<void> {
    await this.update(id, (h) => (ok ? { ...h, successes: h.successes + 1 } : { ...h, failures: h.failures + 1, status: "suspended" }));
  }
  async list(): Promise<Habit[]> {
    return (await this.items.all()).map((e) => e.item);
  }
  private async update(id: string, change: (h: Habit) => Habit): Promise<void> {
    const entry = await this.items.get(id);
    if (!entry) throw new Error(`unknown habit ${id}`);
    await this.items.put({ item: change(entry.item) });
  }
}
