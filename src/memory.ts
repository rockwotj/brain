/**
 * Two-speed memory behind ports, with in-memory implementations:
 *  - EpisodicStore: fast, append-only record of whole runs (written live).
 *  - KnowledgeStore: slow, consolidated statements with provenance (written only by consolidation).
 *  - HabitStore: approved situation -> action shortcuts (registered pending, approved by a human).
 */
import type { Episode, Habit, KnowledgeItem, Prediction, ToolCall } from "./types.ts";

export interface EpisodicStore {
  append(episode: Omit<Episode, "id" | "consolidated">): Episode;
  get(id: string): Episode | undefined;
  list(): Episode[];
  recall(cue: string, limit: number): Episode[];
  markConsolidated(ids: readonly string[]): void;
}

export interface KnowledgeStore {
  add(statement: string, sources: readonly string[]): KnowledgeItem;
  list(): KnowledgeItem[];
  recall(cue: string, limit: number): KnowledgeItem[];
}

export interface HabitStore {
  register(draft: { trigger: string; call: ToolCall; expect: Prediction; sources: string[] }): Habit;
  approve(id: string): void;
  /** Approved habits whose trigger appears in the goal. */
  match(goal: string): Habit[];
  recordOutcome(id: string, ok: boolean): void;
  list(): Habit[];
}

export class InMemoryEpisodicStore implements EpisodicStore {
  private readonly episodes: Episode[] = [];
  append(draft: Omit<Episode, "id" | "consolidated">): Episode {
    const episode = { ...draft, id: `ep-${this.episodes.length + 1}`, consolidated: false };
    this.episodes.push(episode);
    return episode;
  }
  get(id: string): Episode | undefined {
    return this.episodes.find((e) => e.id === id);
  }
  list(): Episode[] {
    return [...this.episodes];
  }
  recall(cue: string, limit: number): Episode[] {
    return rankByOverlap(this.episodes, cue, (e) => `${e.goal} ${e.answer ?? ""}`, limit);
  }
  markConsolidated(ids: readonly string[]): void {
    for (const e of this.episodes) if (ids.includes(e.id)) e.consolidated = true;
  }
}

export class InMemoryKnowledgeStore implements KnowledgeStore {
  private readonly items: KnowledgeItem[] = [];
  add(statement: string, sources: readonly string[]): KnowledgeItem {
    if (!sources.length) throw new Error("knowledge requires provenance");
    const existing = this.items.find((k) => k.statement.toLowerCase() === statement.toLowerCase());
    if (existing) {
      existing.sources = [...new Set([...existing.sources, ...sources])];
      existing.version++;
      return existing;
    }
    const item = { id: `kn-${this.items.length + 1}`, statement, sources: [...sources], version: 1 };
    this.items.push(item);
    return item;
  }
  list(): KnowledgeItem[] {
    return [...this.items];
  }
  recall(cue: string, limit: number): KnowledgeItem[] {
    return rankByOverlap(this.items, cue, (k) => k.statement, limit);
  }
}

export class InMemoryHabitStore implements HabitStore {
  private readonly habits: Habit[] = [];
  register(draft: { trigger: string; call: ToolCall; expect: Prediction; sources: string[] }): Habit {
    const habit: Habit = { ...draft, id: `hb-${this.habits.length + 1}`, status: "pending", successes: 0, failures: 0 };
    this.habits.push(habit);
    return habit;
  }
  approve(id: string): void {
    const habit = this.habits.find((h) => h.id === id);
    if (!habit) throw new Error(`unknown habit ${id}`);
    habit.status = "approved";
  }
  match(goal: string): Habit[] {
    const g = goal.toLowerCase();
    return this.habits.filter((h) => h.status === "approved" && g.includes(h.trigger.toLowerCase()));
  }
  /** A surprising outcome suspends the habit until a human re-approves it. */
  recordOutcome(id: string, ok: boolean): void {
    const habit = this.habits.find((h) => h.id === id);
    if (!habit) return;
    if (ok) habit.successes++;
    else {
      habit.failures++;
      habit.status = "suspended";
    }
  }
  list(): Habit[] {
    return [...this.habits];
  }
}

/** Naive keyword-overlap retrieval; swap for embeddings behind the same port. */
function rankByOverlap<T>(items: readonly T[], cue: string, text: (item: T) => string, limit: number): T[] {
  const words = (s: string) => new Set(s.toLowerCase().match(/[a-z0-9]{4,}/g) ?? []);
  const cueWords = words(cue);
  return items
    .map((item) => ({ item, score: [...words(text(item))].filter((w) => cueWords.has(w)).length }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((x) => x.item);
}
