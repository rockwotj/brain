import { cosine, type Embedder } from "../io/embeddings";
import type { Bid } from "./types";

export type Goal = { id: string; text: string; embedding: number[] };

/**
 * Holds the goal stack and biases what wins the workspace toward the current goal.
 *
 * Brain analogue: prefrontal cortex (top-down attention, task rules).
 * Goal changes are published as shared facts, which loops receive as context deltas.
 */
export class Executive {
  private goals: Goal[] = [];
  private nextId = 1;

  constructor(
    private readonly embedder: Embedder,
    private readonly publish: (key: string, value: string | undefined) => void,
  ) {}

  current(): Goal | undefined {
    return this.goals.at(-1);
  }

  async push(text: string): Promise<Goal> {
    const [embedding] = await this.embedder.embed([text]);
    const goal = { id: `g${this.nextId++}`, text, embedding: embedding! };
    this.goals.push(goal);
    this.publish(`goal ${goal.id}`, text);
    return goal;
  }

  complete(id: string): boolean {
    const before = this.goals.length;
    this.goals = this.goals.filter((g) => g.id !== id);
    if (this.goals.length === before) return false;
    this.publish(`goal ${id}`, undefined);
    return true;
  }

  /** Top-down bias per bid id: favor the user, trouble, and anything related to the current goal. */
  async biasFor(bids: Bid[]): Promise<Map<string, number>> {
    const bias = new Map<string, number>();
    const goal = this.current();
    const embeddings = goal && bids.length ? await this.embedder.embed(bids.map((b) => b.content)) : [];

    bids.forEach((bid, i) => {
      let b = 1;
      if (bid.source === "user") b *= 1.5;
      if (bid.kind === "error" || bid.kind === "alert") b *= 1.3;
      if (goal) b *= 0.6 + 0.8 * Math.max(0, cosine(goal.embedding, embeddings[i]!));
      bias.set(bid.id, b);
    });
    return bias;
  }
}
