import type { Clock } from "../io/clock";
import { text, type Part } from "../io/content";
import type { GenerateRequest, Message } from "../io/llm";
import { estimateTokens } from "../io/tokens";

export type StableContextOptions = {
  /** Never changes: who this loop is, its tools, and its reply format. */
  instructions: string;
  /** Shared, slowly changing facts (goals, user model...). Rendered into the prefix only at compaction. */
  facts?: Record<string, string>;
  /** The model's context window (or the slice of it we allow this loop). */
  maxTokens: number;
  /** How long the provider keeps a cached prefix alive after the last call (often ~5 minutes). */
  cacheTtlMs: number;
  clock: Clock;
};

export type CompactionReason = "overflow" | "cache-expired";

/**
 * A model conversation whose prefix only ever grows, so provider prompt caches keep hitting.
 *
 *   system:   instructions + shared facts + summary of earlier work   ← frozen until compaction
 *   messages: user, assistant, user, assistant, ...                   ← append-only
 *
 * Rules:
 *  1. Never edit anything already sent.
 *  2. Changes to shared facts are NOT written into the system prompt. They become small
 *     "[context update]" deltas that ride along on the next user message.
 *  3. Compaction is the only prefix break, so we do it when we'd lose the cache anyway:
 *       - "overflow":      the context is about to run out of room, or
 *       - "cache-expired": we've been idle past the cache TTL, so the next call re-reads
 *                          everything at full price regardless; rewriting the prefix is free.
 *     Compaction folds all deltas into a fresh system prompt and replaces the log with a summary.
 */
export class StableContext {
  private facts: Map<string, string>;
  private summary = "";
  private system: string;
  private log: Message[] = [];
  private pendingDeltas: string[] = [];
  private deltasSinceCompaction = 0;
  private sentUpTo = 0;
  private lastCallAt?: number;

  /** How many times the prefix was rewritten (each one costs a cache miss). */
  prefixBreaks = 0;

  constructor(private readonly opts: StableContextOptions) {
    this.facts = new Map(Object.entries(opts.facts ?? {}));
    this.system = this.renderSystem();
  }

  /** Change a shared fact. Appends a delta instead of rewriting the prefix. */
  setFact(key: string, value: string | undefined) {
    const old = this.facts.get(key);
    if (old === value) return;
    if (value === undefined) {
      this.facts.delete(key);
      this.pendingDeltas.push(`- removed ${key}`);
    } else {
      this.facts.set(key, value);
      this.pendingDeltas.push(old === undefined ? `- ${key}: ${value}` : `- ${key} is now: ${value} (replaces the earlier value)`);
    }
    this.deltasSinceCompaction++;
  }

  addUser(content: Part[]) {
    const deltas = this.pendingDeltas.length ? [text(`[context update]\n${this.pendingDeltas.join("\n")}`)] : [];
    this.pendingDeltas = [];
    this.log.push({ role: "user", content: [...deltas, ...content] });
  }

  addAssistant(reply: string) {
    this.log.push({ role: "assistant", content: [text(reply)] });
  }

  /** What to send. Everything before `cacheUpTo` was sent last time, unchanged. */
  request(): Pick<GenerateRequest, "system" | "messages" | "cacheUpTo"> {
    return { system: this.system, messages: [...this.log], cacheUpTo: this.sentUpTo };
  }

  /** Call after the model replied (and the reply was appended). */
  markSent() {
    this.sentUpTo = this.log.length;
    this.lastCallAt = this.opts.clock();
  }

  tokens(): number {
    return estimateTokens(this.system) + this.log.reduce((sum, m) => sum + estimateTokens(m.content), 0);
  }

  /** 0..1, how full the context is. */
  pressure(): number {
    return this.tokens() / this.opts.maxTokens;
  }

  needsCompaction(): CompactionReason | undefined {
    if (this.pressure() > 0.8) return "overflow";
    const idleFor = this.lastCallAt === undefined ? 0 : this.opts.clock() - this.lastCallAt;
    const worthIt = this.deltasSinceCompaction > 0 || this.pressure() > 0.25;
    if (idleFor > this.opts.cacheTtlMs && worthIt) return "cache-expired";
    return undefined;
  }

  /** Rewrite the prefix: fold deltas into the facts section and summarize the log. */
  async compact(summarize: (messages: Message[], previousSummary: string) => Promise<string>) {
    if (this.log.length) this.summary = await summarize(this.log, this.summary);
    this.log = [];
    this.pendingDeltas = [];
    this.deltasSinceCompaction = 0;
    this.sentUpTo = 0;
    this.system = this.renderSystem();
    this.prefixBreaks++;
  }

  private renderSystem(): string {
    const sections = [this.opts.instructions];
    if (this.facts.size) {
      // Sorted so the same facts always render to the same bytes.
      const lines = [...this.facts.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => `- ${k}: ${v}`);
      sections.push(`# Shared context\n${lines.join("\n")}`);
    }
    if (this.summary) sections.push(`# Earlier work (summarized)\n${this.summary}`);
    return sections.join("\n\n");
  }
}
