/**
 * Per-agent, append-only transcript designed for prompt-prefix stability.
 *
 * Epoch layout:  [system (immutable)] [user: snapshot + request] [assistant] [user: deltas + request] ...
 *
 * Earlier messages are never rewritten. Each call appends one user message that
 * carries only the events this agent may see since it last looked, plus any
 * private notes (e.g. a record it asked to read). A new epoch starts when:
 *  - the next request would not fit in `contextWindow - reserveOutput` ("overflow"), or
 *  - the provider reported that this prompt's cache has expired and the
 *    transcript is big ("cache_expired"): re-sending a long uncached prompt
 *    costs more than a fresh snapshot. This is an optimization, never free.
 * A new epoch folds everything into a fresh snapshot of the *current* state
 * (including the evidence index), plus a short tail of this agent's own replies.
 * No summarizing model call is made.
 */
import { text, type MediaPart } from "./io/content.ts";
import { estimateMessages, type Message, type TokenEstimator } from "./io/llm.ts";
import { clip, describeEvent, visibleTo, type StateStore } from "./state.ts";

export type TranscriptOptions = {
  contextWindow: number;
  /** Tokens kept free for the model's output and safety headroom. */
  reserveOutput: number;
  estimate: TokenEstimator;
  /** How many of the agent's own replies to carry into a new epoch. */
  tailReplies?: number;
  /** Send image attachments as image parts (the model has vision). */
  vision?: boolean;
  clock?: () => number;
  /** After a cache-expiry hint, compact only if the transcript is at least this many tokens. Default: a quarter of the budget. */
  compactAfterExpiryTokens?: number;
};

export type CompactionReason = "overflow" | "cache_expired";
export type PreparedTurn = { messages: readonly Message[]; stablePrefixLength: number; compacted: false | CompactionReason };

export class ContextBudgetError extends Error {}

export class AgentTranscript {
  private messages: Message[] = [];
  private seenSeq = 0;
  private cacheExpiresAt?: number;
  epoch = 0;

  constructor(
    readonly agent: string,
    readonly systemPrompt: string,
    private readonly options: TranscriptOptions,
  ) {}

  /**
   * Build the next request. Appends exactly one user message; earlier messages are untouched.
   * Atomic: if even a fresh snapshot does not fit, it throws and changes nothing.
   */
  prepare(state: StateStore, request: string, notes: readonly string[] = []): PreparedTurn {
    if (this.messages.at(-1)?.role === "user") throw new Error(`${this.agent}: previous turn has no recorded reply`);
    const system: Message = { role: "system", content: this.systemPrompt };
    let compacted: PreparedTurn["compacted"] = false;
    let epoch = this.epoch;
    let next: Message[];
    if (this.messages.length === 0) {
      next = [system, this.openingMessage(state, request, [], notes, epoch)];
    } else {
      next = [...this.messages, this.deltaMessage(state, request, notes)];
      const reason = !this.fits(next) ? "overflow" : this.cacheExpired() ? "cache_expired" : undefined;
      if (reason) {
        epoch++;
        next = [system, this.openingMessage(state, request, this.ownReplies(), notes, epoch)];
        compacted = reason;
      }
    }
    if (!this.fits(next)) throw new ContextBudgetError(`${this.agent}: snapshot alone exceeds the context budget`);
    this.messages = next;
    this.epoch = epoch;
    if (compacted) this.cacheExpiresAt = undefined;
    this.seenSeq = state.seq;
    return { messages: [...next], stablePrefixLength: next.length - 1, compacted };
  }

  recordReply(reply: string): void {
    this.messages.push({ role: "assistant", content: reply });
  }

  /** Provider hint from the last response (undefined = no hint). */
  noteCacheExpiry(at: number | undefined): void {
    this.cacheExpiresAt = at;
  }

  /** Explicit task boundary: the next request opens a fresh epoch from the current snapshot. */
  reset(): void {
    if (!this.messages.length) return;
    this.messages = [];
    this.epoch++;
  }

  get length(): number {
    return this.messages.length;
  }

  private fits(messages: Message[]): boolean {
    return estimateMessages(messages, this.options.estimate) + this.options.reserveOutput <= this.options.contextWindow;
  }

  private cacheExpired(): boolean {
    if (this.cacheExpiresAt === undefined || (this.options.clock ?? Date.now)() < this.cacheExpiresAt) return false;
    const threshold = this.options.compactAfterExpiryTokens ?? (this.options.contextWindow - this.options.reserveOutput) / 4;
    return estimateMessages(this.messages, this.options.estimate) >= threshold;
  }

  private openingMessage(state: StateStore, request: string, tail: string[], notes: readonly string[], epoch: number): Message {
    const parts = [`EPOCH ${epoch}`, state.renderSnapshot()];
    if (tail.length) {
      parts.push(["YOUR RECENT REPLIES (context only; the snapshot above is authoritative):", ...tail.map((t) => `- ${t}`)].join("\n"));
    }
    return this.userMessage([...parts, ...renderNotes(notes), `REQUEST: ${request}`], state.attachments);
  }

  private deltaMessage(state: StateStore, request: string, notes: readonly string[]): Message {
    const events = state.eventsSince(this.seenSeq).filter((e) => visibleTo(e, this.agent));
    const updates = events.length
      ? [`UPDATES since seq ${this.seenSeq} (newer than anything above):`, ...events.map((e) => describeEvent(e, state.previewChars))].join("\n")
      : "UPDATES: none";
    const media = events.flatMap((e) => (e.delta.type === "goal.set" ? (e.delta.attachments ?? []) : []));
    return this.userMessage([updates, ...renderNotes(notes), `REQUEST: ${request}`], media);
  }

  private userMessage(sections: string[], media: readonly MediaPart[]): Message {
    const body = sections.join("\n\n");
    if (!media.length) return { role: "user", content: body };
    if (!this.options.vision) return { role: "user", content: `${body}\n\n(Attached media is not visible to your model; rely on observations from others.)` };
    return { role: "user", content: [text(body), ...media] };
  }

  private ownReplies(): string[] {
    const keep = this.options.tailReplies ?? 2;
    if (keep <= 0) return [];
    return this.messages
      .filter((m) => m.role === "assistant")
      .slice(-keep)
      .map((m) => clip(typeof m.content === "string" ? m.content : "", 300));
  }
}

function renderNotes(notes: readonly string[]): string[] {
  return notes.length ? [["PRIVATE TO YOU (not broadcast):", ...notes].join("\n\n")] : [];
}
