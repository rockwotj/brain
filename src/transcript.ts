/**
 * Per-agent, append-only transcript designed for prompt-prefix stability.
 *
 * Epoch layout:  [system (immutable)] [user: snapshot + request] [assistant] [user: deltas + request] ...
 *
 * Earlier messages are never rewritten. Each call appends one user message that
 * carries only the deltas since this agent last looked. When the next request
 * would not fit in `contextWindow - reserveOutput`, the transcript starts a new
 * epoch: the deltas are folded into a fresh snapshot of the *current*
 * authoritative state (so nothing stale wins), plus a short tail of this agent's
 * own recent replies for continuity.
 */
import { estimateMessages, type Message, type TokenEstimator } from "./llm.ts";
import { clip, describeEvent, type StateStore } from "./state.ts";

export type TranscriptOptions = {
  contextWindow: number;
  /** Tokens kept free for the model's output and safety headroom. */
  reserveOutput: number;
  estimate: TokenEstimator;
  /** How many of the agent's own replies to carry into a new epoch. */
  tailReplies?: number;
};

export type PreparedTurn = { messages: readonly Message[]; stablePrefixLength: number; compacted: boolean };

export class ContextBudgetError extends Error {}

export class AgentTranscript {
  private messages: Message[] = [];
  private seenSeq = 0;
  epoch = 0;

  constructor(
    readonly agent: string,
    private readonly systemPrompt: string,
    private readonly options: TranscriptOptions,
  ) {}

  /** Build the next request. Appends exactly one user message; earlier messages are untouched. */
  prepare(state: StateStore, request: string): PreparedTurn {
    if (this.messages.length > 0 && this.messages.at(-1)?.role === "user") {
      throw new Error(`${this.agent}: previous turn has no recorded reply`);
    }
    let compacted = false;
    let user = this.messages.length === 0 ? this.openingMessage(state, request, []) : this.deltaMessage(state, request);
    if (this.messages.length === 0) this.messages = [{ role: "system", content: this.systemPrompt }];
    else if (!this.fits([...this.messages, user])) {
      this.epoch++;
      user = this.openingMessage(state, request, this.ownReplies());
      this.messages = [{ role: "system", content: this.systemPrompt }];
      compacted = true;
    }
    const stablePrefixLength = this.messages.length;
    this.messages.push(user);
    if (!this.fits(this.messages)) {
      throw new ContextBudgetError(`${this.agent}: snapshot alone exceeds the context budget`);
    }
    this.seenSeq = state.seq;
    return { messages: [...this.messages], stablePrefixLength, compacted };
  }

  recordReply(text: string): void {
    this.messages.push({ role: "assistant", content: text });
  }

  get length(): number {
    return this.messages.length;
  }

  private fits(messages: Message[]): boolean {
    const used = estimateMessages(messages, this.options.estimate);
    return used + this.options.reserveOutput <= this.options.contextWindow;
  }

  private openingMessage(state: StateStore, request: string, tail: string[]): Message {
    const parts = [`EPOCH ${this.epoch}`, state.renderSnapshot()];
    if (tail.length) {
      parts.push(["YOUR RECENT REPLIES (context only; the snapshot above is authoritative):", ...tail.map((t) => `- ${t}`)].join("\n"));
    }
    parts.push(`REQUEST: ${request}`);
    return { role: "user", content: parts.join("\n\n") };
  }

  private deltaMessage(state: StateStore, request: string): Message {
    const events = state.eventsSince(this.seenSeq);
    const updates = events.length
      ? [`UPDATES since seq ${this.seenSeq} (newer than anything above):`, ...events.map(describeEvent)].join("\n")
      : "UPDATES: none";
    return { role: "user", content: `${updates}\n\nREQUEST: ${request}` };
  }

  private ownReplies(): string[] {
    const keep = this.options.tailReplies ?? 2;
    if (keep <= 0) return [];
    return this.messages
      .filter((m) => m.role === "assistant")
      .slice(-keep)
      .map((m) => clip(m.content, 300));
  }
}
