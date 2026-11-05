/**
 * The shared vocabulary of the brain. Start reading here.
 */
import type { Part } from "../io/content";
import type { Thinking, Tier } from "../io/llm";
import type { Risk, ToolArgs } from "../io/tools";

export type BidKind =
  | "percept" // something sensed (user message, image, audio/video observation)
  | "result" // a surprising tool result
  | "error" // a failed action
  | "thought" // a loop's hypothesis, plan or objection
  | "goal" // a goal flowing down from a higher loop
  | "intention" // a prospective memory that just fired ("when X, do Y")
  | "alert"; // the monitor noticing trouble

/** Something competing for (or holding) a slot in the global workspace. */
export type Bid = {
  id: string;
  kind: BidKind;
  content: string;
  /** Images etc. for multimodal loops. */
  attachments?: Part[];
  /** Who produced it: "user", "gateway", "monitor", a loop id... */
  source: string;
  /** 0..1: how loudly it's shouting for attention. */
  salience: number;
  /** Where the claim came from (for source monitoring / anti-confabulation). */
  provenance?: string[];
  /** Ids of bids this one disagrees with (lateral inhibition). */
  contradicts?: string[];
  /** Ticks before it fades from the workspace. */
  ttl: number;
};

export type NewBid = Omit<Bid, "id" | "ttl"> & { ttl?: number };

/** What a loop expects an action to produce (the "efference copy"). */
export type Prediction = { expect: string; confidence: number };

/** An action a loop wants to take. */
export type Proposal = Prediction & {
  id: string;
  source: string;
  tool: string;
  args: ToolArgs;
  why?: string;
  risk?: Risk;
};

export type NewProposal = Omit<Proposal, "id" | "source">;

/** A few global scalars that change the mode of the whole system (neuromodulators). */
export type Modulators = {
  /** Noradrenaline-ish arousal: someone is waiting. Act sooner, run loops in parallel. */
  urgency: number;
  /** Exploration drive: rises when stuck. Higher temperature, stronger models. */
  explore: number;
  /** Acetylcholine-ish: trust in new evidence vs. predictions. Scales how loud surprises are. */
  precision: number;
  /** Dopamine-ish: running average of reward. */
  progress: number;
  /** Serotonin-ish: willingness to spend for delayed payoff (≈ budget headroom). */
  patience: number;
};

/** How a single model call should run. */
export type Runtime = { tier: Tier; thinking: Thinking; temperature: number };

/** Timescale of a loop: reflex (seconds) → task (minutes) → project (hours) → self (days). */
export type Level = "reflex" | "task" | "project" | "self";

/** What the brain hands a loop when waking it up. */
export type LoopEnv = {
  tick: number;
  /** Pick model tier / thinking / temperature given the loop's default tier and the brain's state. */
  runtimeFor(defaultTier: Tier): Runtime;
  /** Cue-based recall from long-term memory. Results are reconstructions: treat as hypotheses. */
  recall(cue: string): Promise<string[]>;
};

export type LoopOutput = { bids: NewBid[]; proposals: NewProposal[] };

export const nothing = (): LoopOutput => ({ bids: [], proposals: [] });

/** An agentic loop: a specialist, the reflex system, a project-level overseer... */
export interface Loop {
  readonly id: string;
  readonly level: Level;
  /** Sparse activation: which broadcasts wake this loop up. */
  subscribes(bid: Bid): boolean;
  /** React to broadcasts plus private feedback about its own actions. */
  onBroadcast(items: Bid[], env: LoopEnv): Promise<LoopOutput>;
  /** A shared fact changed (goal, user model...). */
  setFact?(key: string, value: string | undefined): void;
  /** 0..1, how full this loop's context is (interoception). */
  contextPressure?(): number;
}
