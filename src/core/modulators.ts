import type { Tier, Usage } from "../io/llm";
import type { Modulators, Runtime } from "./types";

export const calm: Modulators = { urgency: 0.3, explore: 0.2, precision: 0.6, progress: 0.5, patience: 1 };

/**
 * Interoception: how the "body" is doing. Spend against budget and context fullness.
 * Costs are in abstract units; plug in real prices via `costOf`.
 */
export class Body {
  spent = 0;
  contextPressure = 0;

  constructor(
    readonly budget: number,
    private readonly costOf: (tier: Tier, usage: Usage) => number = defaultCost,
  ) {}

  record(tier: Tier, usage: Usage) {
    this.spent += this.costOf(tier, usage);
  }

  budgetLeft(): number {
    return Math.max(0, 1 - this.spent / this.budget);
  }
}

/** Cached input ~10% of fresh input; output ~4x; System 2 ~10x System 1. */
function defaultCost(tier: Tier, u: Usage): number {
  const tokens = u.inputTokens - u.cachedInputTokens + 0.1 * u.cachedInputTokens + 4 * u.outputTokens;
  return (tokens / 1000) * (tier === "system2" ? 10 : 1);
}

export type Signals = {
  userWaiting: boolean;
  meanSurprise: number;
  reward?: number;
  stuck: boolean;
  budgetLeft: number;
};

/** Move each modulator part of the way toward a target (they change smoothly, like hormones). */
export function updateModulators(m: Modulators, s: Signals): Modulators {
  const ease = (from: number, to: number) => from + (to - from) * 0.5;
  return {
    urgency: ease(m.urgency, s.userWaiting ? 0.9 : 0.2),
    explore: ease(m.explore, s.stuck ? 0.9 : 0.15),
    precision: ease(m.precision, 1 - s.meanSurprise),
    progress: s.reward === undefined ? m.progress : ease(m.progress, s.reward),
    patience: s.budgetLeft,
  };
}

/** Decide model tier, thinking budget and temperature for one call. */
export function chooseRuntime(defaultTier: Tier, m: Modulators, body: Body, escalated: boolean): Runtime {
  let tier: Tier = escalated || m.explore > 0.7 ? "system2" : defaultTier;
  // Allostasis: downshift before the budget runs out, not after.
  if (body.budgetLeft() < 0.1) tier = "system1";

  let thinking: Runtime["thinking"] = escalated ? "high" : m.precision < 0.4 ? "medium" : "low";
  if (m.urgency > 0.8 && !escalated) thinking = "low";
  if (tier === "system1") thinking = "none";

  return { tier, thinking, temperature: 0.2 + 0.6 * m.explore };
}

/** How many loops may think at once this tick. */
export function maxParallelLoops(m: Modulators, body: Body): number {
  if (body.budgetLeft() < 0.2) return 1;
  return m.urgency > 0.7 ? 4 : 2;
}
