import type { Proposal } from "./types";

export type Assessment = {
  /** Switch the next calls to the slow path (stronger model, more thinking). */
  escalate: boolean;
  /** Things look bad enough that a human should weigh in. */
  askHuman: boolean;
  reasons: string[];
  /** True when the reasons differ from last tick (so we alert once, not every tick). */
  changed: boolean;
};

/**
 * Watches for the agent going wrong: repeating itself, a run of surprises, disagreement,
 * or no progress.
 *
 * Brain analogue: anterior cingulate cortex (conflict and error monitoring).
 */
export class Monitor {
  private actions: string[] = [];
  private outcomes: { ok: boolean; surprise: number }[] = [];
  private ticksWithoutProgress = 0;
  private lastReasons = "";

  recordAction(p: Proposal) {
    this.actions.push(`${p.tool}(${JSON.stringify(p.args)})`);
  }

  recordOutcome(ok: boolean, surprise: number) {
    this.outcomes.push({ ok, surprise });
  }

  /** Recent average surprise, used by the modulators. */
  meanSurprise(window = 5): number {
    const recent = this.outcomes.slice(-window);
    return recent.length ? recent.reduce((s, o) => s + o.surprise, 0) / recent.length : 0;
  }

  endTick(madeProgress: boolean, contradictions: number): Assessment {
    this.ticksWithoutProgress = madeProgress ? 0 : this.ticksWithoutProgress + 1;
    const reasons: string[] = [];

    const counts = new Map<string, number>();
    for (const a of this.actions.slice(-6)) counts.set(a, (counts.get(a) ?? 0) + 1);
    for (const [action, n] of counts) if (n >= 3) reasons.push(`repeating ${action} (${n} of the last 6 actions)`);

    const last3 = this.outcomes.slice(-3);
    if (last3.length === 3 && last3.every((o) => !o.ok || o.surprise > 0.6)) reasons.push("3 failed or surprising outcomes in a row");

    if (contradictions > 0) reasons.push(`loops disagree (${contradictions} contradicting thoughts)`);
    if (this.ticksWithoutProgress >= 5) reasons.push(`no progress for ${this.ticksWithoutProgress} ticks`);

    const key = reasons.join("|");
    const changed = key !== this.lastReasons;
    this.lastReasons = key;
    return { escalate: reasons.length > 0, askHuman: reasons.length >= 2, reasons, changed };
  }
}
