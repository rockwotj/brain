/**
 * Monitor: compares predictions with outcomes, tracks progress, and decides
 * when the expensive executive is worth waking up.
 */
import type { ActionRecord, Prediction } from "./types.ts";
import type { ToolResult } from "./tools.ts";

export type EscalationReason = "novelty" | "conflict" | "surprise" | "stall";

/** Surprise = the outcome contradicts the prediction (success flag or expected content). */
export function isSurprise(predicted: Prediction, actual: ToolResult): boolean {
  if (predicted.success !== actual.ok) return true;
  if (actual.ok && predicted.contains) return !actual.output.toLowerCase().includes(predicted.contains.toLowerCase());
  return false;
}

export class Monitor {
  private pending = new Set<EscalationReason>();
  private idleSteps = 0;
  private consultedSinceProgress = false;

  constructor(private readonly stallSteps: number) {}

  observeOutcome(record: ActionRecord): void {
    if (record.surprise) this.pending.add("surprise");
  }

  flag(reason: EscalationReason): void {
    this.pending.add(reason);
  }

  hasPending(): boolean {
    return this.pending.size > 0;
  }

  /** Reasons to wake the executive this step; clears them. */
  takeReasons(): EscalationReason[] {
    const reasons = [...this.pending];
    this.pending.clear();
    return reasons;
  }

  executiveConsulted(): void {
    this.consultedSinceProgress = true;
  }

  /**
   * Call once per step. Returns "stop" when there has been no progress for
   * `stallSteps` steps even after the executive was consulted.
   */
  endStep(progressed: boolean): "continue" | "stop" {
    if (progressed) {
      this.idleSteps = 0;
      this.consultedSinceProgress = false;
      return "continue";
    }
    this.idleSteps++;
    if (this.idleSteps < this.stallSteps) return "continue";
    if (this.consultedSinceProgress) return "stop";
    this.pending.add("stall");
    return "continue";
  }
}
