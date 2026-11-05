import type { Storage } from "../io/storage";
import type { Tool } from "../io/tools";
import type { Modulators, Proposal } from "./types";

export type Veto = { proposal: Proposal; reason: string };
export type Selection = { chosen: Proposal[]; vetoed: Veto[] };

const signature = (p: Proposal) => `${p.tool}(${JSON.stringify(p.args)})`;

/**
 * Action gating. Default answer is "no". At most one action per effector is released per tick,
 * and only if it clears a threshold. Values per tool are learned from reward prediction errors.
 *
 * Brain analogue: basal ganglia (tonic inhibition, disinhibit the winner, dopamine learning).
 */
export class Selector {
  private values = new Map<string, number>();
  private lastFired = new Map<string, number>();

  constructor(
    private readonly storage: Storage,
    /** The same exact action can't fire again for this many ticks (anti-perseveration). */
    private readonly refractoryTicks = 2,
  ) {}

  async load() {
    const saved = await this.storage.get<Record<string, number>>("selector/values");
    this.values = new Map(Object.entries(saved ?? {}));
  }

  select(proposals: Proposal[], tools: Map<string, Tool>, mods: Modulators, tick: number): Selection {
    const chosen: Proposal[] = [];
    const vetoed: Veto[] = [];
    const byEffector = new Map<string, Proposal[]>();

    for (const p of proposals) {
      const tool = tools.get(p.tool);
      const last = this.lastFired.get(signature(p));
      if (!tool) vetoed.push({ proposal: p, reason: `unknown tool "${p.tool}"` });
      else if (p.risk === "high") vetoed.push({ proposal: p, reason: "high risk: needs approval" });
      else if (last !== undefined && tick - last <= this.refractoryTicks) vetoed.push({ proposal: p, reason: "refractory: just did this" });
      else byEffector.set(tool.effector, [...(byEffector.get(tool.effector) ?? []), p]);
    }

    // Urgency lowers the bar for acting.
    const threshold = 0.35 - 0.15 * mods.urgency;

    for (const group of byEffector.values()) {
      const [best, ...rest] = group.sort((a, b) => this.score(b) - this.score(a));
      if (this.score(best!) < threshold) {
        group.forEach((p) => vetoed.push({ proposal: p, reason: "below threshold" }));
        continue;
      }
      chosen.push(best!);
      this.lastFired.set(signature(best!), tick);
      rest.forEach((p) => vetoed.push({ proposal: p, reason: `lost to ${best!.tool} from ${best!.source}` }));
    }
    return { chosen, vetoed };
  }

  value(tool: string): number {
    return this.values.get(tool) ?? 0.5;
  }

  /** Update the tool's value toward the reward. Returns the reward prediction error. */
  async learn(tool: string, reward: number): Promise<number> {
    const error = reward - this.value(tool);
    this.values.set(tool, this.value(tool) + 0.3 * error);
    await this.storage.put("selector/values", Object.fromEntries(this.values));
    return error;
  }

  private score(p: Proposal): number {
    const confidence = Math.min(1, Math.max(0, p.confidence));
    return 0.5 * this.value(p.tool) + 0.5 * confidence - (p.risk === "medium" ? 0.15 : 0);
  }
}
