import type { Clock } from "../io/clock";
import { parseJson } from "../io/llm";
import type { Models } from "../io/models";
import { describeTools, type Tool } from "../io/tools";
import { StableContext } from "../context/stable-context";
import type { Bid, Loop, LoopEnv, LoopOutput, NewProposal } from "../core/types";
import { nothing } from "../core/types";
import type { SkillLibrary, Skill } from "../memory/skills";
import { ACTION_FORMAT, renderBroadcast } from "./prompts";

type Run = { skill: Skill; percept: string; context: StableContext; started: boolean };

/**
 * The fast path. If an input matches a learned skill, run that skill step by step on the
 * System 1 model and skip the workspace entirely. The selector still gates every action,
 * and the first surprising result hands control back to the slow path.
 *
 * Brain analogue: habits (cue → routine), dorsal striatum.
 */
export class Reflex implements Loop {
  readonly id = "reflex";
  readonly level = "reflex" as const;
  private run?: Run;
  private readonly instructions: string;

  constructor(
    private readonly skills: SkillLibrary,
    private readonly models: Models,
    tools: Tool[],
    private readonly options: { clock: Clock; maxTokens: number; cacheTtlMs: number },
  ) {
    // Identical for every run, so the provider caches it across habits.
    this.instructions = [
      `You run a known habit (a learned skill) step by step for an agent. Follow the skill's steps, adapting details to the situation.`,
      `# Tools\n${describeTools(tools)}`,
      `# Reply format\nReply with ONLY a JSON object:\n{${ACTION_FORMAT}, "done": boolean}\nPropose only the next step(s). Set "done" to true with your final action, or when nothing is left to do.`,
    ].join("\n\n");
  }

  /** Broadcasts never wake the reflex; it's fed directly by the gateway. */
  subscribes(): boolean {
    return false;
  }

  /** Try to handle a percept with a habit. Returns true if a skill took it. */
  async claim(percept: string): Promise<boolean> {
    if (this.run) return false;
    const skill = await this.skills.match(percept);
    if (!skill) return false;
    // Fresh log per run; the system prompt is shared, so it stays cached across runs.
    const context = new StableContext({ instructions: this.instructions, ...this.options });
    this.run = { skill, percept, context, started: false };
    return true;
  }

  /** A step went wrong: give up on the habit and return the original input for the slow path. */
  async abandon(): Promise<string | undefined> {
    if (!this.run) return undefined;
    const { skill, percept } = this.run;
    this.run = undefined;
    await this.skills.reinforce(skill.id, false);
    return percept;
  }

  async onBroadcast(items: Bid[], env: LoopEnv): Promise<LoopOutput> {
    const run = this.run;
    if (!run) return nothing();

    const skillText = `Skill: ${run.skill.trigger}\nSteps:\n${run.skill.steps.map((s, i) => `${i + 1}. ${s}`).join("\n")}`;
    const update = renderBroadcast(env.tick, items, []);
    run.context.addUser(run.started ? update : [{ type: "text", text: skillText }, ...update]);
    run.started = true;

    const res = await this.models.system1.generate({ ...run.context.request(), json: true, thinking: "none", temperature: 0 });
    run.context.addAssistant(res.text);
    run.context.markSent();

    const reply = parseJson<{ actions?: NewProposal[]; done?: boolean }>(res.text);
    if (reply.done) {
      this.run = undefined;
      await this.skills.reinforce(run.skill.id, true);
    }
    return { bids: [], proposals: reply.actions ?? [] };
  }
}
