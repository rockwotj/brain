import type { Clock } from "../io/clock";
import { parseJson, type Tier } from "../io/llm";
import type { Models } from "../io/models";
import { describeTools, type Tool } from "../io/tools";
import { StableContext } from "../context/stable-context";
import type { Bid, BidKind, Level, Loop, LoopEnv, LoopOutput, NewProposal } from "../core/types";
import { ACTION_FORMAT, renderBroadcast, summarize } from "./prompts";

export type SpecialistConfig = {
  id: string;
  level: Level;
  /** Default model tier; the brain may escalate it. */
  tier: Tier;
  /** What this loop is for (goes into its stable instructions). */
  role: string;
  /** Sparse activation: the bid kinds that wake it up. */
  listensTo: BidKind[];
  /** Can it propose actions, or only post thoughts? */
  canAct: boolean;
};

export type SpecialistOptions = { clock: Clock; maxTokens: number; cacheTtlMs: number };

type Reply = {
  thoughts?: { content: string; salience?: number; kind?: BidKind; contradicts?: string[] }[];
  actions?: NewProposal[];
};

/**
 * A narrow LLM loop with its own long-lived, cache-stable context. It only sees what reaches it
 * (broadcasts it subscribes to + feedback on its own actions) and answers with two things:
 * thoughts (bids for the workspace) and action proposals (with predictions).
 *
 * Brain analogue: a specialized cortical area.
 */
export class Specialist implements Loop {
  private readonly context: StableContext;

  constructor(
    private readonly config: SpecialistConfig,
    private readonly models: Models,
    tools: Tool[],
    options: SpecialistOptions,
  ) {
    this.context = new StableContext({ instructions: instructions(config, tools), ...options });
  }

  get id() {
    return this.config.id;
  }

  get level() {
    return this.config.level;
  }

  get prefixBreaks() {
    return this.context.prefixBreaks;
  }

  subscribes(bid: Bid): boolean {
    return bid.source !== this.id && this.config.listensTo.includes(bid.kind);
  }

  setFact(key: string, value: string | undefined) {
    this.context.setFact(key, value);
  }

  contextPressure(): number {
    return this.context.pressure();
  }

  async onBroadcast(items: Bid[], env: LoopEnv): Promise<LoopOutput> {
    if (this.context.needsCompaction()) {
      await this.context.compact((messages, previous) => summarize(this.models.system1, messages, previous));
    }

    const recalled = await env.recall(items.map((i) => i.content).join("\n"));
    this.context.addUser(renderBroadcast(env.tick, items, recalled));

    // Note: prompt caches are per model, so switching tier (escalation) is a cold call.
    const runtime = env.runtimeFor(this.config.tier);
    const res = await this.models[runtime.tier].generate({
      ...this.context.request(),
      json: true,
      thinking: runtime.thinking,
      temperature: runtime.temperature,
    });
    this.context.addAssistant(res.text);
    this.context.markSent();

    let reply: Reply;
    try {
      reply = parseJson<Reply>(res.text);
    } catch {
      return { bids: [{ kind: "alert", source: this.id, content: `${this.id} returned unparseable output`, salience: 0.5 }], proposals: [] };
    }

    return {
      bids: (reply.thoughts ?? []).map((t) => ({
        kind: t.kind ?? "thought",
        content: t.content,
        source: this.id,
        salience: Math.min(1, Math.max(0, t.salience ?? 0.5)),
        contradicts: t.contradicts,
        provenance: [this.id],
      })),
      proposals: this.config.canAct ? (reply.actions ?? []) : [],
    };
  }
}

function instructions(config: SpecialistConfig, tools: Tool[]): string {
  const format = `{"thoughts": [{"content": string, "salience": 0-1, "kind": "thought" | "goal" | "alert", "contradicts": [bid ids]}]${config.canAct ? `,\n ${ACTION_FORMAT}` : ""}}`;
  return [
    `You are the "${config.id}" loop inside a larger agent. ${config.role}`,
    "You only see what reaches you: workspace broadcasts you subscribe to, and feedback on your own actions. Other loops handle the rest.",
    config.canAct
      ? `# Tools you can propose\n${describeTools(tools)}\nEvery action needs a concrete prediction of its result; surprises are how the agent notices problems.`
      : "You cannot act. You influence the agent only by posting thoughts.",
    `# Reply format\nReply with ONLY a JSON object:\n${format}\nReturn {} when you have nothing useful to add. Be brief: every thought competes for very limited attention.`,
  ].join("\n\n");
}
