import { systemClock, type Clock } from "./io/clock";
import type { LLM, Tier } from "./io/llm";
import type { Models } from "./io/models";
import type { Storage } from "./io/storage";
import type { Tool } from "./io/tools";
import { builtinTools } from "./core/builtins";
import { Executive } from "./core/executive";
import { Gateway, type Appraisal, type BrainEvent } from "./core/gateway";
import { Body, calm, chooseRuntime, maxParallelLoops, updateModulators } from "./core/modulators";
import { Monitor, type Assessment } from "./core/monitor";
import { RiskTagger } from "./core/risk";
import { Selector, type Veto } from "./core/selector";
import type { Bid, Loop, LoopEnv, Modulators, NewProposal, Proposal } from "./core/types";
import { Workspace } from "./core/workspace";
import { Reflex } from "./loops/reflex";
import { defaultRoster } from "./loops/roster";
import { Specialist, type SpecialistConfig } from "./loops/specialist";
import { Consolidation } from "./memory/consolidation";
import { Memory } from "./memory/memory";
import { Senses, type Input } from "./perception/senses";

/** Results less surprising than this go back only to the loop that acted. */
const SURPRISE_THRESHOLD = 0.3;
/** Events quieter than this never bid for the workspace (habituation). */
const SALIENCE_THRESHOLD = 0.3;

export type BrainOptions = {
  models: Models;
  storage: Storage;
  /** Your tools. Built-ins (reply, set_goal, complete_goal, intend) are always added. */
  tools?: Tool[];
  /** Specialist loops to run. Defaults to planner, critic, user-model and overseer. */
  roster?: SpecialistConfig[];
  /** Extra hand-written loops. */
  loops?: Loop[];
  /** Where the "reply" tool sends text. */
  onSay?: (text: string) => void;
  clock?: Clock;
  /** Spend budget in abstract cost units (see Body). */
  budget?: number;
  workspaceCapacity?: number;
  /** Context window per loop, in tokens. */
  contextTokens?: number;
  /** How long the provider keeps a prompt prefix cached. */
  cacheTtlMs?: number;
};

export type Route = "reflex" | "feedback" | "workspace" | "ignored";

export type TickReport = {
  tick: number;
  events: { text: string; surprise: number; route: Route }[];
  broadcast: Bid[];
  woke: string[];
  chosen: Proposal[];
  vetoed: Veto[];
  assessment: Assessment;
  modulators: Modulators;
};

/**
 * Wires everything together and runs the cognitive cycle.
 *
 *   world → senses → gateway ─┬─ familiar input ──────────→ reflex (System 1 habit)
 *                             ├─ expected result ─────────→ back to the loop that acted
 *                             └─ salient / surprising ────→ workspace competition
 *   workspace winners → broadcast to subscribed loops → thoughts + proposals
 *   proposals → risk tagger → selector (one per effector) → act → result events → gateway
 *   monitor + modulators watch it all and retune model tier, thinking and parallelism.
 */
export class Brain {
  readonly memory: Memory;
  readonly workspace: Workspace;
  readonly executive: Executive;
  readonly body: Body;
  readonly loops: Loop[];
  modulators: Modulators = { ...calm };

  private readonly clock: Clock;
  private readonly models: Models;
  private readonly tools: Map<string, Tool>;
  private readonly senses: Senses;
  private readonly gateway: Gateway;
  private readonly selector: Selector;
  private readonly risk: RiskTagger;
  private readonly monitor = new Monitor();
  private readonly reflex: Reflex;
  private readonly consolidation: Consolidation;

  private queue: BrainEvent[] = [];
  private mail = new Map<string, Bid[]>();
  private tickNo = 0;
  private proposalNo = 0;
  private escalated = false;
  private userWaiting = false;

  static async create(options: BrainOptions): Promise<Brain> {
    const brain = new Brain(options);
    await brain.selector.load();
    return brain;
  }

  private constructor(opts: BrainOptions) {
    this.clock = opts.clock ?? systemClock;
    this.body = new Body(opts.budget ?? 1_000);
    this.models = {
      ...opts.models,
      system1: metered(opts.models.system1, "system1", this.body),
      system2: metered(opts.models.system2, "system2", this.body),
    };

    this.memory = new Memory(opts.storage, opts.models.embedder, this.clock);
    this.workspace = new Workspace(opts.workspaceCapacity ?? 7);
    this.executive = new Executive(opts.models.embedder, (key, value) => this.setFact(key, value));
    this.senses = new Senses(opts.models.live);
    this.gateway = new Gateway(this.models.system1);
    this.selector = new Selector(opts.storage);
    this.risk = new RiskTagger(this.models.system1);
    this.consolidation = new Consolidation(this.memory, this.models.system2, opts.storage, this.clock);

    const say = opts.onSay ?? ((text: string) => console.log(text));
    const tools = [...builtinTools({ say, executive: this.executive, memory: this.memory, clock: this.clock }), ...(opts.tools ?? [])];
    this.tools = new Map(tools.map((t) => [t.name, t]));

    const loopOptions = { clock: this.clock, maxTokens: opts.contextTokens ?? 100_000, cacheTtlMs: opts.cacheTtlMs ?? 5 * 60_000 };
    this.reflex = new Reflex(this.memory.skills, this.models, tools, loopOptions);
    this.loops = [
      this.reflex,
      ...(opts.roster ?? defaultRoster).map((config) => new Specialist(config, this.models, tools, loopOptions)),
      ...(opts.loops ?? []),
    ];
  }

  /** Feed input from the world: text, images, audio or video. */
  async hear(input: Input) {
    const focus = this.executive.current()?.text ?? "whatever the user needs";
    for await (const event of this.senses.perceive(input, focus)) this.queue.push(event);
    if (input.source === "user") this.userWaiting = true;
  }

  /** Is anything still waiting to be processed? */
  busy(): boolean {
    return this.queue.length > 0 || this.workspace.hasPending() || this.mail.size > 0;
  }

  /** Tick until things settle down (or we hit the safety limit). */
  async run(maxTicks = 20): Promise<TickReport[]> {
    const reports: TickReport[] = [];
    while (this.busy() && reports.length < maxTicks) reports.push(await this.tick());
    return reports;
  }

  /** One cognitive cycle. */
  async tick(): Promise<TickReport> {
    const tick = ++this.tickNo;
    const events: TickReport["events"] = [];
    let reward: number | undefined;

    // 1. Gateway: appraise each event against its prediction, learn from it, and route it.
    for (const event of this.queue.splice(0)) {
      const appraisal = await this.gateway.appraise(event, this.modulators);
      reward = (await this.learnFrom(appraisal)) ?? reward;
      for (const intention of await this.memory.prospective.triggeredBy(appraisal.text)) {
        this.workspace.submit({ kind: "intention", source: "memory", content: intention.then, salience: 0.8 });
      }
      events.push({ text: appraisal.text, surprise: appraisal.surprise, route: await this.route(appraisal) });
    }

    // 2. Competition for the workspace, biased top-down by the current goal.
    const bias = await this.executive.biasFor(this.workspace.candidates());
    const broadcast = this.workspace.compete((b) => bias.get(b.id) ?? 1);

    // 3. Broadcast: wake only loops that subscribe to something new or have mail.
    const work = this.loops
      .map((loop) => ({ loop, items: unique([...broadcast.filter((b) => loop.subscribes(b)), ...(this.mail.get(loop.id) ?? [])]) }))
      .filter((w) => w.items.length > 0);
    this.mail.clear();

    const env = this.env(tick);
    const outputs = await inBatches(work, maxParallelLoops(this.modulators, this.body), ({ loop, items }) => loop.onBroadcast(items, env));

    const proposals: Proposal[] = [];
    outputs.forEach((output, i) => {
      const source = work[i]!.loop.id;
      for (const bid of output.bids) this.workspace.submit(bid); // they compete next tick
      for (const p of output.proposals) proposals.push(this.normalize(p, source));
    });

    // 4. Risk check, then action selection: default no-go, at most one action per effector.
    const tagged = await this.risk.tag(proposals, this.tools);
    const { chosen, vetoed } = this.selector.select(tagged, this.tools, this.modulators, tick);
    for (const v of vetoed.filter((v) => v.proposal.risk === "high")) {
      this.workspace.submit({ kind: "alert", source: "selector", content: `Blocked ${v.proposal.tool}(${JSON.stringify(v.proposal.args)}): ${v.reason}`, salience: 0.9 });
    }

    // 5. Act. Each action carries its prediction; the result comes back as an event next tick.
    await Promise.all(chosen.map((p) => this.act(p)));

    // 6. Metacognition and global state.
    const madeProgress = reward !== undefined && reward > 0.5;
    const assessment = this.monitor.endTick(madeProgress, broadcast.filter((b) => b.contradicts?.length).length);
    this.escalated = assessment.escalate;
    if (assessment.changed && assessment.reasons.length) {
      const prefix = assessment.askHuman ? "Consider asking the user for help. " : "";
      this.workspace.submit({ kind: "alert", source: "monitor", content: prefix + assessment.reasons.join("; "), salience: 0.9 });
    }
    this.body.contextPressure = Math.max(0, ...this.loops.map((l) => l.contextPressure?.() ?? 0));
    this.modulators = updateModulators(this.modulators, {
      userWaiting: this.userWaiting,
      meanSurprise: this.monitor.meanSurprise(),
      reward,
      stuck: assessment.escalate,
      budgetLeft: this.body.budgetLeft(),
    });

    return { tick, events, broadcast, woke: work.map((w) => w.loop.id), chosen, vetoed, assessment, modulators: this.modulators };
  }

  /**
   * Call when there's nothing to do.
   * Default mode: check time-based intentions. Sleep: consolidate episodes into facts and skills.
   */
  async idle() {
    const due = await this.memory.prospective.due(this.clock());
    for (const intention of due) this.queue.push({ kind: "timer", text: `Reminder: ${intention.then}` });
    const consolidated = await this.consolidation.sleep();
    return { reminders: due.length, ...consolidated };
  }

  /** Publish a shared fact to every loop (arrives as a context delta, not a prefix rewrite). */
  setFact(key: string, value: string | undefined) {
    for (const loop of this.loops) loop.setFact?.(key, value);
  }

  private async route(a: Appraisal): Promise<Route> {
    const event = a.event;

    if (event.kind === "result") {
      const bid = {
        kind: event.ok ? ("result" as const) : ("error" as const),
        source: "gateway",
        content: a.text,
        salience: a.salience,
        provenance: [event.proposal.source],
      };
      // Expected: goes straight back to the loop that acted (like proprioception). Nobody else cares.
      if (a.surprise < SURPRISE_THRESHOLD) {
        this.sendMail(event.proposal.source, this.workspace.create(bid));
        return "feedback";
      }
      // Surprise: compete for the workspace, and make sure the actor hears about it too.
      this.sendMail(event.proposal.source, this.workspace.submit(bid));
      // A habit that misfires hands the original input back to deliberation.
      if (event.proposal.source === this.reflex.id) {
        const original = await this.reflex.abandon();
        if (original) this.workspace.submit({ kind: "percept", source: "user", content: original, salience: 1 });
      }
      return "workspace";
    }

    // Fast path: familiar input runs as a habit on the cheap model, skipping the workspace.
    if (event.kind === "percept" && (await this.reflex.claim(event.text))) {
      this.sendMail(this.reflex.id, this.workspace.create({ kind: "percept", source: event.source, content: event.text, attachments: event.attachments, salience: a.salience }));
      return "reflex";
    }

    // Slow path: bid for the workspace if it's loud enough; otherwise habituate.
    if (a.salience < SALIENCE_THRESHOLD) return "ignored";
    this.workspace.submit(
      event.kind === "percept"
        ? { kind: "percept", source: event.source, content: event.text, attachments: event.attachments, salience: a.salience }
        : { kind: "intention", source: "timer", content: event.text, salience: a.salience },
    );
    return "workspace";
  }

  /** Episodic logging + reward learning. Returns the reward if this was an action result. */
  private async learnFrom(a: Appraisal): Promise<number | undefined> {
    const event = a.event;
    let reward: number | undefined;
    if (event.kind === "result") {
      reward = event.ok ? 1 - a.surprise : 0;
      await this.selector.learn(event.proposal.tool, reward);
      this.monitor.recordOutcome(event.ok, a.surprise);
    }
    await this.memory.episodic.record({
      what: a.text,
      goal: this.executive.current()?.text,
      source: event.kind === "result" ? event.proposal.source : event.kind === "percept" ? event.source : "timer",
      surprise: a.surprise,
      valence: reward === undefined ? 0 : reward * 2 - 1,
    });
    return reward;
  }

  private async act(p: Proposal) {
    const tool = this.tools.get(p.tool)!;
    this.monitor.recordAction(p);
    if (tool.effector === "voice") this.userWaiting = false;
    try {
      this.queue.push({ kind: "result", proposal: p, ok: true, output: await tool.run(p.args) });
    } catch (err) {
      this.queue.push({ kind: "result", proposal: p, ok: false, output: err instanceof Error ? err.message : String(err) });
    }
  }

  private env(tick: number): LoopEnv {
    return {
      tick,
      runtimeFor: (tier) => chooseRuntime(tier, this.modulators, this.body, this.escalated),
      recall: (cue) => this.memory.recall(cue),
    };
  }

  private sendMail(loopId: string, bid: Bid) {
    this.mail.set(loopId, [...(this.mail.get(loopId) ?? []), bid]);
  }

  /** Models produce sloppy JSON; fill in safe defaults. */
  private normalize(p: NewProposal, source: string): Proposal {
    return {
      ...p,
      id: `p${++this.proposalNo}`,
      source,
      args: p.args ?? {},
      expect: p.expect ?? "",
      confidence: Number.isFinite(Number(p.confidence)) ? Number(p.confidence) : 0.5,
    };
  }
}

/** Wrap a model so every call is charged to the body's budget. */
function metered(llm: LLM, tier: Tier, body: Body): LLM {
  return {
    name: llm.name,
    generate: async (req) => {
      const res = await llm.generate(req);
      body.record(tier, res.usage);
      return res;
    },
  };
}

function unique(bids: Bid[]): Bid[] {
  return [...new Map(bids.map((b) => [b.id, b])).values()];
}

async function inBatches<T, R>(items: T[], size: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = [];
  for (let i = 0; i < items.length; i += size) results.push(...(await Promise.all(items.slice(i, i + size).map(fn))));
  return results;
}
