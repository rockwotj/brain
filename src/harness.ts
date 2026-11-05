/**
 * The run loop. One step =
 *   reflex (approved habits) -> woken specialists in parallel (System 1)
 *   -> monitor decides whether to escalate to the executive (System 2)
 *   -> selector releases at most one action -> actuator runs it -> outcome is broadcast.
 * Bounded by max steps, model-call budget, stall detection and an AbortSignal.
 */
import { makeAgent, type Agent, type SpecialistSpec } from "./agents.ts";
import { roughTokenEstimate, type LlmClient, type ModelProfile, type TokenEstimator } from "./llm.ts";
import { InMemoryEpisodicStore, InMemoryHabitStore, InMemoryKnowledgeStore, type EpisodicStore, type HabitStore, type KnowledgeStore } from "./memory.ts";
import { Monitor } from "./monitor.ts";
import { fallbackAnswer, narrate, type Answer } from "./narrator.ts";
import { addCandidate, select, type Candidate, type Origin, type SelectResult } from "./selector.ts";
import { callText, clip, describeEvent, InMemoryEventLog, StateStore, type WorkspaceBudget } from "./state.ts";
import { Actuator, type Permissions, type ToolPort, type ToolSpec } from "./tools.ts";
import type { ActionRecord, Constraint, Episode, HarnessEvent, Proposal, RunStatus } from "./types.ts";

export type Limits = { maxSteps: number; maxModelCalls: number; maxExecutiveCalls: number; stallSteps: number; carefulSteps: number };
const DEFAULT_LIMITS: Limits = { maxSteps: 12, maxModelCalls: 60, maxExecutiveCalls: 3, stallSteps: 2, carefulSteps: 2 };

export type HarnessConfig = {
  llm: LlmClient;
  profiles: { system1: ModelProfile; system2: ModelProfile; narrator?: ModelProfile };
  tools: readonly ToolSpec[];
  toolPort: ToolPort;
  permissions?: Permissions;
  specialists: readonly SpecialistSpec[];
  executiveInstructions?: string;
  memory?: { episodes?: EpisodicStore; knowledge?: KnowledgeStore; habits?: HabitStore };
  workspace?: WorkspaceBudget;
  estimate?: TokenEstimator;
  limits?: Partial<Limits>;
};

export type RunOptions = {
  goal: string;
  constraints?: Constraint[];
  urgency?: "normal" | "high";
  /** Ideas prepared while idle; admitted at low priority. */
  preparedIdeas?: string[];
  signal?: AbortSignal;
  /** Hook for injecting external events (e.g. a new user constraint) before each step. */
  onStep?: (step: number, state: StateStore) => void;
};

export type ModelCall = { step: number; agent: string; profile: string; tier: ModelProfile["tier"] };

export type RunResult = {
  status: RunStatus;
  steps: number;
  answer: Answer;
  episode: Episode;
  actions: ActionRecord[];
  events: HarnessEvent[];
  trace: string[];
  modelCalls: ModelCall[];
  usage: { inputTokens: number; cachedInputTokens: number; outputTokens: number };
  state: StateStore;
};

/** Event types that wake specialists by default (mode changes and notes do not). */
const WAKING = new Set(["goal.set", "constraint.add", "constraint.remove", "workspace.admit", "action.outcome", "action.blocked"]);

const EXECUTIVE_INSTRUCTIONS =
  "You are consulted only on novelty, conflict, surprise or stalls. Diagnose why, add 'focus' observations " +
  "to steer the specialists, and propose at most one next action or a finish backed by evidence.";

export class Harness {
  readonly episodes: EpisodicStore;
  readonly knowledge: KnowledgeStore;
  readonly habits: HabitStore;
  private readonly actuator: Actuator;
  private readonly limits: Limits;
  private readonly estimate: TokenEstimator;

  constructor(private readonly config: HarnessConfig) {
    this.episodes = config.memory?.episodes ?? new InMemoryEpisodicStore();
    this.knowledge = config.memory?.knowledge ?? new InMemoryKnowledgeStore();
    this.habits = config.memory?.habits ?? new InMemoryHabitStore();
    this.actuator = new Actuator(config.tools, config.toolPort, config.permissions ?? { allowEffects: ["read", "write"] });
    this.limits = { ...DEFAULT_LIMITS, ...config.limits };
    this.estimate = config.estimate ?? roughTokenEstimate;
  }

  async run(opts: RunOptions): Promise<RunResult> {
    const { profiles, tools } = this.config;
    const { signal } = opts;
    const trace: string[] = [];
    const events: HarnessEvent[] = [];
    const state = new StateStore(new InMemoryEventLog(), this.config.workspace, (e) => {
      events.push(e);
      trace.push(`  ${describeEvent(e)}`);
    });
    const specialists = this.config.specialists.map((spec) => ({
      spec,
      seen: 0,
      agent: makeAgent({ name: spec.name, role: "a narrow specialist", instructions: spec.instructions, toolNames: spec.tools, canFinish: !!spec.canFinish }, tools, profiles.system1, this.estimate),
    }));
    const executive = makeAgent(
      { name: "executive", role: "the executive (deliberate reasoning)", instructions: this.config.executiveInstructions ?? EXECUTIVE_INSTRUCTIONS, toolNames: tools.map((t) => t.name), canFinish: true },
      tools, profiles.system2, this.estimate);
    const monitor = new Monitor(this.limits.stallSteps);
    const records: ActionRecord[] = [];
    const executedKeys = new Set<string>();
    const reportedBlocks = new Set<string>();
    const modelCalls: ModelCall[] = [];
    const usage = { inputTokens: 0, cachedInputTokens: 0, outputTokens: 0 };
    let status: RunStatus = "max_steps";
    let step = 0;
    // Every model call in this run goes through this meter (specialists, executive, narrator).
    const llm: LlmClient = {
      complete: async (request) => {
        modelCalls.push({ step, agent: request.agent, profile: request.profile.id, tier: request.profile.tier });
        const res = await this.config.llm.complete(request);
        usage.inputTokens += res.usage.inputTokens;
        usage.cachedInputTokens += res.usage.cachedInputTokens ?? 0;
        usage.outputTokens += res.usage.outputTokens;
        return res;
      },
    };

    // One model call is always held back so the narrator can run within maxModelCalls.
    const budgetFor = (calls: number): boolean => modelCalls.length + calls <= this.limits.maxModelCalls - 1;

    const consult = async (agent: Agent, request: string): Promise<Proposal[]> => {
      const reply = await agent.propose(llm, state, request, signal);
      for (const issue of reply.issues) trace.push(`  ! ${agent.name}: ${issue}`);
      return reply.proposals;
    };
    const absorb = (agent: Agent, proposals: Proposal[], origin: Origin, pool: Candidate[]): void => {
      for (const p of proposals) {
        if (p.kind !== "observation") {
          addCandidate(pool, p, origin, { agent: agent.name, profileId: agent.profile.id });
          continue;
        }
        const priority = p.type === "focus" ? "high" : p.confidence === "low" ? "low" : "normal";
        state.admit(agent.name, { kind: p.type, content: p.content, priority, evidence: p.evidence });
      }
    };
    const selectNow = (pool: Candidate[]): SelectResult =>
      select(pool, {
        actuator: this.actuator,
        constraints: state.pinned(),
        executedKeys,
        successfulActions: new Set(records.filter((r) => r.actual.ok).map((r) => r.id)),
        careful: state.mode.carefulSteps > 0,
      });

    // --- set up: goal, pinned constraints, recalled memory (as unverified claims) ---
    state.apply("user", { type: "goal.set", goal: opts.goal });
    for (const constraint of opts.constraints ?? []) state.apply("user", { type: "constraint.add", constraint });
    if (opts.urgency === "high") state.apply("user", { type: "mode.set", mode: { ...state.mode, urgency: "high" } });
    for (const k of this.knowledge.recall(opts.goal, 2)) {
      state.admit("recall", { kind: "memory", content: `Recalled claim (unverified): ${k.statement}`, priority: "normal", evidence: [k.id, ...k.sources] });
    }
    for (const ep of this.episodes.recall(opts.goal, 1)) {
      state.admit("recall", { kind: "memory", content: `Past episode (${ep.status}): ${clip(ep.answer ?? ep.goal, 160)}`, priority: "low", evidence: [ep.id] });
    }
    for (const idea of opts.preparedIdeas ?? []) state.admit("background", { kind: "idea", content: idea, priority: "low", evidence: [] });

    try {
      while (step < this.limits.maxSteps) {
        step++;
        signal?.throwIfAborted();
        opts.onStep?.(step, state);
        const startSeq = state.seq;
        trace.push(`step ${step}`);

        // Reflex path: approved habits propose their call (still subject to policy).
        const habitPool: Candidate[] = [];
        for (const habit of this.habits.match(state.goal)) {
          addCandidate(habitPool, { kind: "action", call: habit.call, rationale: `habit ${habit.id}`, expect: habit.expect }, "habit", { agent: `habit:${habit.id}`, profileId: "habit" }, habit.id);
        }
        const fresh = habitPool.filter((c) => !executedKeys.has(c.key));
        const reflex = this.reflexCandidate(fresh, state, monitor);
        let executed = false;
        if (reflex && reflex.proposal.kind === "action") {
          // Fast path: one unambiguous, read-only, policy-allowed habit and nothing for the
          // monitor to handle -> run it now and skip deliberation (no model calls) this step.
          trace.push(`  reflex ${callText(reflex.proposal.call)}: approved read-only habit ${reflex.habitId}; deliberation skipped`);
          executed = await this.execute(reflex, reflex.proposal.expect, step, state, records, executedKeys, monitor, signal);
        } else {
          const pool: Candidate[] = [...fresh];

          // System 1: wake only specialists with relevant news from someone else; run them in parallel.
          const woken = specialists.filter((s) =>
            state.eventsSince(s.seen).some((e) => e.source !== s.spec.name && (s.spec.wakeOn ?? ((ev) => WAKING.has(ev.delta.type)))(e)));
          if (!budgetFor(woken.length)) {
            status = "budget_exhausted";
            break;
          }
          if (woken.length) trace.push(`  wake [${woken.map((s) => s.spec.name).join(", ")}] in parallel on ${profiles.system1.id}`);
          for (const s of woken) s.seen = state.seq;
          const replies = await Promise.all(woken.map((s) => consult(s.agent, `Step ${step}: propose only what is new and useful; otherwise reply with no proposals.`)));
          signal?.throwIfAborted();
          replies.forEach((proposals, i) => {
            const s = woken[i];
            if (s) absorb(s.agent, proposals, "specialist", pool);
          });

          // Monitor: decide whether the expensive executive is worth it.
          if (step === 1 && pool.length === 0) monitor.flag("novelty");
          let result = selectNow(pool);
          if (result.selection.kind === "hold") monitor.flag("conflict");
          const reasons = monitor.takeReasons();
          const executiveCalls = modelCalls.filter((c) => c.agent === "executive").length;
          if (reasons.length && executiveCalls < this.limits.maxExecutiveCalls && budgetFor(1)) {
            trace.push(`  escalate (${reasons.join(", ")}) -> executive on ${profiles.system2.id}`);
            monitor.executiveConsulted();
            const proposals = await consult(executive, `ESCALATION: ${reasons.join(", ")}. Diagnose, steer with focus items, and propose the next action or a finish.`);
            signal?.throwIfAborted();
            absorb(executive, proposals, "executive", pool);
            result = selectNow(pool);
          } else if (reasons.length) trace.push(`  escalation (${reasons.join(", ")}) skipped: executive or model-call budget used`);

          // Hard policy: blocked proposals are broadcast once so agents can adapt.
          for (const b of result.blocked) {
            const key = `${callText(b.call)}|${b.reason}`;
            if (!reportedBlocks.has(key)) state.apply("policy", { type: "action.blocked", call: b.call, reason: b.reason });
            reportedBlocks.add(key);
          }
          for (const r of result.rejected) trace.push(`  - ${r}`);
          const sel = result.selection;
          const chosen = sel.kind === "act" && sel.candidate.proposal.kind === "action" ? callText(sel.candidate.proposal.call) : sel.kind;
          trace.push(`  select ${chosen}: ${sel.why}`);

          if (sel.kind === "finish" && sel.candidate.proposal.kind === "finish") {
            const p = sel.candidate.proposal;
            state.apply(sel.candidate.supporters[0]?.agent ?? "unknown", { type: "goal.done", summary: p.summary, evidence: p.evidence });
            status = "finished";
            break;
          }
          if (sel.kind === "act" && sel.candidate.proposal.kind === "action") {
            executed = await this.execute(sel.candidate, sel.candidate.proposal.expect, step, state, records, executedKeys, monitor, signal);
          } else if (state.mode.carefulSteps > 0) {
            state.apply("monitor", { type: "mode.set", mode: { ...state.mode, carefulSteps: state.mode.carefulSteps - 1 } });
          }
        }
        const admitted = state.eventsSince(startSeq).some((e) => e.delta.type === "workspace.admit");
        if (monitor.endStep(executed || admitted) === "stop") {
          status = "no_progress";
          break;
        }
      }
    } catch (error) {
      if (!signal?.aborted) throw error;
      status = "cancelled";
    }

    const narratorProfile = profiles.narrator ?? profiles.system1;
    const known = new Set(records.map((r) => r.id));
    const answer: Answer = status !== "finished"
      ? { text: `Stopped (${status}) after ${step} steps without a finished answer.`, citations: [], unknownCitations: [], grounded: false }
      : modelCalls.length < this.limits.maxModelCalls
        ? await narrate(llm, narratorProfile, this.estimate, state, records, signal)
        : fallbackAnswer(state, known);
    const episode = this.episodes.append({ goal: opts.goal, status, actions: records, answer: answer.text });
    trace.push(`end: ${status} after ${step} steps; episode ${episode.id}`);
    return { status, steps: step, answer, episode, actions: records, events, trace, modelCalls, usage, state };
  }

  /** A habit may bypass deliberation only if it is the single eligible one, read-only and allowed. */
  private reflexCandidate(fresh: Candidate[], state: StateStore, monitor: Monitor): Candidate | undefined {
    const only = fresh.length === 1 ? fresh[0] : undefined;
    if (!only || only.proposal.kind !== "action" || monitor.hasPending() || state.mode.carefulSteps > 0) return undefined;
    const decision = this.actuator.check(only.proposal.call, state.pinned());
    return decision.allowed && decision.spec.effect === "read" ? only : undefined;
  }

  /** The only path from a selected proposal to a real side effect. */
  private async execute(
    candidate: Candidate, expect: ActionRecord["predicted"], step: number, state: StateStore,
    records: ActionRecord[], executedKeys: Set<string>, monitor: Monitor, signal?: AbortSignal,
  ): Promise<boolean> {
    if (candidate.proposal.kind !== "action") return false;
    const call = candidate.proposal.call;
    const source = candidate.supporters[0]?.agent ?? "unknown";
    const ex = await this.actuator.execute(call, { step, source, predicted: expect, constraints: state.pinned() }, signal);
    if (ex.kind === "blocked") {
      state.apply("policy", { type: "action.blocked", call, reason: ex.reason });
      return false;
    }
    const r = ex.record;
    records.push(r);
    executedKeys.add(candidate.key);
    state.apply("actuator", { type: "action.outcome", record: r });
    const outcome = r.actual.ok ? r.actual.output : `ERROR ${r.actual.error}`;
    state.admit("actuator", { kind: "outcome", content: `${r.call.tool}: ${outcome}`, priority: r.surprise ? "high" : "normal", evidence: [r.id] });
    monitor.observeOutcome(r);
    if (candidate.habitId) this.habits.recordOutcome(candidate.habitId, !r.surprise);
    const carefulSteps = r.surprise || !r.actual.ok ? this.limits.carefulSteps : Math.max(0, state.mode.carefulSteps - 1);
    if (carefulSteps !== state.mode.carefulSteps) state.apply("monitor", { type: "mode.set", mode: { ...state.mode, carefulSteps } });
    return true;
  }
}
