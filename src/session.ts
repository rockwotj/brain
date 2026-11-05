/**
 * A session is one conversation: a persistent event log plus long-lived agents
 * whose transcripts carry across turns (new turns arrive as deltas).
 *
 * One turn = perceive, then bounded steps of:
 *   habits (reflex) -> woken specialists in parallel (System 1)
 *   -> escalate to the executive (System 2) only if the monitor asks
 *   -> select at most one candidate -> act (via the actuator) and route the outcome, or finish.
 * Bounded by max steps, a model-call budget, stall detection and an AbortSignal.
 */
import { makeAgent, type Agent, type SpecialistSpec } from "./agents.ts";
import { describeMedia, isMedia, textOf, toParts, type ContentPart, type MediaPart } from "./io/content.ts";
import { embedOne } from "./io/embeddings.ts";
import { chooseProfile, UnsupportedContentError, type LlmClient, type ModelProfile } from "./io/llm.ts";
import type { Harness } from "./harness.ts";
import { Monitor } from "./monitor.ts";
import { fallbackAnswer, makeNarrator, narrate, type Answer } from "./narrator.ts";
import { addCandidate, select, type Candidate, type Origin, type SelectResult } from "./selector.ts";
import { callText, clip, describeEvent, outcomeSummary, perceptionSummary, visibleTo, type StateStore } from "./state.ts";
import type { ActionRecord, Completion, Constraint, Effect, Episode, HarnessEvent, PerceptionRecord, Proposal, RecordReadProposal, RunStatus } from "./types.ts";

export type TurnOptions = {
  /** What the user said: text, or content parts (text, image, audio, video). */
  input: string | readonly ContentPart[];
  constraints?: Constraint[];
  urgency?: "normal" | "high";
  /** Ideas prepared while idle; admitted at low priority. */
  preparedIdeas?: string[];
  /** Task boundary: clear workspace and constraints, and start fresh transcript epochs. */
  newTask?: boolean;
  signal?: AbortSignal;
  /** Hook for injecting external events (e.g. a new user constraint) before each step. May be async. */
  onStep?: (step: number, state: StateStore) => unknown;
};

export type ModelCall = { step: number; agent: string; profile: string; tier: ModelProfile["tier"] };

export type RunResult = {
  status: RunStatus;
  steps: number;
  /** Set when status is "finished". */
  completion?: Completion;
  answer: Answer;
  episode: Episode;
  /** Actions executed this turn. */
  actions: ActionRecord[];
  /** Events appended this turn. */
  events: HarnessEvent[];
  trace: string[];
  modelCalls: ModelCall[];
  usage: { inputTokens: number; cachedInputTokens: number; outputTokens: number };
  state: StateStore;
};

/** Event types that wake specialists by default (mode changes and notes do not). */
const WAKING = new Set(["goal.set", "constraint.add", "constraint.remove", "workspace.admit", "action.outcome", "action.blocked", "perception"]);

const EXECUTIVE_INSTRUCTIONS =
  "You are consulted only on novelty, conflict, surprise, stalls, images only you can see, or a record you asked for. Diagnose why, add 'focus' " +
  "observations to steer the specialists, and propose at most one next action or a finish backed by evidence.";

/** Per-turn bookkeeping. */
type Turn = {
  step: number;
  status: RunStatus;
  completion?: Completion;
  records: ActionRecord[];
  events: HarnessEvent[];
  trace: string[];
  modelCalls: ModelCall[];
  usage: RunResult["usage"];
  monitor: Monitor;
  reportedBlocks: Set<string>;
  /** Reads may repeat once something was written (state may have changed). */
  readsSinceWrite: Set<string>;
  /** An identical write runs at most once per turn. */
  writesThisTurn: Set<string>;
  /** A record was delivered this step (counts as progress). */
  delivered: boolean;
  /** Images arrived that only the executive can see, and it has not looked yet: no action may run. */
  mediaUnseen: boolean;
  signal?: AbortSignal;
};

function newTurn(stallSteps: number, signal?: AbortSignal): Turn {
  return {
    step: 0, status: "max_steps", records: [], events: [], trace: [], modelCalls: [],
    usage: { inputTokens: 0, cachedInputTokens: 0, outputTokens: 0 }, monitor: new Monitor(stallSteps),
    reportedBlocks: new Set(), readsSinceWrite: new Set(), writesThisTurn: new Set(), delivered: false, mediaUnseen: false,
    ...(signal ? { signal } : {}),
  };
}

export class Session {
  private readonly specialists: { spec: SpecialistSpec; agent: Agent; seen: number }[];
  private readonly executive: Agent;
  private readonly narrator: Agent;
  /** Every model call goes through this meter (specialists, executive, narrator). */
  private readonly llm: LlmClient;
  private t: Turn;
  private busy = false;

  constructor(readonly id: string, readonly state: StateStore, private readonly h: Harness) {
    const { profiles, tools, clock } = h.config;
    const options = clock ? { clock } : {};
    this.specialists = h.config.specialists.map((spec) => ({
      spec,
      seen: state.seq,
      agent: makeAgent(
        { name: spec.name, role: "a narrow specialist", instructions: spec.instructions, toolNames: spec.tools, canFinish: !!spec.canFinish },
        tools, chooseProfile(profiles, "specialist"), h.estimate, options),
    }));
    this.executive = makeAgent(
      { name: "executive", role: "the executive (deliberate reasoning)", instructions: h.config.executiveInstructions ?? EXECUTIVE_INSTRUCTIONS, toolNames: tools.map((t) => t.name), canFinish: true },
      tools, chooseProfile(profiles, "executive"), h.estimate, options);
    this.narrator = makeNarrator(chooseProfile(profiles, "narrator"), h.estimate, clock);
    this.t = newTurn(h.limits.stallSteps);
    state.onEvent = (e) => {
      this.t.events.push(e);
      this.t.trace.push(`  ${describeEvent(e, state.previewChars)}`);
    };
    this.llm = {
      complete: async (request) => {
        const t = this.t;
        t.modelCalls.push({ step: t.step, agent: request.agent, profile: request.profile.id, tier: request.profile.tier });
        const res = await h.config.llm.complete(request);
        t.usage.inputTokens += res.usage.inputTokens;
        t.usage.cachedInputTokens += res.usage.cachedInputTokens ?? 0;
        t.usage.outputTokens += res.usage.outputTokens;
        return res;
      },
    };
  }

  async turn(opts: TurnOptions): Promise<RunResult> {
    if (this.busy) throw new Error(`session ${this.id}: a turn is already running`);
    this.busy = true;
    const t = (this.t = newTurn(this.h.limits.stallSteps, opts.signal));
    try {
      try {
        await this.perceive(opts);
        while (t.step < this.h.limits.maxSteps) {
          t.step++;
          opts.signal?.throwIfAborted();
          await opts.onStep?.(t.step, this.state);
          t.trace.push(`step ${t.step}`);
          const outcome = await this.step();
          if (outcome !== "continue") {
            t.status = outcome;
            break;
          }
        }
      } catch (error) {
        if (!opts.signal?.aborted) throw error;
        t.status = "cancelled";
      }
      return await this.wrapUp();
    } finally {
      this.busy = false;
    }
  }

  // --- perceive: input -> goal, constraints, live observations, recalled memory ---

  private async perceive(opts: TurnOptions): Promise<void> {
    const parts = toParts(opts.input);
    const media = parts.filter(isMedia);
    const images = media.filter((m) => m.type === "image");
    const streams = media.filter((m) => m.type !== "image");
    for (const m of media) if (!m.base64 && !m.url) throw new UnsupportedContentError(`${describeMedia(m)} needs base64 data or a url`);
    const { profiles, live } = this.h.config;
    if (images.length && !profiles.system1.vision && !profiles.system2.vision) {
      throw new UnsupportedContentError("image input needs a model profile with vision: true");
    }
    const executiveOnlyVision = images.length > 0 && !profiles.system1.vision;
    if (executiveOnlyVision && this.h.limits.maxExecutiveCalls < 1) {
      throw new UnsupportedContentError("image input can only be seen by the executive, but limits.maxExecutiveCalls is 0");
    }
    if (streams[0] && !live) throw new UnsupportedContentError(`${describeMedia(streams[0])} input needs a LiveModel; none is configured`);

    if (opts.newTask) await this.startNewTask();
    const goal = textOf(parts).trim() || "(no text: see the attached media and observations)";
    await this.state.apply("user", { type: "goal.set", goal, ...(images.length ? { attachments: images } : {}) });
    // Only the executive can see these images: consult it before any (blind) action is released.
    if (executiveOnlyVision) {
      this.t.mediaUnseen = true;
      this.t.monitor.flag("media");
    }
    for (const constraint of opts.constraints ?? []) await this.state.apply("user", { type: "constraint.add", constraint });
    const urgency = opts.urgency ?? "normal";
    if (this.state.mode.urgency !== urgency || this.state.mode.carefulSteps !== 0) {
      await this.state.apply("user", { type: "mode.set", mode: { urgency, carefulSteps: 0 } });
    }
    await this.listen(streams, goal);
    await this.recallMemory(goal);
    for (const idea of opts.preparedIdeas ?? []) await this.state.admit("background", { kind: "idea", content: idea, priority: "low", evidence: [] });
  }

  /**
   * Audio/video go to the LiveModel. Its observations (bounded) are collected into one
   * perception record per input; the workspace gets a single pointer to it.
   */
  private async listen(streams: readonly MediaPart[], focus: string): Promise<void> {
    const live = this.h.config.live;
    if (!live) return;
    for (const media of streams) {
      const observations: string[] = [];
      for await (const observation of live.observe(media, focus, this.t.signal)) {
        observations.push(observation);
        if (observations.length >= this.h.limits.maxLiveObservations) break;
      }
      const record: PerceptionRecord = {
        id: this.state.nextPerceptionId(), source: `live:${live.name}`, medium: media.type === "video" ? "video" : "audio", mimeType: media.mimeType, observations,
      };
      await this.state.apply(record.source, { type: "perception", record });
      const preview = perceptionSummary(record, Math.max(40, this.state.budget.maxItemChars - 120));
      await this.state.admit(record.source, { kind: "observation", content: `${record.id} ${preview}`, priority: "high", evidence: [record.id] });
    }
  }

  /** Recalled memory enters as unverified claims. Needs an Embedder. */
  private async recallMemory(goal: string): Promise<void> {
    const embedder = this.h.config.embedder;
    if (!embedder) return;
    const cue = await embedOne(embedder, goal, this.t.signal);
    const { knowledge, episodes, minSimilarity } = this.h.recall;
    for (const k of await this.h.knowledge.recall(cue, { limit: knowledge, minSimilarity })) {
      await this.state.admit("recall", { kind: "memory", content: `Recalled claim (unverified): ${k.statement}`, priority: "normal", evidence: [k.id, ...k.sources] });
    }
    for (const ep of await this.h.episodes.recall(cue, { limit: episodes, minSimilarity })) {
      await this.state.admit("recall", { kind: "memory", content: `Past episode (${ep.status}): ${clip(ep.answer ?? ep.goal, 160)}`, priority: "low", evidence: [ep.id] });
    }
  }

  private async startNewTask(): Promise<void> {
    for (const item of [...this.state.workspace]) await this.state.apply("user", { type: "workspace.evict", id: item.id, reason: "new task" });
    for (const c of this.state.pinned()) await this.state.apply("user", { type: "constraint.remove", id: c.id });
    for (const agent of [this.executive, this.narrator, ...this.specialists.map((s) => s.agent)]) {
      agent.transcript.reset();
      agent.inbox.length = 0;
    }
  }

  // --- one step ---

  private async step(): Promise<RunStatus | "continue"> {
    const t = this.t;
    const startSeq = this.state.seq;
    t.delivered = false;
    const habits = await this.habitCandidates();
    const reflex = this.reflexCandidate(habits);
    let acted = false;
    if (reflex?.proposal.kind === "action") {
      // Fast path: one unambiguous, read-only, policy-allowed habit and nothing for the
      // monitor to handle -> run it now and skip deliberation (no model calls) this step.
      t.trace.push(`  reflex ${callText(reflex.proposal.call)}: approved read-only habit ${reflex.habitId}; deliberation skipped`);
      acted = await this.act(reflex);
    } else {
      const pool = [...habits];
      if (!(await this.consultSpecialists(pool))) return "budget_exhausted";
      if (t.step === 1 && pool.length === 0 && !t.delivered) t.monitor.flag("novelty");
      let result = this.arbitrate(pool);
      if (result.selection.kind === "hold") t.monitor.flag("conflict");
      const escalated = await this.escalate(pool, result);
      if (escalated === "budget_exhausted") return escalated;
      result = escalated;
      await this.report(result);
      if (t.mediaUnseen) {
        // The executive's call failed before it saw the images: release nothing, retry next step.
        t.trace.push("  hold: images not yet examined by the executive");
        t.monitor.flag("media");
        return "continue";
      }
      const sel = result.selection;
      if (sel.kind === "finish") {
        await this.complete(sel.candidate, sel.completion);
        return "finished";
      }
      if (sel.kind === "act") acted = await this.act(sel.candidate);
      else await this.easeCaution();
    }
    return t.monitor.endStep(acted || t.delivered || this.meaningfulChange(startSeq)) === "stop" ? "no_progress" : "continue";
  }

  /**
   * Progress for stall detection: new evidence, executive focus or a goal/constraint change.
   * Plain observations do not count, so rephrased chatter cannot keep a stuck turn alive.
   */
  private meaningfulChange(sinceSeq: number): boolean {
    return this.state.eventsSince(sinceSeq).some((e) =>
      e.delta.type === "goal.set" || e.delta.type === "constraint.add" || e.delta.type === "constraint.remove" ||
      (e.delta.type === "workspace.admit" && (e.delta.item.kind === "focus" || e.delta.item.kind === "outcome")));
  }

  /** Approved habits matching the goal, minus calls that may not repeat yet. */
  private async habitCandidates(): Promise<Candidate[]> {
    const pool: Candidate[] = [];
    for (const habit of await this.h.habits.match(this.state.goal)) {
      addCandidate(pool, { kind: "action", call: habit.call, rationale: `habit ${habit.id}`, expect: habit.expect }, "habit", { agent: `habit:${habit.id}`, profileId: "habit" }, habit.id);
    }
    return pool.filter((c) => !this.repeatBlock(c.key, this.effectOf(c)));
  }

  /** A habit may bypass deliberation only if it is the single eligible one, read-only and allowed. */
  private reflexCandidate(habits: Candidate[]): Candidate | undefined {
    const only = habits.length === 1 ? habits[0] : undefined;
    if (!only || only.proposal.kind !== "action" || this.t.mediaUnseen || this.t.monitor.hasPending() || this.state.mode.carefulSteps > 0) return undefined;
    const decision = this.h.actuator.check(only.proposal.call, this.state.pinned());
    return decision.allowed && decision.spec.effect === "read" ? only : undefined;
  }

  /** System 1: wake only specialists with visible news from someone else (or private notes); run them in parallel. */
  private async consultSpecialists(pool: Candidate[]): Promise<boolean> {
    const t = this.t;
    const woken = this.specialists.filter((s) =>
      s.agent.inbox.length > 0 ||
      this.state.eventsSince(s.seen).some((e) => e.source !== s.spec.name && visibleTo(e, s.spec.name) && (s.spec.wakeOn ?? ((ev) => WAKING.has(ev.delta.type)))(e)));
    // Keep one call in reserve for the executive while it still has images to examine.
    if (!this.budgetFor(woken.length + (t.mediaUnseen ? 1 : 0))) return false;
    if (woken.length) t.trace.push(`  wake [${woken.map((s) => s.spec.name).join(", ")}] in parallel on ${this.h.config.profiles.system1.id}`);
    for (const s of woken) s.seen = this.state.seq;
    const replies = await Promise.all(woken.map(async (s) => (await this.consult(s.agent, `Step ${t.step}: propose only what is new and useful; otherwise reply with no proposals.`)).proposals));
    t.signal?.throwIfAborted();
    for (const [i, proposals] of replies.entries()) {
      const s = woken[i];
      if (s) await this.absorb(s.agent, proposals, "specialist", pool);
    }
    return true;
  }

  /** System 2, on demand: the monitor decides whether the expensive executive is worth it. */
  private async escalate(pool: Candidate[], result: SelectResult): Promise<SelectResult | "budget_exhausted"> {
    const t = this.t;
    const reasons = t.monitor.takeReasons();
    if (this.executive.inbox.length) reasons.push("follow_up");
    if (!reasons.length) return result;
    const used = t.modelCalls.filter((c) => c.agent === "executive").length;
    if (used >= this.h.limits.maxExecutiveCalls || !this.budgetFor(1)) {
      t.trace.push(`  escalation (${reasons.join(", ")}) skipped: executive or model-call budget used`);
      if (t.mediaUnseen) {
        t.trace.push("  stop: images only the executive can see were never examined; no blind action released");
        return "budget_exhausted";
      }
      return result;
    }
    t.trace.push(`  escalate (${reasons.join(", ")}) -> executive on ${this.h.config.profiles.system2.id}`);
    t.monitor.executiveConsulted();
    const reply = await this.consult(this.executive, `ESCALATION: ${reasons.join(", ")}. Diagnose, steer with focus items, and propose the next action or a finish.`);
    t.signal?.throwIfAborted();
    if (!reply.failed) t.mediaUnseen = false;
    await this.absorb(this.executive, reply.proposals, "executive", pool);
    return this.arbitrate(pool);
  }

  private arbitrate(pool: Candidate[]): SelectResult {
    const successful = [...this.state.records.values()].filter((r) => r.actual.ok).map((r) => r.id);
    return select(pool, {
      actuator: this.h.actuator,
      constraints: this.state.pinned(),
      repeatBlock: (key, effect) => this.repeatBlock(key, effect),
      knownIds: this.state.knownIds,
      successfulActions: new Set(successful),
      careful: this.state.mode.carefulSteps > 0,
    });
  }

  /** Hard policy: blocked proposals are broadcast once so agents can adapt. */
  private async report(result: SelectResult): Promise<void> {
    for (const b of result.blocked) {
      const key = `${callText(b.call)}|${b.reason}`;
      if (!this.t.reportedBlocks.has(key)) await this.state.apply("policy", { type: "action.blocked", call: b.call, reason: b.reason });
      this.t.reportedBlocks.add(key);
    }
    for (const r of result.rejected) this.t.trace.push(`  - ${r}`);
    const sel = result.selection;
    const chosen = sel.kind === "act" && sel.candidate.proposal.kind === "action" ? callText(sel.candidate.proposal.call) : sel.kind;
    this.t.trace.push(`  select ${chosen}: ${sel.why}`);
  }

  private async complete(candidate: Candidate, completion: Completion): Promise<void> {
    if (candidate.proposal.kind !== "finish") return;
    const { summary, evidence } = candidate.proposal;
    this.t.completion = completion;
    await this.state.apply(candidate.supporters[0]?.agent ?? "unknown", { type: "goal.done", summary, evidence, completion });
  }

  // --- act and route ---

  /** The only path from a selected proposal to a real side effect. */
  private async act(candidate: Candidate): Promise<boolean> {
    if (candidate.proposal.kind !== "action") return false;
    const t = this.t;
    const { call, expect } = candidate.proposal;
    const source = candidate.supporters[0]?.agent ?? "unknown";
    const ex = await this.h.actuator.execute(call, { id: this.state.nextRecordId(), step: t.step, source, predicted: expect, constraints: this.state.pinned() }, t.signal);
    if (ex.kind === "blocked") {
      await this.state.apply("policy", { type: "action.blocked", call, reason: ex.reason });
      return false;
    }
    const record = ex.record;
    t.records.push(record);
    this.noteExecuted(candidate.key, ex.effect);
    await this.routeOutcome(record, candidate);
    t.monitor.observeOutcome(record);
    if (candidate.habitId) await this.h.habits.recordOutcome(candidate.habitId, !record.surprise);
    const carefulSteps = record.surprise || !record.actual.ok ? this.h.limits.carefulSteps : Math.max(0, this.state.mode.carefulSteps - 1);
    if (carefulSteps !== this.state.mode.carefulSteps) await this.state.apply("monitor", { type: "mode.set", mode: { ...this.state.mode, carefulSteps } });
    return true;
  }

  /**
   * Expected outcomes of a specialist's action go back to the proposing specialist(s) only;
   * surprises, failures and executive/habit actions are broadcast and enter the workspace.
   */
  private async routeOutcome(record: ActionRecord, candidate: Candidate): Promise<void> {
    const actors = candidate.origin === "specialist" ? [...new Set(candidate.supporters.map((s) => s.agent))] : undefined;
    const broadcast = !actors || record.surprise || !record.actual.ok;
    await this.state.apply("actuator", { type: "action.outcome", record }, broadcast ? undefined : actors);
    if (!broadcast) return;
    const preview = outcomeSummary(record, Math.max(40, this.state.budget.maxItemChars - 100));
    await this.state.admit("actuator", { kind: "outcome", content: `${record.call.tool} ${preview}`, priority: record.surprise ? "high" : "normal", evidence: [record.id] });
  }

  private noteExecuted(key: string, effect: Effect): void {
    if (effect === "write") {
      this.t.writesThisTurn.add(key);
      this.t.readsSinceWrite.clear();
    } else this.t.readsSinceWrite.add(key);
  }

  private repeatBlock(key: string, effect: Effect): string | undefined {
    if (effect === "write") return this.t.writesThisTurn.has(key) ? "identical write already executed this turn" : undefined;
    return this.t.readsSinceWrite.has(key) ? "identical read already ran since the last write" : undefined;
  }

  private effectOf(c: Candidate): Effect {
    return c.proposal.kind === "action" ? (this.h.actuator.spec(c.proposal.call.tool)?.effect ?? "write") : "read";
  }

  private async easeCaution(): Promise<void> {
    const mode = this.state.mode;
    if (mode.carefulSteps > 0) await this.state.apply("monitor", { type: "mode.set", mode: { ...mode, carefulSteps: mode.carefulSteps - 1 } });
  }

  // --- agents ---

  private async consult(agent: Agent, request: string): Promise<{ proposals: Proposal[]; failed?: boolean }> {
    const reply = await agent.propose(this.llm, this.state, request, this.t.signal);
    for (const issue of reply.issues) this.t.trace.push(`  ! ${agent.name}: ${issue}`);
    return reply;
  }

  private async absorb(agent: Agent, proposals: Proposal[], origin: Origin, pool: Candidate[]): Promise<void> {
    for (const p of proposals) {
      if (p.kind === "observation") {
        const priority = p.type === "focus" ? "high" : p.confidence === "low" ? "low" : "normal";
        await this.state.admit(agent.name, { kind: p.type, content: p.content, priority, evidence: p.evidence });
      } else if (p.kind === "read_record") {
        this.deliverRecord(agent, p);
      } else {
        addCandidate(pool, p, origin, { agent: agent.name, profileId: agent.profile.id });
      }
    }
  }

  /** Full record content, in chunks, privately to the agent that asked (no side effects, no arbitration). */
  private deliverRecord(agent: Agent, p: RecordReadProposal): void {
    const evidence = this.state.findEvidence(p.id);
    if (!evidence) {
      agent.inbox.push(`RECORD ${p.id}: unknown id (see the evidence index).`);
      return;
    }
    const full = evidence.text;
    const end = Math.min(full.length, p.offset + this.h.evidence.recordChunkChars);
    const more = end < full.length ? `\n[more: read_record ${p.id} offset ${end}]` : "";
    agent.inbox.push(`RECORD ${p.id} ${evidence.label}, chars ${p.offset}-${end} of ${full.length}:\n${full.slice(p.offset, end)}${more}`);
    this.t.delivered = true;
    this.t.trace.push(`  ${agent.name} reads ${p.id} chars ${p.offset}-${end} (private)`);
  }

  /** One model call is always held back so the narrator can run within maxModelCalls. */
  private budgetFor(calls: number): boolean {
    return this.t.modelCalls.length + calls <= this.h.limits.maxModelCalls - 1;
  }

  // --- wrap up ---

  private async wrapUp(): Promise<RunResult> {
    const t = this.t;
    const answer = await this.answer();
    const episode = await this.h.episodes.append({ goal: this.state.goal, status: t.status, actions: t.records, answer: answer.text });
    t.trace.push(`end: ${t.status}${t.completion ? ` (${t.completion})` : ""} after ${t.step} steps; episode ${episode.id}`);
    return {
      status: t.status, steps: t.step, ...(t.completion ? { completion: t.completion } : {}), answer, episode,
      actions: t.records, events: t.events, trace: t.trace, modelCalls: t.modelCalls, usage: t.usage, state: this.state,
    };
  }

  /** Verified finishes are narrated from the trace; answer-only finishes are returned as written. */
  private async answer(): Promise<Answer> {
    const t = this.t;
    if (t.status !== "finished") {
      return { text: `Stopped (${t.status}) after ${t.step} steps without a finished answer.`, citations: [], unknownCitations: [], grounded: false };
    }
    if (t.completion === "answer_only" || t.modelCalls.length >= this.h.limits.maxModelCalls) return fallbackAnswer(this.state, this.state.knownIds);
    this.narrator.transcript.reset(); // the narrator is stateless per turn: fresh snapshot + trace
    return narrate(this.llm, this.narrator, this.state, t.records, this.h.evidence.narratorRecordChars, t.signal);
  }
}
