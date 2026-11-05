/**
 * Idle-time "background mind". Bounded, low-priority reflection that can only
 * produce ideas: it is never given a ToolPort, and action proposals are
 * rejected at the validation boundary. Ideas can be passed to a later turn as
 * `preparedIdeas`, where they enter the workspace at low priority.
 */
import { makeAgent } from "./agents.ts";
import { roughTokenEstimate, type LlmClient, type ModelProfile, type TokenEstimator } from "./io/llm.ts";
import type { EpisodicStore, KnowledgeStore } from "./memory.ts";
import { StateStore } from "./state.ts";

export type IdleInput = {
  llm: LlmClient;
  profile: ModelProfile;
  knowledge: KnowledgeStore;
  episodes: EpisodicStore;
  maxIdeas?: number;
  estimate?: TokenEstimator;
  signal?: AbortSignal;
};

export type IdleReport = { ideas: string[]; discarded: string[] };

export async function idle(input: IdleInput): Promise<IdleReport> {
  const state = new StateStore(); // scratch state, in memory only
  await state.apply("background", { type: "goal.set", goal: "Idle reflection: prepare ideas that could help with likely future requests." });
  for (const k of (await input.knowledge.list()).slice(-3)) {
    await state.admit("background", { kind: "memory", content: k.statement, priority: "normal", evidence: [k.id, ...k.sources] });
  }
  for (const e of (await input.episodes.list()).slice(-2)) {
    await state.admit("background", { kind: "memory", content: `${e.id} (${e.status}): ${e.goal}`, priority: "low", evidence: [e.id] });
  }
  const agent = makeAgent(
    { name: "background", role: "an idle-time reflector", instructions: "Suggest at most two short ideas worth checking next time. You cannot act.", toolNames: [], canFinish: false, allowActions: false },
    [], input.profile, input.estimate ?? roughTokenEstimate);
  const reply = await agent.propose(input.llm, state, "Offer ideas as observations, or nothing.", input.signal);
  const maxIdeas = input.maxIdeas ?? 2;
  const observations = reply.proposals.flatMap((p) => (p.kind === "observation" ? [p.content] : []));
  return {
    ideas: observations.slice(0, maxIdeas),
    discarded: [...reply.issues, ...observations.slice(maxIdeas).map((o) => `over idea limit: ${o}`)],
  };
}
