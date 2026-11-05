/**
 * Agents = (stable system prompt) + (own transcript) + (model profile).
 * They only produce validated proposals; they never touch tools directly.
 *
 * System prompt order is chosen for prompt caching: a SHARED block first
 * (rules, sorted tool list, reply schema; identical for every agent of a
 * harness), then the agent's ROLE block. Agents on the same model therefore
 * share the leading part of their prompts.
 */
import { UnsupportedContentError, withUrgency, type LlmClient, type ModelProfile, type TokenEstimator, type Usage } from "./io/llm.ts";
import type { StateStore } from "./state.ts";
import { renderToolSchemas, type ToolSpec } from "./tools.ts";
import { AgentTranscript } from "./transcript.ts";
import type { HarnessEvent, Proposal } from "./types.ts";
import { parseProposals, type ProposalRules } from "./validate.ts";

export type SpecialistSpec = {
  name: string;
  /** Narrow job description. Keep it short; silence is the default. */
  instructions: string;
  /** Tools this specialist may propose (must exist in the registry). */
  tools: string[];
  canFinish?: boolean;
  /** Router filter: which visible events wake this specialist. Defaults to state changes and outcomes. */
  wakeOn?: (event: HarnessEvent) => boolean;
};

export const OUTPUT_CONTRACT = `OUTPUT CONTRACT
Reply with exactly one JSON object and nothing else:
{"proposals": [ ...zero or more proposals... ]}
An empty list means "nothing useful to add" and is the expected default.
Proposal shapes:
{"kind":"observation","type":"observation|hypothesis|focus","content":"<= 200 chars","evidence":["ids"],"confidence":"low|medium|high"}
{"kind":"action","tool":"<a tool your ROLE allows>","args":{...},"rationale":"why","expect":{"success":true,"contains":"optional text you expect in the output"}}
{"kind":"read_record","id":"act-3","offset":0}  (full output of a record; delivered only to you on your next turn)
{"kind":"finish","summary":"answer to the goal","evidence":["ids that support it"]}
A finish citing a successful act-id is verified. A finish with "evidence": [] is an answer-only reply:
fine when the goal needs no tools (e.g. a direct question), but it loses to any proposed action.
Rules: cite only identifiers that appear in the snapshot, updates or private notes. Proposals are suggestions;
the runtime decides what runs and enforces constraints and tool effects. Pinned constraints always apply.
Outcomes you expected are sent only to you: share what matters with an observation.`;

/** Identical for every agent built from the same tool registry. */
export function buildSharedPrompt(registry: readonly ToolSpec[]): string {
  return [
    "You are one agent inside a multi-agent harness. Several agents work on the same goal; the runtime arbitrates their proposals.",
    renderToolSchemas(registry),
    OUTPUT_CONTRACT,
  ].join("\n\n");
}

export function buildRolePrompt(p: { name: string; role: string; instructions: string; toolNames: readonly string[]; canFinish: boolean }): string {
  return [
    "ROLE",
    `You are "${p.name}", ${p.role}.`,
    `JOB: ${p.instructions}`,
    p.toolNames.length ? `You may propose only these tools: ${[...p.toolNames].sort().join(", ")}.` : "You may not propose actions.",
    p.canFinish ? "You may propose finish." : "Do not propose finish.",
  ].join("\n");
}

export type AgentReply = { text: string; usage?: Usage; failed?: string };

export class Agent {
  readonly transcript: AgentTranscript;
  /** Private notes (e.g. requested records) delivered with this agent's next turn. */
  readonly inbox: string[] = [];

  constructor(
    readonly name: string,
    readonly profile: ModelProfile,
    systemPrompt: string,
    readonly rules: ProposalRules,
    estimate: TokenEstimator,
    clock?: () => number,
  ) {
    this.transcript = new AgentTranscript(name, systemPrompt, {
      contextWindow: profile.contextWindow,
      reserveOutput: profile.maxOutputTokens,
      estimate,
      vision: !!profile.vision,
      ...(clock ? { clock } : {}),
    });
  }

  /**
   * One model turn on this agent's own append-only transcript. Transient model
   * failures become an empty reply; configuration errors (media the model
   * cannot take, context too small) are thrown.
   */
  async call(llm: LlmClient, state: StateStore, request: string, signal?: AbortSignal): Promise<AgentReply> {
    const turn = this.transcript.prepare(state, request, this.inbox);
    this.inbox.length = 0; // delivered (prepare throws without side effects)
    try {
      const res = await llm.complete({
        agent: this.name,
        profile: withUrgency(this.profile, state.mode.urgency),
        messages: turn.messages,
        stablePrefixLength: turn.stablePrefixLength,
        ...(signal ? { signal } : {}),
      });
      this.transcript.recordReply(res.text);
      this.transcript.noteCacheExpiry(res.cacheExpiresAt);
      return { text: res.text, usage: res.usage };
    } catch (error) {
      this.transcript.recordReply("[no reply: call failed]");
      if (signal?.aborted || error instanceof UnsupportedContentError) throw error;
      return { text: "", failed: error instanceof Error ? error.message : String(error) };
    }
  }

  async propose(llm: LlmClient, state: StateStore, request: string, signal?: AbortSignal): Promise<{ proposals: Proposal[]; issues: string[]; failed?: boolean }> {
    const reply = await this.call(llm, state, request, signal);
    if (reply.failed) return { proposals: [], issues: [`model call failed: ${reply.failed}`], failed: true };
    const parsed = parseProposals(reply.text, this.rules);
    return { proposals: parsed.value, issues: parsed.issues };
  }
}

export function makeAgent(
  p: { name: string; role: string; instructions: string; toolNames: readonly string[]; canFinish: boolean; allowActions?: boolean },
  registry: readonly ToolSpec[],
  profile: ModelProfile,
  estimate: TokenEstimator,
  options: { maxProposals?: number; clock?: () => number } = {},
): Agent {
  const allowActions = p.allowActions ?? true;
  const toolNames = allowActions ? registry.filter((t) => p.toolNames.includes(t.name)).map((t) => t.name) : [];
  const prompt = `${buildSharedPrompt(registry)}\n\n${buildRolePrompt({ ...p, toolNames })}`;
  return new Agent(p.name, profile, prompt, {
    allowedTools: new Set(toolNames),
    allowActions,
    allowFinish: p.canFinish,
    maxProposals: options.maxProposals ?? 4,
  }, estimate, options.clock);
}
