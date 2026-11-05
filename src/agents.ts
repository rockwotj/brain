/**
 * Agents = (stable system prompt) + (own transcript) + (model profile).
 * They only produce validated proposals; they never touch tools directly.
 */
import { routeProfile, type LlmClient, type ModelProfile, type TokenEstimator, type Usage } from "./llm.ts";
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
  /** Router filter: which events wake this specialist. Defaults to any event from someone else. */
  wakeOn?: (event: HarnessEvent) => boolean;
};

export const OUTPUT_CONTRACT = `OUTPUT CONTRACT
Reply with exactly one JSON object and nothing else:
{"proposals": [ ...zero or more proposals... ]}
An empty list means "nothing useful to add" and is the expected default.
Proposal shapes:
{"kind":"observation","type":"observation|hypothesis|focus","content":"<= 200 chars","evidence":["ids"],"confidence":"low|medium|high"}
{"kind":"action","tool":"<tool name from TOOLS>","args":{...},"rationale":"why","expect":{"success":true,"contains":"optional text you expect in the output"}}
{"kind":"finish","summary":"answer to the goal","evidence":["act-ids that support it"]}
Rules: cite only identifiers that appear in the snapshot or updates. Proposals are suggestions;
the runtime decides what runs and enforces constraints and tool effects. Pinned constraints always apply.`;

export function buildSystemPrompt(p: { name: string; role: string; instructions: string; tools: readonly ToolSpec[]; canFinish: boolean; contract?: string }): string {
  return [
    `You are "${p.name}", ${p.role} inside an agent harness.`,
    `JOB: ${p.instructions}`,
    renderToolSchemas(p.tools),
    p.canFinish ? "You may propose finish once the evidence answers the goal." : "Do not propose finish.",
    p.contract ?? OUTPUT_CONTRACT,
  ].join("\n\n");
}

export type AgentReply = { text: string; usage?: Usage; failed?: string };

export class Agent {
  readonly transcript: AgentTranscript;
  constructor(
    readonly name: string,
    readonly profile: ModelProfile,
    systemPrompt: string,
    readonly rules: ProposalRules,
    estimate: TokenEstimator,
  ) {
    this.transcript = new AgentTranscript(name, systemPrompt, {
      contextWindow: profile.contextWindow,
      reserveOutput: profile.maxOutputTokens,
      estimate,
    });
  }

  /** One model turn on this agent's own append-only transcript. */
  async call(llm: LlmClient, state: StateStore, request: string, signal?: AbortSignal): Promise<AgentReply> {
    const turn = this.transcript.prepare(state, request);
    try {
      const res = await llm.complete({
        agent: this.name,
        profile: routeProfile(this.profile, state.mode.urgency),
        messages: turn.messages,
        stablePrefixLength: turn.stablePrefixLength,
        ...(signal ? { signal } : {}),
      });
      this.transcript.recordReply(res.text);
      return { text: res.text, usage: res.usage };
    } catch (error) {
      this.transcript.recordReply("[no reply: call failed]");
      if (signal?.aborted) throw error;
      return { text: "", failed: error instanceof Error ? error.message : String(error) };
    }
  }

  async propose(llm: LlmClient, state: StateStore, request: string, signal?: AbortSignal): Promise<{ proposals: Proposal[]; issues: string[] }> {
    const reply = await this.call(llm, state, request, signal);
    if (reply.failed) return { proposals: [], issues: [`model call failed: ${reply.failed}`] };
    const parsed = parseProposals(reply.text, this.rules);
    return { proposals: parsed.value, issues: parsed.issues };
  }
}

export function makeAgent(
  p: { name: string; role: string; instructions: string; toolNames: readonly string[]; canFinish: boolean; allowActions?: boolean },
  registry: readonly ToolSpec[],
  profile: ModelProfile,
  estimate: TokenEstimator,
  maxProposals = 4,
): Agent {
  const allowActions = p.allowActions ?? true;
  const tools = allowActions ? registry.filter((t) => p.toolNames.includes(t.name)) : [];
  const prompt = buildSystemPrompt({ name: p.name, role: p.role, instructions: p.instructions, tools, canFinish: p.canFinish });
  return new Agent(p.name, profile, prompt, {
    allowedTools: new Set(tools.map((t) => t.name)),
    allowActions,
    allowFinish: p.canFinish,
    maxProposals,
  }, estimate);
}
