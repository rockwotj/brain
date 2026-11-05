/**
 * Narrator: writes the user-facing answer from the recorded trace only, and the
 * runtime checks that every citation refers to something that actually happened.
 */
import { Agent, buildSystemPrompt } from "./agents.ts";
import type { LlmClient, ModelProfile, TokenEstimator } from "./llm.ts";
import { callText, clip, describeItem, type StateStore } from "./state.ts";
import type { ActionRecord } from "./types.ts";
import { parseNarration } from "./validate.ts";

/** `grounded` checks citation IDs only, not whether the cited evidence entails the prose. */
export type Answer = { text: string; citations: string[]; unknownCitations: string[]; grounded: boolean };

const NARRATOR_CONTRACT = `OUTPUT CONTRACT
Reply with exactly one JSON object: {"answer":"...","citations":["ids"]}
Use ONLY facts in the TRACE of the request. Cite trace identifiers in square brackets, e.g. [act-2].
If the trace does not establish something, say it is unknown or mark the sentence "(inferred)".`;

export function renderTrace(records: readonly ActionRecord[], state: StateStore): string {
  const actions = records.map((r) => {
    const out = r.actual.ok ? `ok: ${r.actual.output}` : `ERROR: ${r.actual.error}`;
    return `- ${r.id} ${callText(r.call)} -> ${clip(out, 300)}${r.surprise ? " (contradicted prediction)" : ""}`;
  });
  return [
    "TRACE",
    "Actions:",
    ...(actions.length ? actions : ["- (none)"]),
    "Workspace at finish:",
    ...state.workspace.map((i) => `- ${describeItem(i)}`),
    `Finish claim: ${state.done ? `${state.done.summary} [${state.done.evidence.join(", ")}]` : "(none)"}`,
  ].join("\n");
}

export async function narrate(
  llm: LlmClient,
  profile: ModelProfile,
  estimate: TokenEstimator,
  state: StateStore,
  records: readonly ActionRecord[],
  signal?: AbortSignal,
): Promise<Answer> {
  const prompt = buildSystemPrompt({
    name: "narrator",
    role: "the narrator that explains results to the user",
    instructions: "Explain the outcome for the goal in a few sentences, grounded in the trace.",
    tools: [],
    canFinish: false,
    contract: NARRATOR_CONTRACT,
  });
  const agent = new Agent("narrator", profile, prompt, { allowedTools: new Set(), allowActions: false, allowFinish: false, maxProposals: 0 }, estimate);
  const reply = await agent.call(llm, state, renderTrace(records, state), signal);
  const parsed = parseNarration(reply.text);
  // A model can put arbitrary strings in an observation's evidence list. Those
  // pointers do not create records: only actual trace entries count as known.
  const known = new Set([...records.map((r) => r.id), ...state.workspace.map((i) => i.id)]);
  if (!parsed.value) return fallbackAnswer(state, known);
  const inText = [...parsed.value.answer.matchAll(/\[([a-z]+-\d+)\]/g)].map((m) => m[1] ?? "");
  const citations = [...new Set([...parsed.value.citations, ...inText])];
  const unknownCitations = citations.filter((c) => !known.has(c));
  return { text: parsed.value.answer, citations, unknownCitations, grounded: citations.length > 0 && unknownCitations.length === 0 };
}

/** Deterministic answer when the narrator output is unusable: the finish claim, verbatim. */
export function fallbackAnswer(state: StateStore, known: ReadonlySet<string>): Answer {
  const evidence = state.done?.evidence ?? [];
  const text = state.done ? `${state.done.summary} [${evidence.join(", ")}]` : "No grounded answer was produced.";
  return { text, citations: evidence, unknownCitations: evidence.filter((e) => !known.has(e)), grounded: false };
}
