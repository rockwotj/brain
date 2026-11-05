/**
 * Narrator: writes the user-facing answer from the recorded trace only, and the
 * runtime checks that every citation refers to something that actually happened.
 * Used for verified finishes; answer-only finishes are returned as written.
 */
import { Agent } from "./agents.ts";
import type { LlmClient, ModelProfile, TokenEstimator } from "./io/llm.ts";
import { callText, describeItem, outcomeSummary, type StateStore } from "./state.ts";
import type { ActionRecord } from "./types.ts";
import { parseNarration } from "./validate.ts";

/** `grounded` checks citation IDs only, not whether the cited evidence entails the prose. */
export type Answer = { text: string; citations: string[]; unknownCitations: string[]; grounded: boolean };

const NARRATOR_PROMPT = `You are "narrator": you explain the outcome of the current goal to the user in a few sentences.
OUTPUT CONTRACT
Reply with exactly one JSON object: {"answer":"...","citations":["ids"]}
Use ONLY facts in the TRACE of the request. Cite trace identifiers in square brackets, e.g. [act-2].
If the trace does not establish something, say it is unknown or mark the sentence "(inferred)".`;

export function makeNarrator(profile: ModelProfile, estimate: TokenEstimator, clock?: () => number): Agent {
  return new Agent("narrator", profile, NARRATOR_PROMPT, { allowedTools: new Set(), allowActions: false, allowFinish: false, maxProposals: 0 }, estimate, clock);
}

/** The trace shows each output up to `recordChars` characters (not the short broadcast preview). */
export function renderTrace(records: readonly ActionRecord[], state: StateStore, recordChars: number): string {
  return [
    "TRACE",
    "Actions this turn:",
    ...(records.length ? records.map((r) => `- ${r.id} ${callText(r.call)} -> ${outcomeSummary(r, recordChars)}${r.surprise ? " (contradicted prediction)" : ""}`) : ["- (none)"]),
    "Workspace at finish:",
    ...state.workspace.map((i) => `- ${describeItem(i)}`),
    `Finish claim: ${state.done ? `${state.done.summary} [${state.done.evidence.join(", ")}]` : "(none)"}`,
  ].join("\n");
}

export async function narrate(
  llm: LlmClient,
  narrator: Agent,
  state: StateStore,
  records: readonly ActionRecord[],
  recordChars: number,
  signal?: AbortSignal,
): Promise<Answer> {
  const reply = await narrator.call(llm, state, renderTrace(records, state, recordChars), signal);
  const parsed = parseNarration(reply.text);
  // A model can put arbitrary strings in an observation's evidence list. Those
  // pointers do not create records: only actual trace entries count as known.
  const known = new Set([...state.records.keys(), ...state.workspace.map((i) => i.id)]);
  if (!parsed.value) return fallbackAnswer(state, known);
  const inText = [...parsed.value.answer.matchAll(/\[([a-z]+-\d+)\]/g)].map((m) => m[1] ?? "");
  const citations = [...new Set([...parsed.value.citations, ...inText])];
  const unknownCitations = citations.filter((c) => !known.has(c));
  return { text: parsed.value.answer, citations, unknownCitations, grounded: citations.length > 0 && unknownCitations.length === 0 };
}

/** Deterministic answer when the narrator output is unusable: the finish claim, verbatim. */
export function fallbackAnswer(state: StateStore, known: ReadonlySet<string>): Answer {
  const evidence = state.done?.evidence ?? [];
  const text = state.done ? `${state.done.summary}${evidence.length ? ` [${evidence.join(", ")}]` : ""}` : "No grounded answer was produced.";
  return { text, citations: evidence, unknownCitations: evidence.filter((e) => !known.has(e)), grounded: false };
}
