/**
 * Provider-neutral port for a *decision model*: a fast classifier that reads a
 * state and answers predefined questions with typed answers and probabilities.
 * It never writes text, plans or tool calls. (TypeSafe calls this kind of model
 * a "System One" model; no adapter for it ships here.)
 *
 * The harness uses decisions only for soft routing (which specialists to wake,
 * whether to consult the executive). Hard policy never depends on them.
 */

export type DecisionQuestion =
  /** Yes/no. The answer is the probability of "yes". */
  | { type: "boolean"; instructions: string; criteria?: { true: string; false: string } }
  /** One of a fixed set of options (option name -> description). */
  | { type: "choice"; instructions: string; criteria: Readonly<Record<string, string>> }
  /** A position on an ordered scale; the answer is a number from 0 to levels.length - 1. */
  | { type: "score"; instructions: string; levels: readonly string[] };

export type DecisionAnswer =
  | { type: "boolean"; probability: number }
  | { type: "choice"; choice: string; probabilities?: Readonly<Record<string, number>> }
  | { type: "score"; score: number };

export type DecisionRequest = {
  /** What the harness is deciding (for logging), e.g. "route" or "escalate". */
  purpose: string;
  /** The situation to judge, as text. */
  state: string;
  /** Question id -> question. */
  questions: Readonly<Record<string, DecisionQuestion>>;
  signal?: AbortSignal;
};

export interface DecisionModel {
  readonly name: string;
  /** Answers keyed by question id. Output is checked by `checkDecisions` before use. */
  decide(request: DecisionRequest): Promise<Readonly<Record<string, unknown>>>;
}

/** Keep only well-formed answers that match their question; report the rest. */
export function checkDecisions(
  questions: Readonly<Record<string, DecisionQuestion>>,
  raw: Readonly<Record<string, unknown>> | undefined,
): { answers: Record<string, DecisionAnswer>; issues: string[] } {
  const answers: Record<string, DecisionAnswer> = {};
  const issues: string[] = [];
  for (const [id, q] of Object.entries(questions)) {
    const a = raw?.[id];
    const problem = checkAnswer(q, a);
    if (problem) issues.push(`${id}: ${problem}`);
    else answers[id] = a as DecisionAnswer;
  }
  return { answers, issues };
}

function checkAnswer(q: DecisionQuestion, a: unknown): string | undefined {
  if (!a || typeof a !== "object") return "missing answer";
  const v = a as Record<string, unknown>;
  if (v.type !== q.type) return `expected a ${q.type} answer, got ${String(v.type)}`;
  if (q.type === "boolean") return isProbability(v.probability) ? undefined : "probability must be a number in [0, 1]";
  if (q.type === "score") {
    const max = q.levels.length - 1;
    return typeof v.score === "number" && v.score >= 0 && v.score <= max ? undefined : `score must be a number in [0, ${max}]`;
  }
  if (typeof v.choice !== "string" || !(v.choice in q.criteria)) return `choice must be one of ${Object.keys(q.criteria).join(", ")}`;
  if (v.probabilities !== undefined) {
    const p = v.probabilities as Record<string, unknown>;
    if (!p || typeof p !== "object" || !Object.entries(p).every(([k, x]) => k in q.criteria && isProbability(x))) return "probabilities must map options to [0, 1]";
  }
  return undefined;
}

function isProbability(x: unknown): x is number {
  return typeof x === "number" && x >= 0 && x <= 1;
}
