/**
 * Runtime validation at the LLM boundary. Model output is untrusted text:
 * nothing here casts parsed JSON to a type without checking it first.
 * Invalid entries are dropped and reported as issues, never silently trusted.
 */
import type { Confidence, Json, JsonObject, Proposal } from "./types.ts";

export type Parsed<T> = { value: T; issues: string[] };

export function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function isJson(v: unknown): v is Json {
  if (v === null || ["string", "number", "boolean"].includes(typeof v)) return true;
  if (Array.isArray(v)) return v.every(isJson);
  return isRecord(v) && Object.values(v).every(isJson);
}

export function isJsonObject(v: unknown): v is JsonObject {
  return isRecord(v) && Object.values(v).every(isJson);
}

function isStringArray(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === "string");
}

/** Parse a JSON object from model text, tolerating a surrounding ``` fence. */
export function parseJsonObject(text: string): Record<string, unknown> | string {
  const body = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/```$/, "").trim();
  try {
    const value: unknown = JSON.parse(body);
    return isRecord(value) ? value : "reply is not a JSON object";
  } catch {
    return "reply is not valid JSON";
  }
}

export type ProposalRules = {
  allowedTools: ReadonlySet<string>;
  allowActions: boolean;
  allowFinish: boolean;
  maxProposals: number;
};

const CONFIDENCE: readonly Confidence[] = ["low", "medium", "high"];
const OBS_TYPES = ["observation", "hypothesis", "focus"] as const;

export function parseProposals(text: string, rules: ProposalRules): Parsed<Proposal[]> {
  const obj = parseJsonObject(text);
  if (typeof obj === "string") return { value: [], issues: [obj] };
  if (!Array.isArray(obj.proposals)) return { value: [], issues: ["missing proposals array"] };
  const issues: string[] = [];
  const value: Proposal[] = [];
  for (const [i, raw] of obj.proposals.entries()) {
    if (value.length >= rules.maxProposals) {
      issues.push(`proposal ${i}: over the limit of ${rules.maxProposals}`);
      continue;
    }
    const result = parseProposal(raw, rules);
    if (typeof result === "string") issues.push(`proposal ${i}: ${result}`);
    else value.push(result);
  }
  return { value, issues };
}

function parseProposal(raw: unknown, rules: ProposalRules): Proposal | string {
  if (!isRecord(raw)) return "not an object";
  if (raw.kind === "observation") {
    const type = OBS_TYPES.find((t) => t === (raw.type ?? "observation"));
    const confidence = CONFIDENCE.find((c) => c === (raw.confidence ?? "medium"));
    if (!type) return "unknown observation type";
    if (!confidence) return "confidence must be low|medium|high";
    if (typeof raw.content !== "string" || !raw.content.trim()) return "observation needs content";
    const evidence = raw.evidence ?? [];
    if (!isStringArray(evidence)) return "evidence must be a string array";
    return { kind: "observation", type, content: raw.content, evidence, confidence };
  }
  if (raw.kind === "action") {
    if (!rules.allowActions) return "actions are not allowed for this agent";
    if (typeof raw.tool !== "string" || !rules.allowedTools.has(raw.tool)) return `unknown or unavailable tool ${String(raw.tool)}`;
    const args = raw.args ?? {};
    if (!isJsonObject(args)) return "args must be a JSON object";
    const expect = isRecord(raw.expect) ? raw.expect : {};
    const success = typeof expect.success === "boolean" ? expect.success : true;
    const contains = typeof expect.contains === "string" ? expect.contains : undefined;
    return {
      kind: "action",
      call: { tool: raw.tool, args },
      rationale: typeof raw.rationale === "string" ? raw.rationale : "",
      expect: contains ? { success, contains } : { success },
    };
  }
  if (raw.kind === "finish") {
    if (!rules.allowFinish) return "finish is not allowed for this agent";
    if (typeof raw.summary !== "string" || !raw.summary.trim()) return "finish needs a summary";
    if (!isStringArray(raw.evidence) || raw.evidence.length === 0) return "finish needs evidence ids";
    return { kind: "finish", summary: raw.summary, evidence: raw.evidence };
  }
  return `unknown kind ${String(raw.kind)}`;
}

export type Narration = { answer: string; citations: string[] };

export function parseNarration(text: string): Parsed<Narration | undefined> {
  const obj = parseJsonObject(text);
  if (typeof obj === "string") return { value: undefined, issues: [obj] };
  if (typeof obj.answer !== "string" || !isStringArray(obj.citations)) {
    return { value: undefined, issues: ["narration needs answer (string) and citations (string[])"] };
  }
  return { value: { answer: obj.answer, citations: obj.citations }, issues: [] };
}

export type Lesson = { statement: string; episodes: string[] };
export type HabitDraft = { trigger: string; tool: string; args: JsonObject; episodes: string[] };

export function parseConsolidation(text: string): Parsed<{ lessons: Lesson[]; habits: HabitDraft[] }> {
  const obj = parseJsonObject(text);
  if (typeof obj === "string") return { value: { lessons: [], habits: [] }, issues: [obj] };
  const issues: string[] = [];
  const lessons: Lesson[] = [];
  const habits: HabitDraft[] = [];
  for (const raw of Array.isArray(obj.lessons) ? obj.lessons : []) {
    if (isRecord(raw) && typeof raw.statement === "string" && isStringArray(raw.episodes)) {
      lessons.push({ statement: raw.statement, episodes: raw.episodes });
    } else issues.push("invalid lesson");
  }
  for (const raw of Array.isArray(obj.habits) ? obj.habits : []) {
    if (
      isRecord(raw) && typeof raw.trigger === "string" && typeof raw.tool === "string" &&
      isJsonObject(raw.args) && isStringArray(raw.episodes)
    ) {
      habits.push({ trigger: raw.trigger, tool: raw.tool, args: raw.args, episodes: raw.episodes });
    } else issues.push("invalid habit");
  }
  return { value: { lessons, habits }, issues };
}
