/** Shared domain types. Plain data only: no behaviour lives here. */

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };
export type JsonObject = { [key: string]: Json };

/** Ordinal labels instead of uncalibrated floating-point scores. */
export type Priority = "low" | "normal" | "high";
export type Confidence = "low" | "medium" | "high";

/** Side-effect class of a tool. Set by the trusted tool registry, never by a model. */
export type Effect = "read" | "write";

/** Pinned rules. They live outside the workspace budget and are never evicted. */
export type Constraint =
  | { id: string; kind: "read_only"; reason: string }
  | { id: string; kind: "deny_tool"; tool: string; reason: string }
  | { id: string; kind: "instruction"; text: string };

export type WorkspaceKind = "observation" | "hypothesis" | "focus" | "outcome" | "memory" | "idea";

export type WorkspaceItem = {
  id: string;
  kind: WorkspaceKind;
  content: string;
  source: string;
  priority: Priority;
  /** Identifiers (act-*, ws-*, kn-*, ep-*) that support this item. */
  evidence: string[];
};

export type Mode = {
  /** Raised when a user is waiting; may switch profiles to a provider "fast" service mode. */
  urgency: "normal" | "high";
  /** Steps of post-error caution remaining: writes then need executive backing. */
  carefulSteps: number;
};

/** What the proposer expects to happen. Used to detect surprise. */
export type Prediction = { success: boolean; contains?: string };

export type ToolCall = { tool: string; args: JsonObject };

export type ObservationProposal = {
  kind: "observation";
  type: "observation" | "hypothesis" | "focus";
  content: string;
  evidence: string[];
  confidence: Confidence;
};
export type ActionProposal = { kind: "action"; call: ToolCall; rationale: string; expect: Prediction };
export type FinishProposal = { kind: "finish"; summary: string; evidence: string[] };
export type Proposal = ObservationProposal | ActionProposal | FinishProposal;

/** One executed (or attempted) tool call with its prediction and actual outcome. */
export type ActionRecord = {
  id: string;
  step: number;
  call: ToolCall;
  source: string;
  predicted: Prediction;
  actual: { ok: true; output: string } | { ok: false; error: string };
  surprise: boolean;
};

export type Delta =
  | { type: "goal.set"; goal: string }
  | { type: "goal.done"; summary: string; evidence: string[] }
  | { type: "constraint.add"; constraint: Constraint }
  | { type: "constraint.remove"; id: string }
  | { type: "workspace.admit"; item: WorkspaceItem }
  | { type: "workspace.evict"; id: string; reason: string }
  | { type: "action.outcome"; record: ActionRecord }
  | { type: "action.blocked"; call: ToolCall; reason: string }
  | { type: "mode.set"; mode: Mode }
  | { type: "note"; text: string };

/** A delta stamped with its position in the append-only event log. */
export type HarnessEvent = { seq: number; source: string; delta: Delta };

export type Episode = {
  id: string;
  goal: string;
  status: RunStatus;
  actions: ActionRecord[];
  answer?: string;
  consolidated: boolean;
};

export type RunStatus = "finished" | "no_progress" | "budget_exhausted" | "max_steps" | "cancelled";

export type KnowledgeItem = {
  id: string;
  statement: string;
  /** Episodes that support the statement (provenance). Never empty. */
  sources: string[];
  version: number;
};

export type Habit = {
  id: string;
  /** Fires when the goal contains this phrase (case-insensitive). */
  trigger: string;
  call: ToolCall;
  expect: Prediction;
  status: "pending" | "approved" | "suspended";
  successes: number;
  failures: number;
  sources: string[];
};
