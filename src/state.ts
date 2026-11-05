/**
 * Authoritative run state. Every change is an append-only delta in the event
 * log; the in-memory view is just a fold over those deltas. Agents see the
 * same deltas as text, appended to their transcripts (see transcript.ts).
 */
import type { Constraint, Delta, HarnessEvent, Mode, Priority, WorkspaceItem } from "./types.ts";

export interface EventLog {
  append(source: string, delta: Delta): HarnessEvent;
  since(seq: number): HarnessEvent[];
  readonly head: number;
}

export class InMemoryEventLog implements EventLog {
  private readonly events: HarnessEvent[] = [];
  get head(): number {
    return this.events.length;
  }
  append(source: string, delta: Delta): HarnessEvent {
    const event = { seq: this.events.length + 1, source, delta };
    this.events.push(event);
    return event;
  }
  since(seq: number): HarnessEvent[] {
    return this.events.slice(seq);
  }
}

/** Configurable workspace budget. Pinned constraints and the goal do not count against it. */
export type WorkspaceBudget = { maxItems: number; maxItemChars: number };
export const DEFAULT_BUDGET: WorkspaceBudget = { maxItems: 6, maxItemChars: 280 };

const RANK: Record<Priority, number> = { low: 0, normal: 1, high: 2 };

export class StateStore {
  goal = "";
  done?: { summary: string; evidence: string[] };
  readonly constraints = new Map<string, Constraint>();
  workspace: WorkspaceItem[] = [];
  mode: Mode = { urgency: "normal", carefulSteps: 0 };
  private nextItem = 1;

  constructor(
    readonly log: EventLog = new InMemoryEventLog(),
    readonly budget: WorkspaceBudget = DEFAULT_BUDGET,
    private readonly listener?: (event: HarnessEvent) => void,
  ) {}

  get seq(): number {
    return this.log.head;
  }

  /** Pinned constraints as a fresh array (never a one-shot iterator). */
  pinned(): Constraint[] {
    return [...this.constraints.values()];
  }

  eventsSince(seq: number): HarnessEvent[] {
    return this.log.since(seq);
  }

  apply(source: string, delta: Delta): HarnessEvent {
    const event = this.log.append(source, delta);
    this.reduce(delta);
    this.listener?.(event);
    return event;
  }

  /**
   * Gate for workspace writes. Duplicates are ignored; when full, a new item must
   * be at least as important as the weakest (oldest lowest-priority) item to enter.
   */
  admit(source: string, draft: Omit<WorkspaceItem, "id" | "source">): WorkspaceItem | undefined {
    const content = clip(draft.content, this.budget.maxItemChars);
    if (this.workspace.some((i) => normalize(i.content) === normalize(content))) return undefined;
    if (this.workspace.length >= this.budget.maxItems) {
      const weakest = [...this.workspace].sort((a, b) => RANK[a.priority] - RANK[b.priority])[0];
      if (!weakest || RANK[weakest.priority] > RANK[draft.priority]) return undefined;
      this.apply(source, { type: "workspace.evict", id: weakest.id, reason: "workspace budget" });
    }
    const item: WorkspaceItem = { ...draft, content, source, id: `ws-${this.nextItem++}` };
    this.apply(source, { type: "workspace.admit", item });
    return item;
  }

  private reduce(delta: Delta): void {
    switch (delta.type) {
      case "goal.set":
        this.goal = delta.goal;
        break;
      case "goal.done":
        this.done = { summary: delta.summary, evidence: delta.evidence };
        break;
      case "constraint.add":
        this.constraints.set(delta.constraint.id, delta.constraint);
        break;
      case "constraint.remove":
        this.constraints.delete(delta.id);
        break;
      case "workspace.admit":
        this.workspace.push(delta.item);
        break;
      case "workspace.evict":
        this.workspace = this.workspace.filter((i) => i.id !== delta.id);
        break;
      case "mode.set":
        this.mode = delta.mode;
        break;
      case "action.outcome":
      case "action.blocked":
      case "note":
        break; // recorded in the log only
    }
  }

  /** Full authoritative view, used to (re)build an agent's prompt prefix. */
  renderSnapshot(): string {
    const constraints = this.pinned().map((c) => `- ${describeConstraint(c)}`);
    const items = this.workspace.map((i) => `- ${describeItem(i)}`);
    return [
      `STATE SNAPSHOT (authoritative as of seq ${this.seq})`,
      `GOAL: ${this.goal || "(none)"}`,
      "PINNED CONSTRAINTS (always apply):",
      ...(constraints.length ? constraints : ["- (none)"]),
      `WORKSPACE (${this.workspace.length}/${this.budget.maxItems}):`,
      ...(items.length ? items : ["- (empty)"]),
      `MODE: urgency=${this.mode.urgency} careful_steps=${this.mode.carefulSteps}`,
    ].join("\n");
  }
}

export function describeConstraint(c: Constraint): string {
  if (c.kind === "read_only") return `${c.id} READ-ONLY: no write tools (${c.reason})`;
  if (c.kind === "deny_tool") return `${c.id} DENY ${c.tool} (${c.reason})`;
  return `${c.id} RULE: ${c.text}`;
}

export function describeItem(i: WorkspaceItem): string {
  const ev = i.evidence.length ? ` [evidence: ${i.evidence.join(", ")}]` : "";
  return `${i.id} (${i.kind}, ${i.priority}, from ${i.source}) ${i.content}${ev}`;
}

/** One-line text form of an event, as broadcast to agents. */
export function describeEvent(e: HarnessEvent): string {
  const d = e.delta;
  const head = `#${e.seq} [${e.source}]`;
  switch (d.type) {
    case "goal.set":
      return `${head} goal set: ${d.goal}`;
    case "goal.done":
      return `${head} goal done: ${d.summary} [evidence: ${d.evidence.join(", ")}]`;
    case "constraint.add":
      return `${head} constraint added: ${describeConstraint(d.constraint)}`;
    case "constraint.remove":
      return `${head} constraint removed: ${d.id}`;
    case "workspace.admit":
      return `${head} workspace + ${describeItem(d.item)}`;
    case "workspace.evict":
      return `${head} workspace - ${d.id} (${d.reason})`;
    case "action.outcome": {
      const r = d.record;
      const out = r.actual.ok ? `ok: ${r.actual.output}` : `ERROR: ${r.actual.error}`;
      return `${head} ${r.id} ${callText(r.call)} -> ${clip(out, 200)}${r.surprise ? " (SURPRISE)" : ""}`;
    }
    case "action.blocked":
      return `${head} blocked ${callText(d.call)}: ${d.reason}`;
    case "mode.set":
      return `${head} mode: urgency=${d.mode.urgency} careful_steps=${d.mode.carefulSteps}`;
    case "note":
      return `${head} note: ${d.text}`;
  }
}

export function callText(call: { tool: string; args: unknown }): string {
  return `${call.tool}(${JSON.stringify(call.args)})`;
}

export function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

function normalize(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}
