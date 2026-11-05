/**
 * Authoritative session state. Every change is an append-only delta in an
 * event log on the Storage port; the in-memory view is just a fold over those
 * deltas, so a session can be reopened from storage. Agents see the same
 * deltas as text, appended to their transcripts (see transcript.ts).
 */
import { describeMedia, type MediaPart } from "./io/content.ts";
import { MemoryStorage, type Storage } from "./io/storage.ts";
import type { ActionRecord, Constraint, Delta, HarnessEvent, Mode, PerceptionRecord, Priority, WorkspaceItem } from "./types.ts";

/** Configurable workspace budget. Pinned constraints and the goal do not count against it. */
export type WorkspaceBudget = { maxItems: number; maxItemChars: number };
export const DEFAULT_BUDGET: WorkspaceBudget = { maxItems: 6, maxItemChars: 280 };

export type StateOptions = {
  storage?: Storage;
  /** Log name for this session's events. */
  log?: string;
  budget?: WorkspaceBudget;
  /** Characters of a tool output shown inline; the rest is available via read_record. */
  previewChars?: number;
  /** Most recent action records listed in a snapshot's evidence index. */
  indexSize?: number;
};

const RANK: Record<Priority, number> = { low: 0, normal: 1, high: 2 };

export class StateStore {
  goal = "";
  attachments: MediaPart[] = [];
  done?: { summary: string; evidence: string[] };
  readonly constraints = new Map<string, Constraint>();
  workspace: WorkspaceItem[] = [];
  mode: Mode = { urgency: "normal", carefulSteps: 0 };
  /** Every action record of the session, full output included, by id. */
  readonly records = new Map<string, ActionRecord>();
  /** Collected LiveModel observations per audio/video input, by id. */
  readonly perceptions = new Map<string, PerceptionRecord>();
  /** Identifiers that may be cited: records, workspace items (even evicted) and recalled memory. */
  readonly knownIds = new Set<string>();
  /** Called after each applied event (used for traces). */
  onEvent?: (event: HarnessEvent) => void;

  readonly budget: WorkspaceBudget;
  readonly previewChars: number;
  private readonly storage: Storage;
  private readonly log: string;
  private readonly indexSize: number;
  private readonly events: HarnessEvent[] = [];
  private admitted = 0;
  private writes: Promise<void> = Promise.resolve();

  constructor(opts: StateOptions = {}) {
    this.storage = opts.storage ?? new MemoryStorage();
    this.log = opts.log ?? "events";
    this.budget = opts.budget ?? DEFAULT_BUDGET;
    this.previewChars = opts.previewChars ?? 240;
    this.indexSize = opts.indexSize ?? 30;
  }

  /** Rebuild state by replaying the stored event log. */
  static async open(opts: StateOptions = {}): Promise<StateStore> {
    const state = new StateStore(opts);
    for (const event of await state.storage.read<HarnessEvent>(state.log)) {
      state.events.push(event);
      state.reduce(event.delta);
    }
    return state;
  }

  get seq(): number {
    return this.events.length;
  }

  /** Pinned constraints as a fresh array (never a one-shot iterator). */
  pinned(): Constraint[] {
    return [...this.constraints.values()];
  }

  eventsSince(seq: number): HarnessEvent[] {
    return this.events.slice(seq);
  }

  nextRecordId(): string {
    return `act-${this.records.size + 1}`;
  }

  nextPerceptionId(): string {
    return `per-${this.perceptions.size + 1}`;
  }

  /** Full text of an action or perception record, for read_record. */
  findEvidence(id: string): { label: string; text: string } | undefined {
    const r = this.records.get(id);
    if (r) return { label: `${callText(r.call)} ${r.actual.ok ? "ok" : "ERROR"}`, text: recordOutput(r) };
    const p = this.perceptions.get(id);
    if (p) return { label: `${p.medium} perception from ${p.source}`, text: perceptionText(p) };
    return undefined;
  }

  /**
   * Append a delta to the log (in order) and fold it into the view.
   * PoC failure behavior: the in-memory view is updated before the storage
   * write completes; if the write fails, this throws (ending the turn) and the
   * stored log lacks the event, so a reopened session will not see it.
   */
  async apply(source: string, delta: Delta, audience?: string[]): Promise<HarnessEvent> {
    const event: HarnessEvent = { seq: this.events.length + 1, source, delta, ...(audience ? { audience } : {}) };
    this.events.push(event);
    this.reduce(delta);
    const write = this.writes.then(() => this.storage.append(this.log, event)); // keep log order
    this.writes = write.catch(() => undefined);
    await write;
    this.onEvent?.(event);
    return event;
  }

  /**
   * Gate for workspace writes. Duplicates are ignored; when full, a new item must
   * be at least as important as the weakest (oldest lowest-priority) item to enter.
   */
  async admit(source: string, draft: Omit<WorkspaceItem, "id" | "source">): Promise<WorkspaceItem | undefined> {
    const content = clip(draft.content, this.budget.maxItemChars);
    if (this.workspace.some((i) => normalize(i.content) === normalize(content))) return undefined;
    if (this.workspace.length >= this.budget.maxItems) {
      const weakest = [...this.workspace].sort((a, b) => RANK[a.priority] - RANK[b.priority])[0];
      if (!weakest || RANK[weakest.priority] > RANK[draft.priority]) return undefined;
      await this.apply(source, { type: "workspace.evict", id: weakest.id, reason: "workspace budget" });
    }
    const item: WorkspaceItem = { ...draft, content, source, id: `ws-${this.admitted + 1}` };
    await this.apply(source, { type: "workspace.admit", item });
    return item;
  }

  private reduce(delta: Delta): void {
    switch (delta.type) {
      case "goal.set":
        this.goal = delta.goal;
        this.attachments = delta.attachments ?? [];
        this.done = undefined;
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
        this.admitted++;
        this.workspace.push(delta.item);
        this.knownIds.add(delta.item.id);
        if (delta.item.kind === "memory") for (const id of delta.item.evidence) this.knownIds.add(id);
        break;
      case "workspace.evict":
        this.workspace = this.workspace.filter((i) => i.id !== delta.id);
        break;
      case "mode.set":
        this.mode = delta.mode;
        break;
      case "action.outcome":
        this.records.set(delta.record.id, delta.record);
        this.knownIds.add(delta.record.id);
        break;
      case "perception":
        this.perceptions.set(delta.record.id, delta.record);
        this.knownIds.add(delta.record.id);
        break;
      case "action.blocked":
      case "note":
        break; // recorded in the log only
    }
  }

  /** Full authoritative view, used to (re)build an agent's prompt prefix. */
  renderSnapshot(): string {
    const constraints = this.pinned().map((c) => `- ${describeConstraint(c)}`);
    const items = this.workspace.map((i) => `- ${describeItem(i)}`);
    const attached = this.attachments.length ? ` [attached: ${this.attachments.map(describeMedia).join(", ")}]` : "";
    return [
      `STATE SNAPSHOT (authoritative as of seq ${this.seq})`,
      `GOAL: ${this.goal || "(none)"}${attached}`,
      "PINNED CONSTRAINTS (always apply):",
      ...(constraints.length ? constraints : ["- (none)"]),
      `WORKSPACE (${this.workspace.length}/${this.budget.maxItems}):`,
      ...(items.length ? items : ["- (empty)"]),
      ...this.renderEvidenceIndex(),
      `MODE: urgency=${this.mode.urgency} careful_steps=${this.mode.carefulSteps}`,
    ].join("\n");
  }

  /**
   * Compact list of all records. Survives compaction; full content via read_record.
   * Deliberately global: agents can see that a record routed to someone else exists
   * (id, call, short preview) and read it if they need it.
   */
  private renderEvidenceIndex(): string[] {
    const perceptions = [...this.perceptions.values()].map((p) => `- ${p.id} ${perceptionSummary(p, 80)}`);
    const all = [...this.records.values()];
    if (!all.length && !perceptions.length) return [];
    const shown = all.slice(-this.indexSize);
    const older = all.length - shown.length;
    return [
      "EVIDENCE INDEX (propose read_record with an id to see the full output):",
      ...perceptions,
      ...(older ? [`- (${older} older records: ${all[0]?.id} .. ${all[older - 1]?.id})`] : []),
      ...shown.map((r) => `- ${r.id} ${callText(r.call)} ${outcomeSummary(r, 80)}${r.surprise ? " (SURPRISE)" : ""}`),
    ];
  }
}

export function visibleTo(event: HarnessEvent, agent: string): boolean {
  return !event.audience || event.audience.includes(agent);
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

export function recordOutput(r: ActionRecord): string {
  return r.actual.ok ? r.actual.output : r.actual.error;
}

export function perceptionText(p: PerceptionRecord): string {
  return p.observations.map((o) => `- ${o}`).join("\n");
}

/** "audio (audio/wav) via live:x, 3 observations (120 chars): preview…" */
export function perceptionSummary(p: PerceptionRecord, previewChars: number): string {
  const full = perceptionText(p);
  const head = `${p.medium} (${p.mimeType}) via ${p.source}, ${p.observations.length} observations (${full.length} chars)`;
  return full.length <= previewChars ? `${head}: ${full}` : `${head}: ${clip(full, previewChars)} [cut; read_record ${p.id}]`;
}

/** "ok (1003 chars): preview…" with a pointer to read_record when the preview is cut. */
export function outcomeSummary(r: ActionRecord, previewChars: number): string {
  const full = recordOutput(r);
  const head = `${r.actual.ok ? "ok" : "ERROR"} (${full.length} chars)`;
  return full.length <= previewChars ? `${head}: ${full}` : `${head}: ${clip(full, previewChars)} [cut; read_record ${r.id}]`;
}

/** One-line text form of an event, as shown to agents. */
export function describeEvent(e: HarnessEvent, previewChars: number): string {
  const d = e.delta;
  const head = `#${e.seq} [${e.source}]${e.audience ? ` (to ${e.audience.join(", ")} only)` : ""}`;
  switch (d.type) {
    case "goal.set": {
      const attached = d.attachments?.length ? ` [attached: ${d.attachments.map(describeMedia).join(", ")}]` : "";
      return `${head} goal set: ${d.goal}${attached}`;
    }
    case "goal.done":
      return `${head} goal done (${d.completion}): ${d.summary} [evidence: ${d.evidence.join(", ") || "none"}]`;
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
      return `${head} ${r.id} ${callText(r.call)} -> ${outcomeSummary(r, previewChars)}${r.surprise ? " (SURPRISE)" : ""}`;
    }
    case "action.blocked":
      return `${head} blocked ${callText(d.call)}: ${d.reason}`;
    case "perception":
      return `${head} ${d.record.id} ${perceptionSummary(d.record, previewChars)}`;
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
