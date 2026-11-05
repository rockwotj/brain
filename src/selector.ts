/**
 * Action selector: many candidates in, at most one released per step.
 * Deterministic lexicographic ranking with a plain-English explanation,
 * after hard policy filtering. No weighted float scores.
 */
import type { ActionProposal, Constraint, FinishProposal, JsonObject, ToolCall } from "./types.ts";
import type { Actuator } from "./tools.ts";

export type Origin = "executive" | "habit" | "specialist";
export type Supporter = { agent: string; profileId: string };

export type Candidate = {
  key: string;
  proposal: ActionProposal | FinishProposal;
  origin: Origin;
  supporters: Supporter[];
  habitId?: string;
};

export type Selection =
  | { kind: "act"; candidate: Candidate; why: string }
  | { kind: "finish"; candidate: Candidate; why: string }
  | { kind: "hold"; why: string }
  | { kind: "none"; why: string };

export type SelectContext = {
  actuator: Actuator;
  constraints: readonly Constraint[];
  executedKeys: ReadonlySet<string>;
  /** Ids of successful action records; a finish must cite at least one. */
  successfulActions: ReadonlySet<string>;
  careful: boolean;
};

export type SelectResult = { selection: Selection; blocked: { call: ToolCall; reason: string }[]; rejected: string[] };

const ORIGIN_RANK: Record<Origin, number> = { executive: 2, habit: 1, specialist: 0 };

export function candidateKey(p: ActionProposal | FinishProposal): string {
  return p.kind === "finish" ? "finish" : `${p.call.tool}:${canonical(p.call.args)}`;
}

/** Merge a proposal into the pool; identical proposals become one candidate with several supporters. */
export function addCandidate(pool: Candidate[], proposal: ActionProposal | FinishProposal, origin: Origin, by: Supporter, habitId?: string): void {
  const key = candidateKey(proposal);
  const existing = pool.find((c) => c.key === key);
  if (!existing) {
    pool.push({ key, proposal, origin, supporters: [by], ...(habitId ? { habitId } : {}) });
    return;
  }
  existing.supporters.push(by);
  if (ORIGIN_RANK[origin] > ORIGIN_RANK[existing.origin]) Object.assign(existing, { origin, proposal });
}

/**
 * Agreement only counts once per distinct model profile: several prompts on the
 * same model are correlated samples, not independent evidence.
 */
export function independentSupport(c: Candidate): number {
  return new Set(c.supporters.map((s) => s.profileId)).size;
}

export function select(pool: readonly Candidate[], ctx: SelectContext): SelectResult {
  const blocked: SelectResult["blocked"] = [];
  const rejected: string[] = [];
  const actions: { c: Candidate; rank: number[]; effect: string }[] = [];

  for (const c of pool) {
    const p = c.proposal;
    if (p.kind === "finish") {
      if (p.evidence.some((id) => ctx.successfulActions.has(id))) {
        return { selection: { kind: "finish", candidate: c, why: `finish backed by ${p.evidence.join(", ")}` }, blocked, rejected };
      }
      rejected.push("finish rejected: cites no successful action");
      continue;
    }
    if (ctx.executedKeys.has(c.key)) {
      rejected.push(`${p.call.tool} rejected: identical call already executed`);
      continue;
    }
    const decision = ctx.actuator.check(p.call, ctx.constraints);
    if (!decision.allowed) {
      blocked.push({ call: p.call, reason: decision.reason });
      continue;
    }
    const effect = decision.spec.effect;
    if (ctx.careful && effect === "write" && c.origin !== "executive") {
      rejected.push(`${p.call.tool} held: careful mode requires executive backing for writes`);
      continue;
    }
    actions.push({ c, effect, rank: [ORIGIN_RANK[c.origin], effect === "read" ? 1 : 0, independentSupport(c)] });
  }

  if (!actions.length) return { selection: { kind: "none", why: "no eligible candidates" }, blocked, rejected };
  actions.sort((a, b) => compare(b.rank, a.rank)); // stable: ties keep proposal order
  const [best, second] = actions;
  if (!best) return { selection: { kind: "none", why: "no eligible candidates" }, blocked, rejected };
  if (second && compare(best.rank, second.rank) === 0 && (best.effect === "write" || second.effect === "write")) {
    return {
      selection: { kind: "hold", why: `conflict: ${best.c.key} vs ${second.c.key} tie and at least one writes` },
      blocked,
      rejected,
    };
  }
  const n = independentSupport(best.c);
  const why =
    `origin=${best.c.origin}, effect=${best.effect}, independent support=${n}` +
    ` (${best.c.supporters.length} proposals)` +
    (second && compare(best.rank, second.rank) === 0 ? "; tie broken by proposal order" : "");
  return { selection: { kind: "act", candidate: best.c, why }, blocked, rejected };
}

function compare(a: number[], b: number[]): number {
  for (let i = 0; i < a.length; i++) if ((a[i] ?? 0) !== (b[i] ?? 0)) return (a[i] ?? 0) - (b[i] ?? 0);
  return 0;
}

function canonical(args: JsonObject): string {
  return JSON.stringify(Object.fromEntries(Object.entries(args).sort(([a], [b]) => a.localeCompare(b))));
}
