/**
 * Action selector: many candidates in, at most one released per step.
 * Deterministic lexicographic ranking with a plain-English explanation,
 * after hard policy filtering. No weighted float scores.
 *
 * Rank = (origin, kind, independent support):
 *   origin: executive > habit > specialist
 *   kind:   verified finish > read > write > answer-only finish
 * So a finish competes like any other candidate: an executive action beats a
 * specialist's finish, and an answer-only finish wins only when nobody
 * proposes an action.
 */
import type { ActionProposal, Completion, Constraint, Effect, FinishProposal, JsonObject, ToolCall } from "./types.ts";
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
  | { kind: "finish"; candidate: Candidate; completion: Completion; why: string }
  | { kind: "hold"; why: string }
  | { kind: "none"; why: string };

export type SelectContext = {
  actuator: Actuator;
  constraints: readonly Constraint[];
  /** Why an identical call may not run again right now (undefined = allowed). */
  repeatBlock: (key: string, effect: Effect) => string | undefined;
  /** Identifiers a finish may cite. */
  knownIds: ReadonlySet<string>;
  /** Ids of successful action records; citing one makes a finish verified. */
  successfulActions: ReadonlySet<string>;
  careful: boolean;
};

export type SelectResult = { selection: Selection; blocked: { call: ToolCall; reason: string }[]; rejected: string[] };

const ORIGIN_RANK: Record<Origin, number> = { executive: 2, habit: 1, specialist: 0 };
const KIND_RANK = { verified: 3, read: 2, write: 1, answer_only: 0 } as const;

export function candidateKey(p: ActionProposal | FinishProposal): string {
  return p.kind === "finish" ? `finish:${p.summary.trim()}` : `${p.call.tool}:${canonical(p.call.args)}`;
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

type Ranked = { c: Candidate; rank: number[]; label: string; writes: boolean; completion?: Completion };

export function select(pool: readonly Candidate[], ctx: SelectContext): SelectResult {
  const blocked: SelectResult["blocked"] = [];
  const rejected: string[] = [];
  const ranked: Ranked[] = [];

  for (const c of pool) {
    const p = c.proposal;
    if (p.kind === "finish") {
      const unknown = p.evidence.filter((id) => !ctx.knownIds.has(id));
      if (unknown.length) {
        rejected.push(`finish rejected: cites unknown ${unknown.join(", ")}`);
        continue;
      }
      const completion: Completion = p.evidence.some((id) => ctx.successfulActions.has(id)) ? "verified" : "answer_only";
      ranked.push({ c, completion, writes: false, label: `${completion} finish`, rank: [ORIGIN_RANK[c.origin], KIND_RANK[completion], independentSupport(c)] });
      continue;
    }
    const decision = ctx.actuator.check(p.call, ctx.constraints);
    if (!decision.allowed) {
      blocked.push({ call: p.call, reason: decision.reason });
      continue;
    }
    const effect = decision.spec.effect;
    const repeat = ctx.repeatBlock(c.key, effect);
    if (repeat) {
      rejected.push(`${p.call.tool} rejected: ${repeat}`);
      continue;
    }
    if (ctx.careful && effect === "write" && c.origin !== "executive") {
      rejected.push(`${p.call.tool} held: careful mode requires executive backing for writes`);
      continue;
    }
    ranked.push({ c, writes: effect === "write", label: effect, rank: [ORIGIN_RANK[c.origin], KIND_RANK[effect], independentSupport(c)] });
  }

  ranked.sort((a, b) => compare(b.rank, a.rank)); // stable: ties keep proposal order
  const [best, second] = ranked;
  if (!best) return { selection: { kind: "none", why: "no eligible candidates" }, blocked, rejected };
  const tied = !!second && compare(best.rank, second.rank) === 0;
  if (tied && (best.writes || second.writes)) {
    return { selection: { kind: "hold", why: `conflict: ${best.c.key} vs ${second.c.key} tie and at least one writes` }, blocked, rejected };
  }
  const why =
    `origin=${best.c.origin}, kind=${best.label}, independent support=${independentSupport(best.c)}` +
    ` (${best.c.supporters.length} proposals)` + (tied ? "; tie broken by proposal order" : "") +
    (ranked.length > 1 ? `; ${ranked.length - 1} other candidate(s) ranked lower` : "");
  if (best.completion) return { selection: { kind: "finish", candidate: best.c, completion: best.completion, why }, blocked, rejected };
  return { selection: { kind: "act", candidate: best.c, why }, blocked, rejected };
}

function compare(a: number[], b: number[]): number {
  for (let i = 0; i < a.length; i++) if ((a[i] ?? 0) !== (b[i] ?? 0)) return (a[i] ?? 0) - (b[i] ?? 0);
  return 0;
}

function canonical(args: JsonObject): string {
  return JSON.stringify(Object.fromEntries(Object.entries(args).sort(([a], [b]) => a.localeCompare(b))));
}
