import { describe, expect, test } from "bun:test";
import { Actuator, addCandidate, FunctionTools, independentSupport, parseProposals, select, type Candidate, type SelectContext } from "../src/index.ts";
import { TOOLS } from "./fixtures/export-incident.ts";

const actuator = new Actuator(TOOLS, new FunctionTools({}), { allowEffects: ["read", "write"] });
const ctx: SelectContext = {
  actuator, constraints: [], repeatBlock: () => undefined,
  knownIds: new Set(["act-1", "act-2", "ws-1"]), successfulActions: new Set(["act-1"]), careful: false,
};
const action = (tool: string, args: Record<string, string> = { job: "j" }) => ({ kind: "action" as const, call: { tool, args }, rationale: "", expect: { success: true } });
const finish = (summary: string, evidence: string[] = []) => ({ kind: "finish" as const, summary, evidence });
const s1 = (agent: string) => ({ agent, profileId: "s1" });

describe("selector", () => {
  test("agreement from the same model profile counts once", () => {
    const pool: Candidate[] = [];
    addCandidate(pool, action("read_job_logs"), "specialist", s1("a"));
    addCandidate(pool, action("read_job_logs"), "specialist", s1("b"));
    addCandidate(pool, action("read_job_logs"), "specialist", { agent: "c", profileId: "s2" });
    expect(pool).toHaveLength(1);
    expect(independentSupport(pool[0]!)).toBe(2);
    const { selection } = select(pool, ctx);
    expect(selection.kind === "act" && selection.why).toContain("independent support=2 (3 proposals)");
  });

  test("prefers reads over writes, and holds on a tied write conflict", () => {
    const pool: Candidate[] = [];
    addCandidate(pool, action("restart_job"), "specialist", s1("a"));
    addCandidate(pool, action("read_job_logs"), "specialist", s1("b"));
    const first = select(pool, ctx).selection;
    expect(first.kind === "act" && first.candidate.key).toStartWith("read_job_logs");

    const conflict: Candidate[] = [];
    addCandidate(conflict, action("restart_job", { job: "a" }), "specialist", s1("a"));
    addCandidate(conflict, action("restart_job", { job: "b" }), "specialist", s1("b"));
    expect(select(conflict, ctx).selection.kind).toBe("hold");
  });

  test("careful mode holds non-executive writes", () => {
    const pool: Candidate[] = [];
    addCandidate(pool, action("restart_job"), "habit", { agent: "habit:hb-1", profileId: "habit" });
    const result = select(pool, { ...ctx, careful: true });
    expect(result.selection.kind).toBe("none");
    expect(result.rejected[0]).toContain("careful mode");
  });

  test("repeat rules come from the context and see the tool's effect", () => {
    const pool: Candidate[] = [];
    addCandidate(pool, action("read_job_logs"), "specialist", s1("a"));
    const result = select(pool, { ...ctx, repeatBlock: (_key, effect) => (effect === "read" ? "seen" : undefined) });
    expect(result.selection.kind).toBe("none");
    expect(result.rejected).toEqual(["read_job_logs rejected: seen"]);
  });

  test("actuator re-checks policy even if called directly", async () => {
    const tools = new FunctionTools({ restart_job: () => "restarted" });
    const direct = new Actuator(TOOLS, tools, { allowEffects: ["read", "write"] });
    const meta = { id: "act-1", step: 1, source: "executive", predicted: { success: true } };
    const ex = await direct.execute({ tool: "restart_job", args: { job: "j" } }, { ...meta, constraints: [{ id: "c-1", kind: "read_only", reason: "audit" }] });
    expect(ex.kind).toBe("blocked");
    const controller = new AbortController();
    controller.abort();
    await expect(direct.execute({ tool: "read_job_logs", args: { job: "j" } }, { ...meta, constraints: [] }, controller.signal)).rejects.toThrow();
    expect(tools.calls).toHaveLength(0);
  });
});

describe("finish arbitration", () => {
  test("an executive action beats a specialist's verified finish (no early return)", () => {
    const pool: Candidate[] = [];
    addCandidate(pool, finish("done", ["act-1"]), "specialist", s1("triage"));
    addCandidate(pool, action("get_input_stats"), "executive", { agent: "executive", profileId: "s2" });
    const { selection } = select(pool, ctx);
    expect(selection.kind).toBe("act");
  });

  test("at equal origin: verified finish > read > write > answer-only finish", () => {
    const verified: Candidate[] = [];
    addCandidate(verified, action("read_job_logs"), "specialist", s1("a"));
    addCandidate(verified, finish("timed out", ["act-1"]), "specialist", s1("b"));
    const v = select(verified, ctx).selection;
    expect(v.kind === "finish" && v.completion).toBe("verified");

    const answerOnly: Candidate[] = [];
    addCandidate(answerOnly, finish("probably fine"), "specialist", s1("a"));
    addCandidate(answerOnly, action("restart_job"), "specialist", s1("b"));
    expect(select(answerOnly, ctx).selection.kind).toBe("act");

    const alone: Candidate[] = [];
    addCandidate(alone, finish("Hello! How can I help?"), "specialist", s1("a"));
    const a = select(alone, ctx).selection;
    expect(a.kind === "finish" && a.completion).toBe("answer_only");
  });

  test("a finish citing only workspace items is answer-only; unknown ids are rejected", () => {
    const pool: Candidate[] = [];
    addCandidate(pool, finish("from notes", ["ws-1", "act-2"]), "specialist", s1("a"));
    const sel = select(pool, ctx).selection;
    expect(sel.kind === "finish" && sel.completion).toBe("answer_only"); // act-2 exists but did not succeed

    const bad: Candidate[] = [];
    addCandidate(bad, finish("made up", ["act-1", "act-9"]), "specialist", s1("a"));
    const result = select(bad, ctx);
    expect(result.selection.kind).toBe("none");
    expect(result.rejected).toEqual(["finish rejected: cites unknown act-9"]);
  });
});

describe("LLM boundary validation", () => {
  const rules = { allowedTools: new Set(["read_job_logs"]), allowActions: true, allowFinish: false, maxProposals: 3 };
  test("parses fenced JSON and drops invalid entries with reasons", () => {
    const text = '```json\n{"proposals":[{"kind":"observation","content":"ok","confidence":"0.93"},{"kind":"finish","summary":"x","evidence":["act-1"]},{"kind":"action","tool":"read_job_logs","args":{"job":"j"}}]}\n```';
    const parsed = parseProposals(text, rules);
    expect(parsed.value).toHaveLength(1);
    expect(parsed.value[0]!.kind).toBe("action");
    expect(parsed.issues).toEqual(["proposal 0: confidence must be low|medium|high", "proposal 1: finish is not allowed for this agent"]);
  });
  test("rejects non-JSON and caps proposal count", () => {
    expect(parseProposals("sure! here you go", rules).issues).toEqual(["reply is not valid JSON"]);
    const many = JSON.stringify({ proposals: [1, 2, 3, 4].map((n) => ({ kind: "observation", content: `o${n}` })) });
    expect(parseProposals(many, rules).value).toHaveLength(3);
  });
  test("read_record and evidence-free finish proposals", () => {
    const text = JSON.stringify({ proposals: [
      { kind: "read_record", id: "act-2" },
      { kind: "read_record", id: "act-2", offset: -1 },
      { kind: "finish", summary: "Hi there" },
    ] });
    const parsed = parseProposals(text, { ...rules, allowFinish: true });
    expect(parsed.value).toEqual([{ kind: "read_record", id: "act-2", offset: 0 }, { kind: "finish", summary: "Hi there", evidence: [] }]);
    expect(parsed.issues).toEqual(["proposal 1: read_record offset must be a non-negative integer"]);
  });
});
