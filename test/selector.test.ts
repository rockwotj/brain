import { describe, expect, test } from "bun:test";
import { Actuator, addCandidate, FunctionTools, independentSupport, parseProposals, select, type Candidate } from "../src/index.ts";
import { TOOLS } from "../examples/export-incident.ts";

const actuator = new Actuator(TOOLS, new FunctionTools({}), { allowEffects: ["read", "write"] });
const ctx = { actuator, constraints: [], executedKeys: new Set<string>(), successfulActions: new Set<string>(), careful: false };
const action = (tool: string, args: Record<string, string> = { job: "j" }) => ({ kind: "action" as const, call: { tool, args }, rationale: "", expect: { success: true } });

describe("selector", () => {
  test("agreement from the same model profile counts once", () => {
    const pool: Candidate[] = [];
    addCandidate(pool, action("read_job_logs"), "specialist", { agent: "a", profileId: "s1" });
    addCandidate(pool, action("read_job_logs"), "specialist", { agent: "b", profileId: "s1" });
    addCandidate(pool, action("read_job_logs"), "specialist", { agent: "c", profileId: "s2" });
    expect(pool).toHaveLength(1);
    expect(independentSupport(pool[0]!)).toBe(2);
    const { selection } = select(pool, ctx);
    expect(selection.kind === "act" && selection.why).toContain("independent support=2 (3 proposals)");
  });

  test("prefers reads over writes, and holds on a tied write conflict", () => {
    const pool: Candidate[] = [];
    addCandidate(pool, action("restart_job"), "specialist", { agent: "a", profileId: "s1" });
    addCandidate(pool, action("read_job_logs"), "specialist", { agent: "b", profileId: "s1" });
    const first = select(pool, ctx).selection;
    expect(first.kind === "act" && first.candidate.key).toStartWith("read_job_logs");

    const conflict: Candidate[] = [];
    addCandidate(conflict, action("restart_job", { job: "a" }), "specialist", { agent: "a", profileId: "s1" });
    addCandidate(conflict, action("restart_job", { job: "b" }), "specialist", { agent: "b", profileId: "s1" });
    expect(select(conflict, ctx).selection.kind).toBe("hold");
  });

  test("careful mode holds non-executive writes", () => {
    const pool: Candidate[] = [];
    addCandidate(pool, action("restart_job"), "habit", { agent: "habit:hb-1", profileId: "habit" });
    const result = select(pool, { ...ctx, careful: true });
    expect(result.selection.kind).toBe("none");
    expect(result.rejected[0]).toContain("careful mode");
  });

  test("actuator re-checks policy even if called directly", async () => {
    const tools = new FunctionTools({ restart_job: () => "restarted" });
    const direct = new Actuator(TOOLS, tools, { allowEffects: ["read", "write"] });
    const ex = await direct.execute({ tool: "restart_job", args: { job: "j" } }, {
      step: 1, source: "executive", predicted: { success: true }, constraints: [{ id: "c-1", kind: "read_only", reason: "audit" }],
    });
    expect(ex.kind).toBe("blocked");
    const controller = new AbortController();
    controller.abort();
    const call = direct.execute({ tool: "read_job_logs", args: { job: "j" } }, { step: 1, source: "t", predicted: { success: true }, constraints: [] }, controller.signal);
    await expect(call).rejects.toThrow();
    expect(tools.calls).toHaveLength(0);
  });
});

describe("LLM boundary validation", () => {
  const rules = { allowedTools: new Set(["read_job_logs"]), allowActions: true, allowFinish: false, maxProposals: 2 };
  test("parses fenced JSON and drops invalid entries with reasons", () => {
    const text = '```json\n{"proposals":[{"kind":"observation","content":"ok","confidence":"0.93"},{"kind":"finish","summary":"x","evidence":["act-1"]},{"kind":"action","tool":"read_job_logs","args":{"job":"j"}}]}\n```';
    const parsed = parseProposals(text, rules);
    expect(parsed.value).toHaveLength(1);
    expect(parsed.value[0]!.kind).toBe("action");
    expect(parsed.issues).toEqual(["proposal 0: confidence must be low|medium|high", "proposal 1: finish is not allowed for this agent"]);
  });
  test("rejects non-JSON and caps proposal count", () => {
    expect(parseProposals("sure! here you go", rules).issues).toEqual(["reply is not valid JSON"]);
    const many = JSON.stringify({ proposals: [1, 2, 3].map((n) => ({ kind: "observation", content: `o${n}` })) });
    expect(parseProposals(many, rules).value).toHaveLength(2);
  });
});
