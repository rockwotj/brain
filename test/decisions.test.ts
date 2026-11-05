/** The DecisionModel port: typed answers, validation, and its (soft) effect on routing and escalation. */
import { describe, expect, test } from "bun:test";
import { checkDecisions, type DecisionQuestion, type MediaPart } from "../src/index.ts";
import { FAST, REASONING } from "./fixtures/export-incident.ts";
import { ScriptedDecisions, yes } from "./fixtures/scripted-decisions.ts";
import { build, reply } from "./helpers.ts";

const two = [
  { name: "triage", instructions: "investigate", tools: ["read_job_logs", "check_quota"], canFinish: true },
  { name: "ops", instructions: "remediate", tools: ["restart_job"] },
];
const agents = (requests: { agent: string }[]) => [...new Set(requests.map((r) => r.agent))].sort();

describe("typed decisions", () => {
  const questions: Record<string, DecisionQuestion> = {
    refund: { type: "boolean", instructions: "Does this ask for a refund?" },
    team: { type: "choice", instructions: "Which team?", criteria: { billing: "money", tech: "bugs" } },
    mood: { type: "score", instructions: "How frustrated?", levels: ["calm", "annoyed", "angry"] },
  };

  test("well-formed boolean, choice and score answers are accepted as typed values", () => {
    const raw = { refund: yes(0.9), team: { type: "choice", choice: "billing", probabilities: { billing: 0.8, tech: 0.2 } }, mood: { type: "score", score: 1.4 } };
    expect(checkDecisions(questions, raw)).toEqual({ answers: raw as never, issues: [] });
  });

  test("missing, mistyped or out-of-range answers are dropped with a reason", () => {
    const raw = { refund: yes(1.5), team: { type: "choice", choice: "legal" }, extra: yes(1) };
    const { answers, issues } = checkDecisions(questions, raw);
    expect(answers).toEqual({});
    expect(issues).toEqual([
      "refund: probability must be a number in [0, 1]",
      "team: choice must be one of billing, tech",
      "mood: missing answer",
    ]);
    expect(checkDecisions(questions, { mood: { type: "boolean", probability: 0.5 } }).issues).toContain("mood: expected a score answer, got boolean");
  });
});

describe("decision model in the loop (soft routing only)", () => {
  test("routing: only specialists judged relevant are woken; the others stay silent and cost nothing", async () => {
    const decisions = new ScriptedDecisions((r) => (r.purpose === "route" ? { triage: yes(0.92), ops: yes(0.1) } : { escalate: yes(0.05) }));
    const { harness, llm } = build(() => reply({ kind: "finish", summary: "Nothing to do.", evidence: [] }), { specialists: two, decisions: { model: decisions } });
    const result = await harness.run({ input: "is everything fine?" });
    expect(agents(llm.requests)).toEqual(["triage"]);
    expect(result.status).toBe("finished");
    expect(result.trace.join("\n")).toContain("route: triage p=0.92 wake, ops p=0.10 skip");
    expect(result.decisions[0]).toEqual({ step: 1, purpose: "route", answers: { triage: yes(0.92), ops: yes(0.1) }, issues: [] } as never);
    // The decision model is asked typed questions about a text state; it never sees a chat prompt.
    expect(decisions.requests[0]!.state).toContain("GOAL: is everything fine?");
    expect(Object.values(decisions.requests[0]!.questions).every((q) => q.type === "boolean")).toBe(true);
  });

  test("escalation: a judgment consults the executive before a proposed action runs", async () => {
    const decisions = new ScriptedDecisions((r) => (r.purpose === "route" ? { triage: yes(0.9), ops: yes(0.9) } : { escalate: yes(0.8) }));
    const { harness, llm, tools } = build((r) => (r.agent === "triage" ? reply({ kind: "action", tool: "read_job_logs", args: { job: "j" } }) : reply()),
      { specialists: two, decisions: { model: decisions }, limits: { maxSteps: 1 } });
    const result = await harness.run({ input: "why did the job fail?" });
    expect(result.trace.join("\n")).toContain("escalate (judged)");
    expect(llm.requests.findIndex((r) => r.agent === "executive")).toBeGreaterThan(-1);
    const escalateState = decisions.requests.find((r) => r.purpose === "escalate")!.state;
    expect(escalateState).toContain("read_job_logs");
    expect(tools.calls.map((c) => c.tool)).toEqual(["read_job_logs"]); // the executive added nothing, so the read still runs
  });

  test("escalation: a low judgment replaces the novelty rule (no executive call on an empty first step)", async () => {
    const decisions = new ScriptedDecisions((r) => (r.purpose === "route" ? {} : { escalate: yes(0.1) }));
    const { harness, llm } = build(() => reply(), { specialists: two, decisions: { model: decisions }, limits: { maxSteps: 1 } });
    const result = await harness.run({ input: "hello" });
    expect(llm.requests.some((r) => r.agent === "executive")).toBe(false);
    expect(result.trace.join("\n")).toContain("judge escalation: p=0.10");
  });

  test("invalid answers or a failing decision model fall back to the deterministic rules", async () => {
    const invalid = new ScriptedDecisions(() => ({ triage: yes(2) })); // out of range; ops and escalate missing
    const a = build(() => reply(), { specialists: two, decisions: { model: invalid }, limits: { maxSteps: 1 } });
    const ra = await a.harness.run({ input: "hello" });
    expect(agents(a.llm.requests)).toEqual(["executive", "ops", "triage"]); // all due specialists wake; novelty escalates
    expect(ra.trace.join("\n")).toContain("escalate (novelty)");
    expect(ra.decisions.flatMap((d) => d.issues)).toContain("triage: probability must be a number in [0, 1]");

    const failing = new ScriptedDecisions(() => { throw new Error("decision service unavailable"); });
    const b = build(() => reply(), { specialists: two, decisions: { model: failing }, limits: { maxSteps: 1 } });
    const rb = await b.harness.run({ input: "hello" });
    expect(agents(b.llm.requests)).toEqual(["executive", "ops", "triage"]);
    expect(rb.trace.join("\n")).toContain("scripted-decisions failed: decision service unavailable (deterministic fallback)");
  });

  test("hard reasons and policy do not depend on probabilities", async () => {
    // The decision model wants nothing escalated, but images only the executive can see still escalate first.
    const image: MediaPart = { type: "image", mimeType: "image/png", base64: "iVBORw0KGgo=" };
    const never = new ScriptedDecisions((r) => (r.purpose === "route" ? { triage: yes(1), ops: yes(1) } : { escalate: yes(0) }));
    const { harness, tools } = build((r) => (r.agent === "executive" ? reply() : reply({ kind: "action", tool: "restart_job", args: { job: "j" } })),
      { specialists: two, decisions: { model: never }, profiles: { fast: FAST, reasoning: { ...REASONING, vision: true } }, limits: { maxSteps: 1 } });
    const result = await harness.run({ input: [{ type: "text", text: "look" }, image], constraints: [{ id: "c1", kind: "read_only", reason: "read only" }] });
    expect(result.trace.join("\n")).toContain("escalate (media)");
    expect(never.requests.some((r) => r.purpose === "escalate")).toBe(false);
    expect(tools.calls).toHaveLength(0); // the write is still blocked by the read-only constraint
  });
});
