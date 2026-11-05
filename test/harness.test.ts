import { describe, expect, test } from "bun:test";
import { FunctionTools, withUrgency, type LlmRequest } from "../src/index.ts";
import { FAST, GOAL, makeScenario, READ_ONLY } from "./fixtures/export-incident.ts";
import { approveHabit, build, isPrefix, narration, reply } from "./helpers.ts";

describe("end-to-end scenario (scripted models)", () => {
  test("finishes with a grounded answer, cheap models first and the executive only on surprise", async () => {
    const { harness, llm, tools } = await makeScenario();
    const result = await harness.run({ input: GOAL, constraints: [{ ...READ_ONLY }] });
    expect(result.status).toBe("finished");
    expect(result.completion).toBe("verified");
    expect(result.answer.grounded).toBe(true);
    expect(result.answer.citations).toEqual(["act-1", "act-2", "act-3"]);

    const tiers = result.modelCalls.map((c) => c.tier);
    const firstReasoning = tiers.indexOf("reasoning");
    expect(firstReasoning).toBeGreaterThan(0);
    expect(tiers.slice(0, firstReasoning).every((t) => t === "fast")).toBe(true);
    expect(tiers.filter((t) => t === "reasoning")).toHaveLength(1);
    const executiveCall = result.modelCalls.find((c) => c.agent === "executive")!;
    const surprise = result.actions.find((a) => a.surprise)!;
    expect(executiveCall.step).toBe(surprise.step + 1);
    expect(result.trace.join("\n")).toContain("escalate (surprise)");

    // Policy held: the write proposal never reached the tool port.
    expect(tools.calls.map((c) => c.tool)).not.toContain("restart_job");
    expect(result.events.some((e) => e.delta.type === "action.blocked")).toBe(true);
    // The recalled claim (via embeddings) was treated as a claim and contradicted by evidence.
    expect(result.events.some((e) => e.delta.type === "workspace.admit" && e.delta.item.content.startsWith("Recalled claim (unverified)"))).toBe(true);
    expect(surprise.call.tool).toBe("check_quota");
    expect(llm.requests.every((r) => r.profile.serviceMode === "standard")).toBe(true);
  });

  test("each agent's repeated invocations extend the exact same message prefix", async () => {
    const { harness, llm } = await makeScenario();
    await harness.run({ input: GOAL, constraints: [{ ...READ_ONLY }] });
    const byAgent = new Map<string, LlmRequest[]>();
    for (const r of llm.requests) byAgent.set(r.agent, [...(byAgent.get(r.agent) ?? []), r]);
    expect(byAgent.get("triage")!.length).toBeGreaterThan(2);
    for (const requests of byAgent.values()) {
      for (let i = 1; i < requests.length; i++) {
        expect(isPrefix(requests[i - 1]!.messages, requests[i]!.messages)).toBe(true);
        expect(requests[i]!.stablePrefixLength).toBe(requests[i - 1]!.messages.length + 1);
      }
    }
    const system = byAgent.get("triage")![0]!.messages[0]!.content;
    expect(system).toContain("check_quota [read]");
    expect(system).toContain("OUTPUT CONTRACT");
  });

  test("urgency switches the provider service lane without changing tier or effort", async () => {
    const { harness, llm } = await makeScenario();
    await harness.run({ input: GOAL, constraints: [{ ...READ_ONLY }], urgency: "high" });
    const triage = llm.requests.find((r) => r.agent === "triage")!;
    expect(triage.profile.serviceMode).toBe("fast");
    expect(triage.profile.tier).toBe("fast");
    expect(llm.requests.find((r) => r.agent === "executive")!.profile.reasoningEffort).toBe("high");
  });

  test("urgency never downgrades an explicitly faster lane", () => {
    expect(withUrgency({ ...FAST, serviceMode: "ultrafast" }, "high").serviceMode).toBe("ultrafast");
    expect(withUrgency({ ...FAST, serviceMode: "fast" }, "high").serviceMode).toBe("fast");
    expect(withUrgency(FAST, "high")).toMatchObject({ serviceMode: "fast", model: FAST.model, tier: "fast", reasoningEffort: "none" });
    expect(withUrgency({ ...FAST, serviceMode: "ultrafast" }, "normal").serviceMode).toBe("ultrafast");
  });
});

describe("specialists", () => {
  test("woken specialists run in parallel and stay silent by default", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const names = ["a", "b", "c"];
    const { harness } = build(
      async () => {
        inFlight++;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await new Promise((r) => setTimeout(r, 15));
        inFlight--;
        return reply();
      },
      { specialists: names.map((name) => ({ name, instructions: "watch", tools: [] })), limits: { maxExecutiveCalls: 0 } },
    );
    const result = await harness.run({ input: "anything" });
    expect(maxInFlight).toBe(3);
    expect(result.modelCalls.filter((c) => c.step === 1)).toHaveLength(3);
    // Nobody spoke, so nobody is woken again: no broadcast storm.
    expect(result.modelCalls.filter((c) => c.step > 1)).toHaveLength(0);
  });

  test("invalid model output is dropped and reported, never trusted", async () => {
    const { harness, tools } = build(() => '{"proposals":[{"kind":"action","tool":"drop_database","args":{}}, {"kind":"action","tool":"read_job_logs","args":"x"}]}', {
      limits: { maxExecutiveCalls: 0 },
    });
    const result = await harness.run({ input: "g" });
    expect(tools.calls).toHaveLength(0);
    expect(result.trace.join("\n")).toContain("unknown or unavailable tool drop_database");
    expect(result.trace.join("\n")).toContain("args must be a JSON object");
  });
});

describe("reflex path", () => {
  test("a single approved read-only habit runs without any model call that step", async () => {
    const { harness, tools } = build(() => reply());
    await approveHabit(harness, { tool: "read_job_logs", args: { job: "j" } });
    const result = await harness.run({ input: "export status" });
    expect(tools.calls[0]?.tool).toBe("read_job_logs");
    expect(result.actions[0]!.step).toBe(1);
    expect(result.modelCalls.filter((c) => c.step === 1)).toHaveLength(0);
    expect(result.trace.join("\n")).toContain("deliberation skipped");
  });

  test("ambiguous or write habits go through deliberation instead", async () => {
    const two = build(() => reply());
    await approveHabit(two.harness, { tool: "read_job_logs", args: { job: "a" } });
    await approveHabit(two.harness, { tool: "check_quota", args: { project: "p" } });
    const r1 = await two.harness.run({ input: "export status" });
    expect(r1.modelCalls.filter((c) => c.step === 1).length).toBeGreaterThan(0);
    expect(r1.actions[0]!.source).toStartWith("habit:"); // still reused, but as a ranked candidate

    const write = build(() => reply());
    await approveHabit(write.harness, { tool: "restart_job", args: { job: "j" } });
    const r2 = await write.harness.run({ input: "export status" });
    expect(r2.trace.join("\n")).not.toContain("deliberation skipped");
    expect(r2.modelCalls.filter((c) => c.step === 1).length).toBeGreaterThan(0);
  });
});

describe("policy gate", () => {
  test("neither the executive nor an approved habit can bypass read-only", async () => {
    const { harness, tools } = build((r) =>
      r.agent === "executive"
        ? reply({ kind: "action", tool: "restart_job", args: { job: "nightly-export" }, effect: "read", expect: { success: true } })
        : reply());
    await approveHabit(harness, { tool: "restart_job", args: { job: "nightly-export" } });
    const result = await harness.run({ input: "restart the export", constraints: [{ ...READ_ONLY }] });
    expect(tools.calls).toHaveLength(0);
    expect(result.events.filter((e) => e.delta.type === "action.blocked").length).toBeGreaterThan(0);
    expect(result.modelCalls.some((c) => c.agent === "executive")).toBe(true);
    expect(result.status).toBe("no_progress");
  });

  test("deployment permissions and mid-run deny rules are enforced too", async () => {
    const { harness, tools } = build(() => reply({ kind: "action", tool: "check_quota", args: { project: "p" } }), {
      permissions: { allowEffects: ["read"] },
    });
    await approveHabit(harness, { tool: "restart_job", args: { job: "nightly-export" } });
    const result = await harness.run({
      input: "export check",
      onStep: async (step, state) => {
        if (step === 1) await state.apply("user", { type: "constraint.add", constraint: { id: "c-7", kind: "deny_tool", tool: "check_quota", reason: "rate limited" } });
      },
    });
    expect(tools.calls).toHaveLength(0);
    const reasons = result.events.flatMap((e) => (e.delta.type === "action.blocked" ? [e.delta.reason] : []));
    expect(reasons.some((r) => r.includes("permission: write"))).toBe(true);
    expect(reasons.some((r) => r.includes("c-7 denies check_quota"))).toBe(true);
  });
});

describe("outcomes, monitoring and termination", () => {
  test("logs tool errors as failed outcomes, flags surprise and suspends a failing habit", async () => {
    const toolPort = new FunctionTools({ read_job_logs: () => { throw new Error("log service unavailable"); } });
    const { harness } = build(() => reply(), { toolPort });
    await approveHabit(harness, { tool: "read_job_logs", args: { job: "nightly-export" } });
    const result = await harness.run({ input: "export status" });
    const record = result.actions[0]!;
    expect(record.actual).toEqual({ ok: false, error: "log service unavailable" });
    expect(record.surprise).toBe(true);
    expect(result.trace.join("\n")).toContain("ERROR (23 chars): log service unavailable");
    expect((await harness.habits.list())[0]!.status).toBe("suspended");
    expect(result.events.some((e) => e.delta.type === "mode.set" && e.delta.mode.carefulSteps > 0)).toBe(true);
  });

  test("a successful expected outcome is not a surprise and records the output", async () => {
    const { harness } = build((r, i) =>
      r.agent === "triage" && i === 0
        ? reply({ kind: "action", tool: "read_job_logs", args: { job: "j" }, expect: { success: true, contains: "timeout" } })
        : r.agent === "triage"
          ? reply({ kind: "finish", summary: "it timed out", evidence: ["act-1"] })
          : narration("It timed out [act-1].", ["act-1"]));
    const result = await harness.run({ input: "g" });
    expect(result.status).toBe("finished");
    expect(result.actions[0]!.actual).toEqual({ ok: true, output: "log: timeout" });
    expect(result.actions[0]!.surprise).toBe(false);
    expect(result.modelCalls.some((c) => c.agent === "executive")).toBe(false);
    expect(result.answer.grounded).toBe(true);
  });

  test("stops with no_progress when nothing moves even after escalation", async () => {
    const { harness, llm } = build(() => reply());
    const result = await harness.run({ input: "g" });
    expect(result.status).toBe("no_progress");
    expect(result.steps).toBeLessThanOrEqual(3);
    expect(llm.requests.filter((r) => r.agent === "executive")).toHaveLength(1);
    expect(result.answer.grounded).toBe(false);
  });

  test("repeating a read with nothing written in between does not loop forever", async () => {
    const { harness, tools } = build(() => reply({ kind: "action", tool: "read_job_logs", args: { job: "j" } }), { limits: { maxSteps: 10 } });
    const result = await harness.run({ input: "g" });
    expect(tools.calls).toHaveLength(1);
    expect(result.status).toBe("no_progress");
    expect(result.trace.join("\n")).toContain("identical read already ran since the last write");
  });

  test("finish claims citing unknown ids are rejected", async () => {
    const { harness } = build(() => reply({ kind: "finish", summary: "done", evidence: ["act-42"] }));
    const result = await harness.run({ input: "g" });
    expect(result.status).not.toBe("finished");
    expect(result.trace.join("\n")).toContain("finish rejected: cites unknown act-42");
  });

  test("keeps one model call in reserve so the narrator stays within budget", async () => {
    const script = (r: LlmRequest, i: number) =>
      r.agent === "narrator" ? narration("Timed out [act-1].", ["act-1"])
        : i === 0 ? reply({ kind: "action", tool: "read_job_logs", args: { job: "j" } })
          : reply({ kind: "finish", summary: "timed out", evidence: ["act-1"] });
    const enough = await build(script, { limits: { maxModelCalls: 3 } }).harness.run({ input: "g" });
    expect(enough.status).toBe("finished");
    expect(enough.modelCalls.map((c) => c.agent)).toEqual(["triage", "triage", "narrator"]);
    expect(enough.answer.grounded).toBe(true);

    const tight = await build(script, { limits: { maxModelCalls: 2 } }).harness.run({ input: "g" });
    expect(tight.status).toBe("budget_exhausted");
    expect(tight.modelCalls.length).toBeLessThanOrEqual(2);
  });

  test("cancellation during a model call prevents the proposed side effect", async () => {
    const controller = new AbortController();
    const { harness, tools } = build(() => {
      controller.abort();
      return reply({ kind: "action", tool: "read_job_logs", args: { job: "j" } });
    });
    const result = await harness.run({ input: "g", signal: controller.signal });
    expect(result.status).toBe("cancelled");
    expect(tools.calls).toHaveLength(0);
    expect(result.modelCalls.some((c) => c.agent === "narrator")).toBe(false);
  });

  test("respects the model-call budget and cancellation", async () => {
    const specialists = ["a", "b", "c"].map((name) => ({ name, instructions: "x", tools: [] }));
    const budget = await build(() => reply(), { specialists, limits: { maxModelCalls: 2 } }).harness.run({ input: "g" });
    expect(budget.status).toBe("budget_exhausted");
    expect(budget.modelCalls).toHaveLength(0);

    const controller = new AbortController();
    const { harness } = build((_r, i) => reply({ kind: "observation", content: `note ${i}` }));
    const cancelled = await harness.run({ input: "g", signal: controller.signal, onStep: (step) => step === 2 && controller.abort() });
    expect(cancelled.status).toBe("cancelled");
    expect(cancelled.steps).toBe(2);
  });
});
