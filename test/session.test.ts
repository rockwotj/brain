/** Behaviour added by the converged design: sessions, evidence, routing, completion, media. */
import { describe, expect, test } from "bun:test";
import { buildSharedPrompt, FunctionTools, UnsupportedContentError, type LiveModel, type LlmRequest, type MediaPart } from "../src/index.ts";
import { SYSTEM1, SYSTEM2, TOOLS } from "./fixtures/export-incident.ts";
import { lastText, transcriptText } from "./fixtures/scripted-llm.ts";
import { build, isPrefix, narration, reply } from "./helpers.ts";

const two = [
  { name: "triage", instructions: "investigate", tools: ["read_job_logs", "check_quota"], canFinish: true },
  { name: "ops", instructions: "remediate", tools: ["restart_job"] },
];
const IMAGE: MediaPart = { type: "image", mimeType: "image/png", base64: "iVBORw0KGgo=" };
const AUDIO: MediaPart = { type: "audio", mimeType: "audio/wav", base64: "UklGRg==" };
const hasMedia = (r: LlmRequest) => r.messages.some((m) => typeof m.content !== "string" && m.content.some((p) => p.type !== "text"));

describe("shared prompt prefix", () => {
  test("every agent's system prompt starts with the same shared block; the role comes last", async () => {
    const { harness, llm } = build(() => reply(), { specialists: two });
    await harness.run({ input: "anything" });
    const systems = new Map(llm.requests.map((r) => [r.agent, String(r.messages[0]!.content)]));
    expect([...systems.keys()].sort()).toEqual(["executive", "ops", "triage"]);
    const shared = buildSharedPrompt(TOOLS);
    for (const system of systems.values()) {
      expect(system.startsWith(`${shared}\n\nROLE\n`)).toBe(true);
    }
    // Tool order in the shared block does not depend on registration order.
    expect(buildSharedPrompt([...TOOLS].reverse())).toBe(shared);
  });
});

describe("sessions", () => {
  test("agent transcripts carry across turns; the new goal arrives as a delta", async () => {
    const { harness, llm } = build((r) => (r.agent === "triage" ? reply({ kind: "finish", summary: "Hello!", evidence: [] }) : reply()));
    const session = await harness.openSession("s-1");
    await session.turn({ input: "hi" });
    const first = llm.requests.filter((r) => r.agent === "triage").at(-1)!;
    await session.turn({ input: "and again" });
    const second = llm.requests.filter((r) => r.agent === "triage").at(-1)!;
    expect(isPrefix(first.messages, second.messages)).toBe(true);
    expect(lastText(second)).toContain("goal set: and again");
    expect(second.stablePrefixLength).toBe(first.messages.length + 1);

    // An explicit task boundary starts a fresh epoch from the current snapshot.
    await session.turn({ input: "new topic", newTask: true });
    const third = llm.requests.filter((r) => r.agent === "triage").at(-1)!;
    expect(third.messages).toHaveLength(2);
    expect(lastText(third)).toContain("EPOCH 1");
  });
});

describe("completion", () => {
  test("answer-only finish: one cheap call, no tools, no narrator, marked answer_only", async () => {
    const { harness, tools } = build(() => reply({ kind: "finish", summary: "Paris is the capital of France.", evidence: [] }));
    const result = await harness.run({ input: "What is the capital of France?" });
    expect(result.status).toBe("finished");
    expect(result.completion).toBe("answer_only");
    expect(result.answer).toEqual({ text: "Paris is the capital of France.", citations: [], unknownCitations: [], grounded: false });
    expect(result.modelCalls.map((c) => c.agent)).toEqual(["triage"]);
    expect(tools.calls).toHaveLength(0);
  });

  test("a proposed read wins over an answer-only finish; the finish can win afterwards", async () => {
    const { harness, tools } = build((r) => {
      if (r.agent === "narrator") return narration("Timed out [act-1].", ["act-1"]);
      if (r.agent === "ops") return reply({ kind: "finish", summary: "Probably fine." });
      return transcriptText(r).includes("act-1") ? reply({ kind: "finish", summary: "It timed out.", evidence: ["act-1"] })
        : reply({ kind: "action", tool: "read_job_logs", args: { job: "j" } });
    }, { specialists: [two[0]!, { ...two[1]!, canFinish: true }] });
    const result = await harness.run({ input: "is the job ok?" });
    expect(tools.calls.map((c) => c.tool)).toEqual(["read_job_logs"]);
    expect(result.completion).toBe("verified");
    expect(result.answer.text).toBe("Timed out [act-1].");
  });
});

describe("repeat rules", () => {
  test("a read may repeat after a write (e.g. verify a restart), an identical write may not", async () => {
    const { harness, tools } = build((r, i) => {
      if (r.agent !== "triage") return reply();
      const read = { kind: "action", tool: "read_job_logs", args: { job: "j" } };
      const restart = { kind: "action", tool: "restart_job", args: { job: "j" } };
      const plan = [[read], [restart], [read], [restart, read]];
      return reply(...(plan[i] ?? []));
    }, { specialists: [{ name: "triage", instructions: "x", tools: ["read_job_logs", "restart_job"] }], limits: { maxSteps: 6, maxExecutiveCalls: 0 } });
    const result = await harness.run({ input: "restart and verify" });
    expect(tools.calls.map((c) => c.tool)).toEqual(["read_job_logs", "restart_job", "read_job_logs"]);
    expect(result.trace.join("\n")).toContain("restart_job rejected: identical write already executed this turn");
    expect(result.trace.join("\n")).toContain("read_job_logs rejected: identical read already ran since the last write");
  });
});

describe("evidence and routing", () => {
  const LONG = `BEGIN ${"a".repeat(7_000)} MIDDLE ${"b".repeat(4_000)} END`;

  test("expected outcomes go only to the actor; surprises are broadcast", async () => {
    const toolPort = new FunctionTools({ read_job_logs: () => "fine", check_quota: () => "quota 99%" });
    const { harness } = build((r, i) => {
      if (r.agent !== "triage" || i > 0) return reply();
      return reply(
        { kind: "action", tool: "read_job_logs", args: { job: "j" }, expect: { success: true } },
      );
    }, { specialists: two, toolPort, limits: { maxExecutiveCalls: 0 } });
    const quiet = await harness.run({ input: "check" });
    const outcome = quiet.events.find((e) => e.delta.type === "action.outcome")!;
    expect(outcome.audience).toEqual(["triage"]);
    expect(quiet.state.workspace.some((i) => i.kind === "outcome")).toBe(false);
    expect(quiet.modelCalls.filter((c) => c.agent === "ops")).toHaveLength(1); // not woken by triage's private outcome

    const loud = build((r, i) => (r.agent === "triage" && i === 0
      ? reply({ kind: "action", tool: "check_quota", args: { project: "p" }, expect: { success: true, contains: "healthy" } })
      : reply()), { specialists: two, toolPort, limits: { maxExecutiveCalls: 0 } });
    const result = await loud.harness.run({ input: "check" });
    const surprise = result.events.find((e) => e.delta.type === "action.outcome")!;
    expect(surprise.audience).toBeUndefined();
    expect(result.state.workspace.some((i) => i.kind === "outcome" && i.priority === "high")).toBe(true);
    expect(result.modelCalls.filter((c) => c.agent === "ops").length).toBeGreaterThan(1);
  });

  test("long outputs are previewed; read_record delivers chunks privately, also after compaction", async () => {
    const toolPort = new FunctionTools({ read_job_logs: () => LONG });
    const small = { ...SYSTEM1, contextWindow: 2_600, maxOutputTokens: 300 };
    const { harness, llm } = build((r) => {
      if (r.agent !== "triage") return reply();
      const seen = lastText(r);
      if (seen.includes(" END")) return reply({ kind: "finish", summary: "Read it all.", evidence: ["act-1"] });
      if (seen.includes("[more: read_record act-1 offset 6000]")) return reply({ kind: "read_record", id: "act-1", offset: 6_000 });
      if (seen.includes("read_record act-1")) return reply({ kind: "read_record", id: "act-1" });
      return reply({ kind: "action", tool: "read_job_logs", args: { job: "j" } });
    }, { specialists: two, toolPort, profiles: { system1: small, system2: SYSTEM2 } });
    const result = await harness.run({ input: "read the log" });
    expect(result.status).toBe("finished");
    const triage = llm.requests.filter((r) => r.agent === "triage").map(lastText);
    // The broadcast preview is short and points at read_record.
    expect(triage.some((t) => t.includes(`ok (${LONG.length} chars): BEGIN`) && t.includes("[cut; read_record act-1]"))).toBe(true);
    // Chunk 1 arrives; the second chunk would overflow the small window, so the next turn compacts
    // and the evidence index in the new snapshot still names act-1.
    const compacted = triage.find((t) => t.startsWith("EPOCH 1"))!;
    expect(compacted).toContain("act-1 read_job_logs");
    expect(triage.some((t) => t.includes(`chars 6000-${LONG.length} of ${LONG.length}`) && t.includes(" END"))).toBe(true);
    // Private: ops never saw the record body.
    expect(llm.requests.filter((r) => r.agent === "ops").some((r) => transcriptText(r).includes("MIDDLE"))).toBe(false);
  });
});

describe("stall detection", () => {
  test("rephrased observations are not progress: the turn stops early", async () => {
    const { harness } = build((_r, i) => reply({ kind: "observation", content: `Still thinking about it, take ${i}.` }), { limits: { maxSteps: 12 } });
    const result = await harness.run({ input: "g" });
    expect(result.status).toBe("no_progress");
    expect(result.steps).toBeLessThanOrEqual(3);
  });
});

describe("media input", () => {
  test("images reach only vision profiles; the executive is consulted before any blind action", async () => {
    const { harness, llm, tools } = build((r) => (r.agent === "executive"
      ? reply({ kind: "observation", type: "focus", content: "The screenshot shows a timeout error.", evidence: [], confidence: "high" })
      : reply({ kind: "action", tool: "restart_job", args: { job: "j" } })),
    { specialists: two, profiles: { system1: SYSTEM1, system2: { ...SYSTEM2, vision: true } }, limits: { maxSteps: 1 } });
    const result = await harness.run({ input: [{ type: "text", text: "what is wrong here?" }, IMAGE] });
    const exec = llm.requests.find((r) => r.agent === "executive")!;
    expect(hasMedia(exec)).toBe(true);
    expect(llm.requests.filter((r) => r.agent !== "executive").some(hasMedia)).toBe(false);
    expect(lastText(llm.requests.find((r) => r.agent === "ops")!)).toContain("Attached media is not visible to your model");
    expect(result.trace.join("\n")).toContain("escalate (media)");
    expect(result.modelCalls.find((c) => c.agent === "executive")!.step).toBe(1);
    expect(tools.calls.length).toBeLessThanOrEqual(1);
  });

  test("unsupported media is rejected up front", async () => {
    const { harness } = build(() => reply());
    await expect(harness.run({ input: [IMAGE] })).rejects.toThrow(UnsupportedContentError);
    await expect(harness.run({ input: [AUDIO] })).rejects.toThrow("needs a LiveModel");
    await expect(harness.run({ input: [{ type: "image", mimeType: "image/png" }] })).rejects.toThrow("needs base64 data or a url");
  });

  const visionExec = { system1: SYSTEM1, system2: { ...SYSTEM2, vision: true } };
  const blindAction = () => reply({ kind: "action", tool: "restart_job", args: { job: "j" } });

  test("executive-only images with no executive budget are a configuration error", async () => {
    const { harness, llm, tools } = build(blindAction, { specialists: two, profiles: visionExec, limits: { maxExecutiveCalls: 0 } });
    await expect(harness.run({ input: [{ type: "text", text: "look" }, IMAGE] })).rejects.toThrow("maxExecutiveCalls is 0");
    expect(llm.requests).toHaveLength(0);
    expect(tools.calls).toHaveLength(0);
  });

  test("executive-only images without enough model budget stop before any action", async () => {
    // Two specialists + the executive do not fit in the model-call budget (one call stays reserved for the narrator).
    const { harness, llm, tools } = build(blindAction, { specialists: two, profiles: visionExec, limits: { maxModelCalls: 3 } });
    const result = await harness.run({ input: [{ type: "text", text: "look" }, IMAGE] });
    expect(result.status).toBe("budget_exhausted");
    expect(llm.requests).toHaveLength(0);
    expect(tools.calls).toHaveLength(0);
  });

  test("if the executive's call fails before it sees the images, nothing is released", async () => {
    const { harness, tools } = build(async (r) => {
      if (r.agent === "executive") throw new Error("503 from provider");
      return blindAction();
    }, { specialists: two, profiles: visionExec, limits: { maxExecutiveCalls: 1 } });
    const result = await harness.run({ input: [{ type: "text", text: "look" }, IMAGE] });
    expect(result.status).toBe("budget_exhausted");
    expect(result.trace.join("\n")).toContain("hold: images not yet examined by the executive");
    expect(result.trace.join("\n")).toContain("no blind action released");
    expect(tools.calls).toHaveLength(0);
  });

  test("a provider's unsupported-content error is surfaced, not swallowed as an empty reply", async () => {
    const { harness } = build(() => reply());
    const failing = build(() => reply(), {
      llm: { complete: async () => { throw new UnsupportedContentError("model x is text-only"); } },
    });
    await expect(failing.harness.run({ input: "hi" })).rejects.toThrow("model x is text-only");
    expect((await harness.run({ input: "hi" })).status).toBe("no_progress");
  });

  test("audio goes through the injected LiveModel into one perception record", async () => {
    // TEST FIXTURE: a LiveModel double that streams fixed observations.
    const live: LiveModel = {
      name: "test-live",
      async *observe(media, focus) {
        for (let i = 1; i <= 20; i++) yield `${media.type} note ${i} (focus: ${focus})`;
      },
    };
    const { harness, llm } = build((r) => (r.agent === "triage" && lastText(r).includes("RECORD per-1")
      ? reply({ kind: "finish", summary: "Heard it.", evidence: ["per-1"] })
      : r.agent === "triage" ? reply({ kind: "read_record", id: "per-1" }) : reply()),
    { live, limits: { maxLiveObservations: 12 } });
    const result = await harness.run({ input: [{ type: "text", text: "listen" }, AUDIO] });
    const perception = result.state.perceptions.get("per-1")!;
    expect(perception.observations).toHaveLength(12);
    expect(result.state.workspace.filter((i) => i.source === "live:test-live")).toHaveLength(1);
    expect(llm.requests.some((r) => lastText(r).includes("audio note 12 (focus: listen)"))).toBe(true);
    // Perception is citable, but it is not a verified tool result.
    expect(result.completion).toBe("answer_only");
  });
});
