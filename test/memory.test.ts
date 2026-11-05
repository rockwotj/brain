import { describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FileStorage } from "../src/adapters/file-storage.ts";
import { consolidate, EpisodicStore, FunctionTools, HabitStore, idle, KnowledgeStore, MemoryStorage, type HarnessEvent, type LlmRequest } from "../src/index.ts";
import { GOAL, makeScenario, READ_ONLY, FAST, REASONING, TOOLS } from "./fixtures/export-incident.ts";
import { ScriptedLlm } from "./fixtures/scripted-llm.ts";
import { WordHashEmbedder } from "./fixtures/word-hash-embedder.ts";
import { build, reply } from "./helpers.ts";

describe("consolidation", () => {
  test("writes knowledge only with valid episode provenance and proposes habits as pending", async () => {
    const { harness, llm, episodes, knowledge, habits } = await makeScenario();
    const run = await harness.run({ input: GOAL, constraints: [{ ...READ_ONLY }] });
    const before = (await knowledge.list()).length;
    const report = await consolidate({ llm, profile: REASONING, episodes, knowledge, habits, tools: TOOLS });

    expect(report.replayed).toEqual([run.episode.id]);
    expect(report.learned).toHaveLength(1);
    expect(report.learned[0]!.sources).toEqual([run.episode.id]);
    expect(await knowledge.list()).toHaveLength(before + 1);
    expect(report.rejected.some((r) => r.includes("without valid provenance"))).toBe(true);
    expect(report.habitsProposed[0]!.status).toBe("pending");
    expect((await habits.match("export")).map((h) => h.call.tool)).not.toContain("get_input_stats");
    expect((await episodes.get(run.episode.id))!.consolidated).toBe(true);

    // Nothing left to replay: a second pass is a no-op and makes no model call.
    const calls = llm.requests.length;
    expect((await consolidate({ llm, profile: REASONING, episodes, knowledge, habits, tools: TOOLS })).replayed).toEqual([]);
    expect(llm.requests.length).toBe(calls);
  });

  test("replays surprising episodes first and rejects habits with invalid tool args", async () => {
    const episodes = new EpisodicStore(new MemoryStorage());
    const calm = await episodes.append({ goal: "calm", status: "finished", actions: [] });
    const surprising = await episodes.append({
      goal: "surprising", status: "finished",
      actions: [{ id: "act-1", step: 1, call: { tool: "check_quota", args: { project: "p" } }, source: "t", predicted: { success: true }, actual: { ok: false, error: "boom" }, surprise: true }],
    });
    const llm = new ScriptedLlm(() => JSON.stringify({ lessons: [], habits: [{ trigger: "x", tool: "check_quota", args: { nope: 1 }, episodes: [surprising.id] }] }));
    const storage = new MemoryStorage();
    const report = await consolidate({ llm, profile: REASONING, episodes, knowledge: new KnowledgeStore(storage), habits: new HabitStore(storage), tools: TOOLS, maxEpisodes: 1 });
    expect(report.replayed).toEqual([surprising.id]);
    expect((await episodes.get(calm.id))!.consolidated).toBe(false);
    expect(report.rejected[0]).toContain("unexpected argument nope");
  });

  test("autoApproveReadOnlyHabits approves read-only habits only, and is off by default", async () => {
    const episodes = new EpisodicStore(new MemoryStorage());
    const ep = await episodes.append({ goal: "g", status: "finished", actions: [] });
    const script = () => JSON.stringify({ lessons: [], habits: [
      { trigger: "export", tool: "get_input_stats", args: { job: "j" }, episodes: [ep.id] },
      { trigger: "export", tool: "restart_job", args: { job: "j" }, episodes: [ep.id] },
    ] });
    const habits = new HabitStore(new MemoryStorage());
    const report = await consolidate({ llm: new ScriptedLlm(script), profile: REASONING, episodes, knowledge: new KnowledgeStore(new MemoryStorage()), habits, tools: TOOLS, autoApproveReadOnlyHabits: true });
    expect(report.habitsProposed.map((h) => [h.call.tool, h.status])).toEqual([["get_input_stats", "approved"], ["restart_job", "pending"]]);
  });

  test("knowledge store refuses items without provenance", async () => {
    await expect(new KnowledgeStore(new MemoryStorage()).add("fact", [])).rejects.toThrow();
  });
});

describe("recall via the Embedder port", () => {
  test("recalls consolidated knowledge by cosine similarity, as an unverified claim with sources", async () => {
    const embedder = new WordHashEmbedder();
    const { harness } = build(() => reply(), { embedder, recall: { minSimilarity: 0.2 } });
    await harness.knowledge.add("Deploys on Fridays need a second reviewer.", ["ep-3"]);
    await harness.knowledge.add("Export timeouts often follow input volume spikes.", ["ep-7"]);
    const result = await harness.run({ input: "why did the export timeouts happen" });
    const recalled = result.state.workspace.filter((i) => i.kind === "memory");
    expect(recalled).toHaveLength(1);
    expect(recalled[0]!.content).toBe("Recalled claim (unverified): Export timeouts often follow input volume spikes.");
    expect(recalled[0]!.evidence).toEqual(["kn-2", "ep-7"]);
    // Items were embedded when written, and the goal was embedded as the cue.
    expect(embedder.calls.flat()).toContain("why did the export timeouts happen");
  });

  test("without an embedder nothing is recalled (no keyword fallback)", async () => {
    const { harness } = build(() => reply());
    await harness.knowledge.add("Export timeouts often follow input volume spikes.", ["ep-7"]);
    const result = await harness.run({ input: "why did the export timeouts happen" });
    expect(result.state.workspace.some((i) => i.kind === "memory")).toBe(false);
  });
});

describe("idle background mind", () => {
  test("produces bounded ideas and can never run tools", async () => {
    const tools = new FunctionTools({ restart_job: () => "restarted" });
    const llm = new ScriptedLlm(() => reply(
      { kind: "observation", content: "idea one" },
      { kind: "observation", content: "idea two" },
      { kind: "observation", content: "idea three" },
      { kind: "action", tool: "restart_job", args: { job: "x" } },
    ));
    const storage = new MemoryStorage();
    const report = await idle({ llm, profile: FAST, knowledge: new KnowledgeStore(storage), episodes: new EpisodicStore(storage), maxIdeas: 2 });
    expect(report.ideas).toEqual(["idea one", "idea two"]);
    expect(report.discarded.some((d) => d.includes("actions are not allowed"))).toBe(true);
    expect(report.discarded.some((d) => d.includes("idea three"))).toBe(true);
    expect(tools.calls).toHaveLength(0);
    expect(llm.requests[0]!.messages[0]!.content).toContain("TOOLS: none");
  });

});

describe("session idle", () => {
  /** The idle reflector suggests ideas; every other agent stays silent. */
  const ideaScript = (...ideas: string[]) => (r: LlmRequest) =>
    r.agent === "background" ? reply(...ideas.map((content) => ({ kind: "observation", content }))) : reply();
  const ideasIn = (events: HarnessEvent[]) =>
    events.flatMap((e) => (e.delta.type === "workspace.admit" && e.delta.item.kind === "idea" ? [e.delta.item] : []));

  test("next turn admits pending ideas automatically, at low priority, once", async () => {
    const { harness, llm } = build(ideaScript("check volume first"));
    const session = await harness.openSession("s-idle");
    await session.turn({ input: "why did the export fail" });
    const report = await session.idle();
    expect(report.ideas).toEqual(["check volume first"]);
    expect(await session.pendingIdeas()).toEqual(["check volume first"]);
    const idleRequest = llm.requests.find((r) => r.agent === "background")!;
    expect(JSON.stringify(idleRequest.messages)).toContain("Most recent goal: why did the export fail");

    const second = await session.turn({ input: "did it finish tonight" });
    const admitted = ideasIn(second.events);
    expect(admitted.map((i) => [i.content, i.priority])).toEqual([["check volume first", "low"]]);
    expect(await session.pendingIdeas()).toEqual([]);

    const third = await session.turn({ input: "anything else" });
    expect(ideasIn(third.events)).toEqual([]);
  });

  test("pending ideas survive reopening the session from file storage", async () => {
    const dir = await mkdtemp(join(tmpdir(), "brain-idle-"));
    try {
      const first = build(ideaScript("check volume first"), { storage: new FileStorage(dir) });
      await (await first.harness.openSession("s-file")).idle();

      const second = build(() => reply(), { storage: new FileStorage(dir) });
      const reopened = await second.harness.openSession("s-file");
      expect(await reopened.pendingIdeas()).toEqual(["check volume first"]);
      const result = await reopened.turn({ input: "did it finish" });
      expect(ideasIn(result.events).map((i) => i.content)).toEqual(["check volume first"]);
      expect(await reopened.pendingIdeas()).toEqual([]);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("ideas stay in the session that prepared them", async () => {
    const { harness } = build(ideaScript("check volume first"));
    const a = await harness.openSession("a");
    const b = await harness.openSession("b");
    await a.idle();
    expect(ideasIn((await b.turn({ input: "other work" })).events)).toEqual([]);
    expect(ideasIn((await harness.run({ input: "one-shot" })).events)).toEqual([]);
    expect(await a.pendingIdeas()).toEqual(["check volume first"]);
  });

  test("newTask discards ideas prepared for the previous task", async () => {
    const { harness } = build(ideaScript("check volume first"));
    const session = await harness.openSession("s-new");
    await session.turn({ input: "why did the export fail" });
    await session.idle();
    const result = await session.turn({ input: "different question", newTask: true });
    expect(ideasIn(result.events)).toEqual([]);
    expect(await session.pendingIdeas()).toEqual([]);
  });

  test("a later idle replaces earlier pending ideas", async () => {
    let round = 0;
    const { harness } = build((r) => (r.agent === "background" ? reply({ kind: "observation", content: `idea ${++round}` }) : reply()));
    const session = await harness.openSession("s-replace");
    await session.idle();
    await session.idle();
    expect(await session.pendingIdeas()).toEqual(["idea 2"]);
  });
});
