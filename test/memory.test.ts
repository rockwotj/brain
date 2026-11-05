import { describe, expect, test } from "bun:test";
import { consolidate, EpisodicStore, FunctionTools, HabitStore, idle, KnowledgeStore, MemoryStorage } from "../src/index.ts";
import { GOAL, makeScenario, READ_ONLY, SYSTEM1, SYSTEM2, TOOLS } from "./fixtures/export-incident.ts";
import { ScriptedLlm } from "./fixtures/scripted-llm.ts";
import { WordHashEmbedder } from "./fixtures/word-hash-embedder.ts";
import { build, reply } from "./helpers.ts";

describe("consolidation", () => {
  test("writes knowledge only with valid episode provenance and proposes habits as pending", async () => {
    const { harness, llm, episodes, knowledge, habits } = await makeScenario();
    const run = await harness.run({ input: GOAL, constraints: [{ ...READ_ONLY }] });
    const before = (await knowledge.list()).length;
    const report = await consolidate({ llm, profile: SYSTEM2, episodes, knowledge, habits, tools: TOOLS });

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
    expect((await consolidate({ llm, profile: SYSTEM2, episodes, knowledge, habits, tools: TOOLS })).replayed).toEqual([]);
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
    const report = await consolidate({ llm, profile: SYSTEM2, episodes, knowledge: new KnowledgeStore(storage), habits: new HabitStore(storage), tools: TOOLS, maxEpisodes: 1 });
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
    const report = await consolidate({ llm: new ScriptedLlm(script), profile: SYSTEM2, episodes, knowledge: new KnowledgeStore(new MemoryStorage()), habits, tools: TOOLS, autoApproveReadOnlyHabits: true });
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
    const report = await idle({ llm, profile: SYSTEM1, knowledge: new KnowledgeStore(storage), episodes: new EpisodicStore(storage), maxIdeas: 2 });
    expect(report.ideas).toEqual(["idea one", "idea two"]);
    expect(report.discarded.some((d) => d.includes("actions are not allowed"))).toBe(true);
    expect(report.discarded.some((d) => d.includes("idea three"))).toBe(true);
    expect(tools.calls).toHaveLength(0);
    expect(llm.requests[0]!.messages[0]!.content).toContain("TOOLS: none");
  });

  test("prepared ideas enter a later run at low priority", async () => {
    const { harness } = build(() => reply());
    const result = await harness.run({ input: "g", preparedIdeas: ["check volume first"] });
    expect(result.state.workspace.find((i) => i.kind === "idea")?.priority).toBe("low");
  });
});
