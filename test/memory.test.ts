import { describe, expect, test } from "bun:test";
import { consolidate, FunctionTools, idle, InMemoryEpisodicStore, InMemoryHabitStore, InMemoryKnowledgeStore, ScriptedLlm } from "../src/index.ts";
import { GOAL, makeScenario, READ_ONLY, SYSTEM1, SYSTEM2, TOOLS } from "../examples/export-incident.ts";
import { build, reply } from "./helpers.ts";

describe("consolidation", () => {
  test("writes knowledge only with valid episode provenance and proposes habits as pending", async () => {
    const { harness, llm, episodes, knowledge, habits } = makeScenario();
    const run = await harness.run({ goal: GOAL, constraints: [{ ...READ_ONLY }] });
    const before = knowledge.list().length;
    const report = await consolidate({ llm, profile: SYSTEM2, episodes, knowledge, habits, tools: TOOLS });

    expect(report.replayed).toEqual([run.episode.id]);
    expect(report.learned).toHaveLength(1);
    expect(report.learned[0]!.sources).toEqual([run.episode.id]);
    expect(knowledge.list()).toHaveLength(before + 1);
    expect(report.rejected.some((r) => r.includes("without valid provenance"))).toBe(true);
    expect(report.habitsProposed[0]!.status).toBe("pending");
    expect(habits.match("export").map((h) => h.call.tool)).not.toContain("get_input_stats");
    expect(episodes.get(run.episode.id)!.consolidated).toBe(true);

    // Nothing left to replay: a second pass is a no-op and makes no model call.
    const calls = llm.requests.length;
    expect((await consolidate({ llm, profile: SYSTEM2, episodes, knowledge, habits, tools: TOOLS })).replayed).toEqual([]);
    expect(llm.requests.length).toBe(calls);
  });

  test("replays surprising episodes first and rejects habits with invalid tool args", async () => {
    const episodes = new InMemoryEpisodicStore();
    const calm = episodes.append({ goal: "calm", status: "finished", actions: [] });
    const surprising = episodes.append({
      goal: "surprising", status: "finished",
      actions: [{ id: "act-1", step: 1, call: { tool: "check_quota", args: { project: "p" } }, source: "t", predicted: { success: true }, actual: { ok: false, error: "boom" }, surprise: true }],
    });
    const llm = new ScriptedLlm(() => JSON.stringify({ lessons: [], habits: [{ trigger: "x", tool: "check_quota", args: { nope: 1 }, episodes: [surprising.id] }] }));
    const report = await consolidate({ llm, profile: SYSTEM2, episodes, knowledge: new InMemoryKnowledgeStore(), habits: new InMemoryHabitStore(), tools: TOOLS, maxEpisodes: 1 });
    expect(report.replayed).toEqual([surprising.id]);
    expect(episodes.get(calm.id)!.consolidated).toBe(false);
    expect(report.rejected[0]).toContain("unexpected argument nope");
  });

  test("knowledge store refuses items without provenance", () => {
    expect(() => new InMemoryKnowledgeStore().add("fact", [])).toThrow();
  });

  test("consolidated knowledge is recalled in a later run as an unverified claim with sources", async () => {
    const { harness } = build(() => reply());
    harness.knowledge.add("Export timeouts often follow input volume spikes.", ["ep-7"]);
    const result = await harness.run({ goal: "why did the export timeouts happen" });
    const memory = result.events.find((e) => e.delta.type === "workspace.admit" && e.delta.item.kind === "memory");
    expect(memory?.delta.type === "workspace.admit" && memory.delta.item.content).toContain("unverified");
    expect(memory?.delta.type === "workspace.admit" && memory.delta.item.evidence).toEqual(["kn-1", "ep-7"]);
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
    const report = await idle({ llm, profile: SYSTEM1, knowledge: new InMemoryKnowledgeStore(), episodes: new InMemoryEpisodicStore(), maxIdeas: 2 });
    expect(report.ideas).toEqual(["idea one", "idea two"]);
    expect(report.discarded.some((d) => d.includes("actions are not allowed"))).toBe(true);
    expect(report.discarded.some((d) => d.includes("idea three"))).toBe(true);
    expect(tools.calls).toHaveLength(0);
    expect(llm.requests[0]!.messages[0]!.content).toContain("TOOLS: none");
  });

  test("prepared ideas enter a later run at low priority", async () => {
    const { harness } = build(() => reply());
    const result = await harness.run({ goal: "g", preparedIdeas: ["check volume first"] });
    expect(result.state.workspace.find((i) => i.kind === "idea")?.priority).toBe("low");
  });
});
