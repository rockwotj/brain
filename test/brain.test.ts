import { expect, test } from "bun:test";
import { Brain } from "../src";
import { HashEmbedder } from "../src/adapters/hash-embedder";
import { MemoryStorage } from "../src/io/storage";
import type { Tool } from "../src/io/tools";
import { calculator, toyLive, toySystem1, toySystem2 } from "../examples/toy-models";

async function setup(tools: Tool[] = [calculator]) {
  const said: string[] = [];
  const system1 = toySystem1();
  const system2 = toySystem2();
  const brain = await Brain.create({
    models: { system1, system2, live: toyLive, embedder: new HashEmbedder() },
    storage: new MemoryStorage(),
    tools,
    onSay: (t) => said.push(t),
  });
  const ask = async (question: string) => {
    await brain.hear({ source: "user", parts: [{ type: "text", text: question }] });
    return brain.run();
  };
  return { brain, said, system1, system2, ask };
}

test("novel input takes the slow path; expected results stay private", async () => {
  const { said, ask } = await setup();
  const reports = await ask("What is 17 * 23?");
  expect(said).toEqual(["The answer is 391."]);
  expect(reports[0]!.events[0]!.route).toBe("workspace");
  expect(reports[1]!.events[0]!.route).toBe("feedback");
  expect(reports[1]!.woke).toEqual(["planner"]);
});

test("sleep compiles a skill; the habit then runs without System 2", async () => {
  const { brain, said, system2, ask } = await setup();
  await ask("What is 17 * 23?");
  expect((await brain.idle()).skills).toBe(1);

  const before = system2.calls.length;
  const reports = await ask("What is 12 * 11?");
  expect(said.at(-1)).toBe("The answer is 132.");
  expect(reports[0]!.events[0]!.route).toBe("reflex");
  expect(system2.calls.length).toBe(before);
});

test("audio goes through the live model", async () => {
  const { brain, said } = await setup();
  await brain.hear({ source: "user", parts: [{ type: "audio", mimeType: "audio/wav", data: new TextEncoder().encode("What is 6 * 7?") }] });
  await brain.run();
  expect(said).toEqual(["The answer is 42."]);
});

test("a surprising result breaks the habit and the slow path takes over", async () => {
  const { brain, said, ask } = await setup();
  await ask("What is 17 * 23?");
  await brain.idle();
  const reports = await ask("What is 999 * 999?");
  expect(reports[0]!.events[0]!.route).toBe("reflex");
  expect(reports[1]!.events[0]!.route).toBe("workspace");
  expect(reports[1]!.woke).toContain("planner");
  expect(said.at(-1)).toContain("calculator choked");
});

test("the planner's prompt prefix grows append-only across ticks", async () => {
  const { system2, ask } = await setup();
  await ask("What is 17 * 23?");
  await ask("What is 2 * 3?");
  const plannerCalls = system2.calls.filter((c) => c.system.includes('You are the "planner"'));
  expect(plannerCalls.length).toBeGreaterThan(2);
  for (let i = 1; i < plannerCalls.length; i++) {
    const prev = plannerCalls[i - 1]!;
    const next = plannerCalls[i]!;
    expect(next.system).toBe(prev.system);
    expect(next.messages.slice(0, prev.messages.length)).toEqual(prev.messages);
  }
});

test("goal changes reach loops as deltas, not prefix rewrites", async () => {
  const { brain, system2, ask } = await setup();
  await ask("What is 17 * 23?");
  const planner = () => system2.calls.filter((c) => c.system.includes('You are the "planner"'));
  const before = planner().length;
  const systemBefore = planner().at(-1)!.system;

  await brain.executive.push("help the user with math homework");
  await ask("What is 2 * 3?");

  const next = planner()[before]!;
  expect(next.system).toBe(systemBefore);
  expect(JSON.stringify(next.messages.at(-1))).toContain("goal g1: help the user with math homework");
});
