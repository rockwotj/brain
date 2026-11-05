import { expect, test } from "bun:test";
import { calm } from "../src/core/modulators";
import { Selector } from "../src/core/selector";
import type { Proposal } from "../src/core/types";
import { MemoryStorage } from "../src/io/storage";
import type { Tool } from "../src/io/tools";

const tool = (name: string, effector: string): Tool => ({ name, description: "", parameters: {}, effector, risk: "low", run: async () => "" });
const tools = new Map([tool("say", "voice"), tool("shout", "voice"), tool("type", "keyboard")].map((t) => [t.name, t]));
const proposal = (id: string, toolName: string, confidence = 0.8, extra: Partial<Proposal> = {}): Proposal => ({
  id, source: "test", tool: toolName, args: { id }, expect: "", confidence, ...extra,
});

test("releases at most one action per effector", () => {
  const selector = new Selector(new MemoryStorage());
  const { chosen, vetoed } = selector.select([proposal("1", "say", 0.9), proposal("2", "shout", 0.6), proposal("3", "type")], tools, calm, 1);
  expect(chosen.map((p) => p.id).sort()).toEqual(["1", "3"]);
  expect(vetoed.map((v) => v.proposal.id)).toEqual(["2"]);
});

test("default is no-go: weak and high-risk proposals are blocked", () => {
  const selector = new Selector(new MemoryStorage());
  const { chosen, vetoed } = selector.select([proposal("1", "say", 0), proposal("2", "type", 0.9, { risk: "high" })], tools, calm, 1);
  expect(chosen).toEqual([]);
  expect(vetoed.map((v) => v.reason)).toEqual(["high risk: needs approval", "below threshold"]);
});

test("the exact same action can't fire again right away", () => {
  const selector = new Selector(new MemoryStorage());
  expect(selector.select([proposal("1", "say")], tools, calm, 1).chosen).toHaveLength(1);
  expect(selector.select([proposal("1", "say")], tools, calm, 2).vetoed[0]!.reason).toContain("refractory");
  expect(selector.select([proposal("1", "say")], tools, calm, 5).chosen).toHaveLength(1);
});

test("learns tool values from reward prediction errors", async () => {
  const storage = new MemoryStorage();
  const selector = new Selector(storage);
  expect(await selector.learn("say", 1)).toBeCloseTo(0.5);
  expect(selector.value("say")).toBeCloseTo(0.65);
  const reloaded = new Selector(storage);
  await reloaded.load();
  expect(reloaded.value("say")).toBeCloseTo(0.65);
});
