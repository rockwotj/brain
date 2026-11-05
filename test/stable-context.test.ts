import { expect, test } from "bun:test";
import { StableContext } from "../src/context/stable-context";
import { text } from "../src/io/content";

function makeContext(clock: { now: number }, maxTokens = 10_000) {
  return new StableContext({ instructions: "You are a test loop.", facts: { "goal g1": "ship it" }, maxTokens, cacheTtlMs: 60_000, clock: () => clock.now });
}

test("fact changes become deltas; the sent prefix never changes", () => {
  const clock = { now: 0 };
  const ctx = makeContext(clock);
  ctx.addUser([text("hello")]);
  ctx.addAssistant("{}");
  ctx.markSent();
  const first = ctx.request();

  ctx.setFact("goal g1", "ship it today");
  ctx.addUser([text("next")]);
  const second = ctx.request();

  expect(second.system).toBe(first.system);
  expect(second.messages.slice(0, first.messages.length)).toEqual(first.messages);
  expect(second.cacheUpTo).toBe(2);
  const delta = second.messages.at(-1)!.content[0]!;
  expect(delta.type === "text" && delta.text).toContain("goal g1 is now: ship it today");
});

test("compacts just before overflow and folds deltas into the new prefix", async () => {
  const clock = { now: 0 };
  const ctx = makeContext(clock, 200);
  ctx.setFact("user", "prefers short answers");
  for (let i = 0; i < 10; i++) {
    ctx.addUser([text("x".repeat(60))]);
    ctx.addAssistant("{}");
  }
  expect(ctx.needsCompaction()).toBe("overflow");

  await ctx.compact(async () => "did ten things");
  const req = ctx.request();
  expect(ctx.prefixBreaks).toBe(1);
  expect(req.messages).toEqual([]);
  expect(req.system).toContain("- user: prefers short answers");
  expect(req.system).toContain("did ten things");
});

test("compacts for free once the cache has expired", () => {
  const clock = { now: 0 };
  const ctx = makeContext(clock);
  ctx.addUser([text("hi")]);
  ctx.addAssistant("{}");
  ctx.markSent();
  ctx.setFact("goal g2", "new goal");

  expect(ctx.needsCompaction()).toBeUndefined(); // cache still warm: keep appending deltas
  clock.now = 61_000;
  expect(ctx.needsCompaction()).toBe("cache-expired");
});
