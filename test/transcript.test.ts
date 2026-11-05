import { describe, expect, test } from "bun:test";
import { AgentTranscript, ContextBudgetError, estimateMessages, roughTokenEstimate, StateStore, type ActionRecord, type Message } from "../src/index.ts";
import { isPrefix } from "./helpers.ts";

const SYSTEM = "You are a test agent. Reply with JSON.";
const text = (m: Message | undefined) => (typeof m?.content === "string" ? m.content : JSON.stringify(m?.content));
const record = (id: string, output: string): ActionRecord => ({
  id, step: 1, call: { tool: "read_file", args: { path: "big.txt" } }, source: "a", predicted: { success: true }, actual: { ok: true, output }, surprise: false,
});

describe("AgentTranscript", () => {
  test("keeps an exact, growing prefix while workspace and global deltas arrive", async () => {
    const state = new StateStore();
    await state.apply("user", { type: "goal.set", goal: "find the cause" });
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 100_000, reserveOutput: 1_000, estimate: roughTokenEstimate });

    const turns = [];
    for (let i = 0; i < 5; i++) {
      await state.admit("someone", { kind: "observation", content: `finding number ${i}`, priority: "normal", evidence: [] });
      if (i === 2) await state.apply("user", { type: "constraint.add", constraint: { id: "c-9", kind: "instruction", text: "be brief" } });
      if (i === 3) await state.apply("monitor", { type: "mode.set", mode: { urgency: "high", carefulSteps: 0 } });
      const turn = t.prepare(state, "go");
      t.recordReply(`reply ${i}`);
      turns.push(turn);
    }
    expect(turns[0]?.messages[0]).toEqual({ role: "system", content: SYSTEM });
    for (let i = 1; i < turns.length; i++) {
      const prev = turns[i - 1]!;
      const cur = turns[i]!;
      expect(isPrefix(prev.messages, cur.messages)).toBe(true);
      expect(cur.messages[prev.messages.length]).toEqual({ role: "assistant", content: `reply ${i - 1}` });
      expect(cur.stablePrefixLength).toBe(prev.messages.length + 1);
      expect(cur.compacted).toBe(false);
    }
    // Later deltas arrive as appended updates, not as edits of the snapshot.
    expect(text(turns[3]!.messages.at(-1))).toContain("UPDATES since seq");
    expect(text(turns[3]!.messages.at(-1))).toContain("mode: urgency=high");
    expect(text(turns[2]!.messages.at(-1))).toContain("constraint added: c-9");
    expect(text(turns[0]!.messages[1])).not.toContain("c-9");
  });

  test("compacts just before overflow into a fresh snapshot that keeps the evidence index", async () => {
    const state = new StateStore({ budget: { maxItems: 3, maxItemChars: 200 } });
    await state.apply("user", { type: "goal.set", goal: "old goal" });
    await state.apply("user", { type: "constraint.add", constraint: { id: "c-1", kind: "read_only", reason: "audit" } });
    await state.apply("user", { type: "constraint.add", constraint: { id: "c-2", kind: "deny_tool", tool: "restart_job", reason: "temporary" } });
    await state.apply("actuator", { type: "action.outcome", record: record("act-1", "x".repeat(5_000)) });
    const options = { contextWindow: 900, reserveOutput: 200, estimate: roughTokenEstimate, tailReplies: 1 };
    const t = new AgentTranscript("a", SYSTEM, options);

    let compactedAt = -1;
    let previous: Message[] = [];
    for (let i = 0; i < 30 && compactedAt < 0; i++) {
      await state.admit("worker", { kind: "observation", content: `observation ${i} `.repeat(6), priority: "normal", evidence: [] });
      if (i === 4) await state.apply("user", { type: "constraint.remove", id: "c-2" });
      if (i === 5) await state.apply("user", { type: "goal.set", goal: "new goal" });
      const turn = t.prepare(state, "go");
      expect(estimateMessages(turn.messages, roughTokenEstimate) + options.reserveOutput).toBeLessThanOrEqual(options.contextWindow);
      if (turn.compacted) compactedAt = i;
      else if (previous.length) expect(isPrefix(previous, turn.messages)).toBe(true);
      t.recordReply(`reply ${i}`);
      previous = [...turn.messages, { role: "assistant", content: `reply ${i}` }];
      if (turn.compacted) {
        expect(turn.compacted).toBe("overflow");
        expect(turn.messages).toHaveLength(2);
        expect(turn.stablePrefixLength).toBe(1);
        const snapshot = text(turn.messages[1]);
        expect(snapshot).toContain("EPOCH 1");
        expect(snapshot).toContain(`authoritative as of seq ${state.seq}`);
        expect(snapshot).toContain("GOAL: new goal"); // latest state, not the epoch-0 base
        expect(snapshot).toContain("c-1 READ-ONLY"); // pinned constraint survives
        expect(snapshot).not.toContain("c-2"); // removed constraint is not resurrected
        expect(snapshot).not.toContain("observation 0 "); // evicted item is gone
        expect(snapshot).toContain(`reply ${i - 1}`); // short tail of own replies
        // The evidence index survives compaction: id, call, size and a pointer to read_record.
        expect(snapshot).toContain('act-1 read_file({"path":"big.txt"}) ok (5000 chars)');
        expect(snapshot).toContain("[cut; read_record act-1]");
      }
    }
    expect(compactedAt).toBeGreaterThan(0);
    await state.admit("worker", { kind: "observation", content: "post compaction", priority: "high", evidence: [] });
    const next = t.prepare(state, "go");
    expect(isPrefix(previous, next.messages)).toBe(true);
    expect(text(next.messages.at(-1))).toContain("post compaction");
  });

  test("compacts on a provider cache-expiry hint only once the transcript is big enough", async () => {
    const state = new StateStore();
    await state.apply("user", { type: "goal.set", goal: "g" });
    let now = 1_000;
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 100_000, reserveOutput: 1_000, estimate: roughTokenEstimate, clock: () => now, compactAfterExpiryTokens: 200 });
    t.prepare(state, "go");
    t.recordReply("small");
    t.noteCacheExpiry(1_500);
    now = 2_000;
    expect(t.prepare(state, "go").compacted).toBe(false); // expired, but too small to bother
    t.recordReply("x".repeat(1_000));
    t.noteCacheExpiry(2_500);
    expect(t.prepare(state, "go").compacted).toBe(false); // big, but not expired yet
    t.recordReply("ok");
    t.noteCacheExpiry(2_500);
    now = 3_000;
    const turn = t.prepare(state, "go");
    expect(turn.compacted).toBe("cache_expired");
    expect(turn.messages).toHaveLength(2);
  });

  test("an oversized snapshot throws without changing the transcript, which then recovers", async () => {
    const state = new StateStore();
    await state.apply("user", { type: "goal.set", goal: "small" });
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 400, reserveOutput: 100, estimate: roughTokenEstimate });
    t.prepare(state, "go");
    t.recordReply("ok");
    await state.apply("user", { type: "goal.set", goal: "x".repeat(2_000) });
    expect(() => t.prepare(state, "go")).toThrow(ContextBudgetError);
    expect(t.length).toBe(3); // untouched
    await state.apply("user", { type: "goal.set", goal: "small again" });
    const turn = t.prepare(state, "go"); // the old huge delta does not fit, so it compacts to the current state
    expect(turn.compacted).toBe("overflow");
    expect(text(turn.messages[1])).toContain("GOAL: small again");
  });

  test("shows only events visible to this agent, plus its private notes", async () => {
    const state = new StateStore();
    await state.apply("user", { type: "goal.set", goal: "g" });
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 100_000, reserveOutput: 1_000, estimate: roughTokenEstimate });
    t.prepare(state, "go");
    t.recordReply("ok");
    await state.apply("x", { type: "note", text: "for b only" }, ["b"]);
    await state.apply("x", { type: "note", text: "for a only" }, ["a"]);
    await state.apply("x", { type: "note", text: "for everyone" });
    const last = text(t.prepare(state, "go", ["RECORD act-1 ..."]).messages.at(-1));
    expect(last).not.toContain("for b only");
    expect(last).toContain("(to a only) note: for a only");
    expect(last).toContain("for everyone");
    expect(last).toContain("PRIVATE TO YOU (not broadcast):\n\nRECORD act-1");
  });

  test("tailReplies: 0 carries no earlier replies into a new epoch", async () => {
    const state = new StateStore();
    await state.apply("user", { type: "goal.set", goal: "g" });
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 260, reserveOutput: 50, estimate: roughTokenEstimate, tailReplies: 0 });
    let compacted;
    for (let i = 0; i < 20 && !compacted; i++) {
      await state.admit("w", { kind: "observation", content: `item ${i} `.repeat(5), priority: "high", evidence: [] });
      const turn = t.prepare(state, "go");
      t.recordReply(`secret reply ${i}`);
      if (turn.compacted) compacted = turn;
    }
    expect(compacted).toBeDefined();
    expect(text(compacted!.messages[1])).not.toContain("secret reply");
  });

  test("uses the injected token estimator", async () => {
    const state = new StateStore();
    await state.apply("user", { type: "goal.set", goal: "g" });
    let calls = 0;
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 10_000, reserveOutput: 10, estimate: (s) => (calls++, s.length) });
    t.prepare(state, "go");
    expect(calls).toBeGreaterThan(0);
  });
});
