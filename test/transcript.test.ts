import { describe, expect, test } from "bun:test";
import { AgentTranscript, ContextBudgetError, estimateMessages, roughTokenEstimate, StateStore, type Message } from "../src/index.ts";
import { isPrefix } from "./helpers.ts";

const SYSTEM = "You are a test agent. Reply with JSON.";

describe("AgentTranscript", () => {
  test("keeps an exact, growing prefix while workspace and global deltas arrive", () => {
    const state = new StateStore();
    state.apply("user", { type: "goal.set", goal: "find the cause" });
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 100_000, reserveOutput: 1_000, estimate: roughTokenEstimate });

    const turns = [];
    for (let i = 0; i < 5; i++) {
      state.admit("someone", { kind: "observation", content: `finding number ${i}`, priority: "normal", evidence: [] });
      if (i === 2) state.apply("user", { type: "constraint.add", constraint: { id: "c-9", kind: "instruction", text: "be brief" } });
      if (i === 3) state.apply("monitor", { type: "mode.set", mode: { urgency: "high", carefulSteps: 0 } });
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
    const fourth = turns[3]!.messages.at(-1)!.content;
    expect(fourth).toContain("UPDATES since seq");
    expect(fourth).toContain("mode: urgency=high");
    expect(turns[2]!.messages.at(-1)!.content).toContain("constraint added: c-9");
    expect(turns[0]!.messages[1]!.content).not.toContain("c-9");
  });

  test("compacts just before the limit, folding deltas into a fresh authoritative snapshot", () => {
    const state = new StateStore(undefined, { maxItems: 3, maxItemChars: 200 });
    state.apply("user", { type: "goal.set", goal: "old goal" });
    state.apply("user", { type: "constraint.add", constraint: { id: "c-1", kind: "read_only", reason: "audit" } });
    state.apply("user", { type: "constraint.add", constraint: { id: "c-2", kind: "deny_tool", tool: "restart_job", reason: "temporary" } });
    const options = { contextWindow: 700, reserveOutput: 200, estimate: roughTokenEstimate, tailReplies: 1 };
    const t = new AgentTranscript("a", SYSTEM, options);

    let compactedAt = -1;
    let previous: Message[] = [];
    for (let i = 0; i < 30 && compactedAt < 0; i++) {
      state.admit("worker", { kind: "observation", content: `observation ${i} `.repeat(6), priority: "normal", evidence: [] });
      if (i === 4) state.apply("user", { type: "constraint.remove", id: "c-2" });
      if (i === 5) state.apply("user", { type: "goal.set", goal: "new goal" });
      const turn = t.prepare(state, "go");
      // Every request leaves room for the output reserve.
      expect(estimateMessages(turn.messages, roughTokenEstimate) + options.reserveOutput).toBeLessThanOrEqual(options.contextWindow);
      if (turn.compacted) compactedAt = i;
      else if (previous.length) expect(isPrefix(previous, turn.messages)).toBe(true);
      t.recordReply(`reply ${i}`);
      previous = [...turn.messages, { role: "assistant", content: `reply ${i}` }];
      if (turn.compacted) {
        expect(turn.messages).toHaveLength(2);
        expect(turn.messages[0]).toEqual({ role: "system", content: SYSTEM });
        expect(turn.stablePrefixLength).toBe(1);
        const snapshot = turn.messages[1]!.content;
        expect(snapshot).toContain("EPOCH 1");
        expect(snapshot).toContain(`authoritative as of seq ${state.seq}`);
        expect(snapshot).toContain("GOAL: new goal"); // latest state, not the epoch-0 base
        expect(snapshot).toContain("c-1 READ-ONLY"); // pinned constraint survives
        expect(snapshot).not.toContain("c-2"); // removed constraint is not resurrected
        for (const item of state.workspace) expect(snapshot).toContain(item.id);
        expect(snapshot).not.toContain("observation 0 "); // evicted item is gone
        expect(snapshot).toContain(`reply ${i - 1}`); // short tail of own replies
      }
    }
    expect(compactedAt).toBeGreaterThan(0);
    // After compaction, the new epoch is again append-only.
    state.admit("worker", { kind: "observation", content: "post compaction", priority: "high", evidence: [] });
    const next = t.prepare(state, "go");
    expect(isPrefix(previous, next.messages)).toBe(true);
    expect(next.messages.at(-1)!.content).toContain("post compaction");
  });

  test("throws instead of silently overflowing when even a snapshot does not fit", () => {
    const state = new StateStore();
    state.apply("user", { type: "goal.set", goal: "x".repeat(2_000) });
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 300, reserveOutput: 100, estimate: roughTokenEstimate });
    expect(() => t.prepare(state, "go")).toThrow(ContextBudgetError);
  });

  test("tailReplies: 0 carries no earlier replies into a new epoch", () => {
    const state = new StateStore();
    state.apply("user", { type: "goal.set", goal: "g" });
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 260, reserveOutput: 50, estimate: roughTokenEstimate, tailReplies: 0 });
    let compacted;
    for (let i = 0; i < 20 && !compacted; i++) {
      state.admit("w", { kind: "observation", content: `item ${i} `.repeat(5), priority: "high", evidence: [] });
      const turn = t.prepare(state, "go");
      t.recordReply(`secret reply ${i}`);
      if (turn.compacted) compacted = turn;
    }
    expect(compacted).toBeDefined();
    expect(compacted!.messages[1]!.content).not.toContain("secret reply");
    expect(compacted!.messages[1]!.content).not.toContain("YOUR RECENT REPLIES");
  });

  test("uses the injected token estimator", () => {
    const state = new StateStore();
    state.apply("user", { type: "goal.set", goal: "g" });
    let calls = 0;
    const t = new AgentTranscript("a", SYSTEM, { contextWindow: 10_000, reserveOutput: 10, estimate: (s) => (calls++, s.length) });
    t.prepare(state, "go");
    expect(calls).toBeGreaterThan(0);
  });
});
