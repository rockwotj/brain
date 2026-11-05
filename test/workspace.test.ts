import { expect, test } from "bun:test";
import { Workspace } from "../src/core/workspace";

test("keeps only the top bids and reports what entered", () => {
  const ws = new Workspace(2);
  ws.submit({ kind: "thought", source: "a", content: "low", salience: 0.1 });
  ws.submit({ kind: "thought", source: "b", content: "high", salience: 0.9 });
  ws.submit({ kind: "thought", source: "c", content: "mid", salience: 0.5 });
  const entered = ws.compete(() => 1);
  expect(entered.map((b) => b.content)).toEqual(["high", "mid"]);
});

test("old items fade so new ones can take over", () => {
  const ws = new Workspace(1);
  ws.submit({ kind: "thought", source: "a", content: "old", salience: 0.6 });
  ws.compete(() => 1);
  ws.submit({ kind: "thought", source: "b", content: "new", salience: 0.5 });
  expect(ws.compete(() => 1).map((b) => b.content)).toEqual(["new"]);
});

test("contradicting bids inhibit the weaker one", () => {
  const ws = new Workspace(2);
  const claim = ws.submit({ kind: "thought", source: "planner", content: "plan A", salience: 0.6 });
  ws.submit({ kind: "thought", source: "critic", content: "plan A is wrong", salience: 0.5, contradicts: [claim.id] });
  ws.submit({ kind: "thought", source: "user-model", content: "user is in a hurry", salience: 0.4 });
  // Without inhibition the top two would be "plan A" and the objection; the weaker side gets halved.
  expect(ws.compete(() => 1).map((b) => b.content)).toEqual(["plan A", "user is in a hurry"]);
});

test("one source can't flood the stage", () => {
  const ws = new Workspace(7, 2);
  for (let i = 0; i < 5; i++) ws.submit({ kind: "thought", source: "chatty", content: `t${i}`, salience: 0.5 });
  expect(ws.compete(() => 1)).toHaveLength(2);
});
