import { expect, test } from "bun:test";
import { makeNarrator, narrate, roughTokenEstimate, StateStore } from "../src/index.ts";
import { FAST } from "./fixtures/export-incident.ts";
import { ScriptedLlm } from "./fixtures/scripted-llm.ts";

test("an observation cannot launder an invented evidence ID into a valid citation", async () => {
  const state = new StateStore();
  await state.apply("user", { type: "goal.set", goal: "Investigate the job" });
  await state.admit("specialist", { kind: "hypothesis", content: "The job ran out of memory", priority: "normal", evidence: ["act-999"] });
  const llm = new ScriptedLlm(() => JSON.stringify({ answer: "The job ran out of memory [act-999].", citations: ["act-999"] }));
  const answer = await narrate(llm, makeNarrator(FAST, roughTokenEstimate), state, [], 2_000);
  expect(answer.grounded).toBe(false);
  expect(answer.unknownCitations).toEqual(["act-999"]);
});
