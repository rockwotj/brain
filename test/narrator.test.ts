import { expect, test } from "bun:test";
import { ScriptedLlm, roughTokenEstimate } from "../src/llm.ts";
import { InMemoryEventLog, StateStore } from "../src/state.ts";
import { narrate } from "../src/narrator.ts";
import { SYSTEM1 } from "../examples/export-incident.ts";

test("an observation cannot launder an invented evidence ID into a valid citation", async () => {
  const state = new StateStore(new InMemoryEventLog());
  state.apply("user", { type: "goal.set", goal: "Investigate the job" });
  state.admit("specialist", {
    kind: "hypothesis", content: "The job ran out of memory",
    priority: "normal", evidence: ["act-999"],
  });
  const llm = new ScriptedLlm(() => JSON.stringify({
    answer: "The job ran out of memory [act-999].", citations: ["act-999"],
  }));
  const answer = await narrate(llm, SYSTEM1, roughTokenEstimate, state, []);
  expect(answer.grounded).toBe(false);
  expect(answer.unknownCitations).toEqual(["act-999"]);
});
