/** Deterministic end-to-end demo: bun examples/demo.ts */
import { consolidate, idle } from "../src/index.ts";
import { GOAL, makeScenario, READ_ONLY, SYSTEM1, SYSTEM2, TOOLS } from "./export-incident.ts";

const { llm, tools, harness, episodes, knowledge, habits } = makeScenario();

console.log(`GOAL: ${GOAL}\nCONSTRAINT: ${READ_ONLY.id} read-only\n`);
const result = await harness.run({ goal: GOAL, constraints: [{ ...READ_ONLY }] });
console.log("TRACE");
for (const line of result.trace) console.log(line);

console.log(`\nSTATUS: ${result.status} in ${result.steps} steps`);
console.log(`ANSWER: ${result.answer.text}`);
console.log(`  citations=${result.answer.citations.join(", ")} grounded=${result.answer.grounded}`);

const byTier = (tier: string) => result.modelCalls.filter((c) => c.tier === tier).length;
console.log(`\nMODEL CALLS: ${result.modelCalls.length} (system1=${byTier("system1")}, system2=${byTier("system2")})`);
console.log(`TOOLS EXECUTED: ${tools.calls.map((c) => c.tool).join(", ")}`);
const u = result.usage;
console.log(`TOKENS (estimated): input=${u.inputTokens}, prefix shared with previous call=${u.cachedInputTokens} ` +
  `(${Math.round((100 * u.cachedInputTokens) / Math.max(1, u.inputTokens))}%; simulated upper bound, real cache hits are provider-dependent)`);

console.log("\nOFFLINE CONSOLIDATION");
const report = await consolidate({ llm, profile: SYSTEM2, episodes, knowledge, habits, tools: TOOLS });
console.log(`  replayed: ${report.replayed.join(", ")}`);
for (const k of report.learned) console.log(`  learned ${k.id}: ${k.statement} (sources: ${k.sources.join(", ")})`);
for (const h of report.habitsProposed) console.log(`  habit candidate ${h.id} [${h.status}]: "${h.trigger}" -> ${h.call.tool}`);
for (const r of report.rejected) console.log(`  rejected: ${r}`);

console.log("\nIDLE (no tools available)");
const before = tools.calls.length;
const ideas = await idle({ llm, profile: SYSTEM1, knowledge, episodes });
for (const i of ideas.ideas) console.log(`  idea: ${i}`);
for (const d of ideas.discarded) console.log(`  discarded: ${d}`);
console.log(`  tool calls during idle: ${tools.calls.length - before}`);
console.log(`\nTotal scripted model requests: ${llm.requests.length}`);
