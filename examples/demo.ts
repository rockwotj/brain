/**
 * Offline walkthrough with toy models (no API keys needed):  bun examples/demo.ts
 *
 *  1. A new question goes the slow way: workspace → planner (System 2) → calculator → reply.
 *  2. Sleep turns that episode into a skill.
 *  3. The same kind of question now runs as a habit on System 1, skipping the workspace.
 *  4. Audio input goes through the live model and also hits the habit.
 *  5. A surprising tool result makes the habit hand control back to the slow path.
 */
import { Brain, type TickReport } from "../src";
import { HashEmbedder } from "../src/adapters/hash-embedder";
import { MemoryStorage } from "../src/io/storage";
import { calculator, toyLive, toySystem1, toySystem2 } from "./toy-models";

const system1 = toySystem1();
const system2 = toySystem2();

const brain = await Brain.create({
  models: { system1, system2, live: toyLive, embedder: new HashEmbedder() },
  storage: new MemoryStorage(),
  tools: [calculator],
  onSay: (text) => console.log(`  🧠 ${text}`),
});

function show(reports: TickReport[]) {
  for (const r of reports) {
    const events = r.events.map((e) => `${e.route}(${e.surprise.toFixed(1)})`).join(" ");
    const acts = r.chosen.map((p) => `${p.source}→${p.tool}`).join(", ");
    const vetoes = r.vetoed.map((v) => `${v.proposal.tool}: ${v.reason}`).join(", ");
    console.log(`    tick ${r.tick}: events[${events}] woke[${r.woke.join(",")}] act[${acts}]${vetoes ? ` vetoed[${vetoes}]` : ""}`);
    if (r.assessment.changed && r.assessment.reasons.length) console.log(`      monitor: ${r.assessment.reasons.join("; ")}`);
  }
}

async function ask(question: string) {
  console.log(`\n👤 ${question}`);
  await brain.hear({ source: "user", parts: [{ type: "text", text: question }] });
  show(await brain.run());
}

console.log("— 1. Novel request: slow path —");
await ask("What is 17 * 23?");

console.log("\n— 2. Sleep —");
console.log("  ", await brain.idle());

console.log("\n— 3. Familiar request: habit on System 1 —");
const before = system2.totals.calls;
await ask("What is 12 * 11?");
console.log(`    System 2 calls during this request: ${system2.totals.calls - before}`);

console.log("\n— 4. Audio input via the live model —");
console.log("👤 (voice memo)");
await brain.hear({ source: "user", parts: [{ type: "audio", mimeType: "audio/wav", data: new TextEncoder().encode("What is 6 * 7?") }] });
show(await brain.run());

console.log("\n— 5. Surprise: the habit fails and the slow path takes over —");
await ask("What is 999 * 999?");

console.log("\n— Usage —");
for (const m of [system1, system2]) {
  const t = m.totals;
  const hit = t.inputTokens ? Math.round((100 * t.cachedInputTokens) / t.inputTokens) : 0;
  console.log(`  ${m.name}: ${t.calls} calls, ${t.inputTokens} input tokens (${hit}% from simulated prefix cache), ${t.outputTokens} output tokens`);
}
console.log(`  budget left: ${(brain.body.budgetLeft() * 100).toFixed(1)}%`);
console.log(`  modulators:`, Object.fromEntries(Object.entries(brain.modulators).map(([k, v]) => [k, Number(v.toFixed(2))])));
