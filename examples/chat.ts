/**
 * Interactive chat with the brain over any OpenAI-compatible API.
 *
 *   OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=... \
 *   SYSTEM1_MODEL=... SYSTEM2_MODEL=... EMBEDDING_MODEL=... bun run chat
 *
 * Optional: SYSTEM2_REASONING=1 (send reasoning_effort), BRAIN_DATA=dir (default .brain-data).
 * Commands: /sleep (reminders + memory consolidation), /quit.
 * Text only: no live-model adapter is included, so audio/video input isn't wired up here.
 */
import { createInterface } from "node:readline/promises";
import { Brain, type TickReport } from "../src";
import { FileStorage } from "../src/adapters/file-storage";
import { OpenAICompatibleEmbedder, OpenAICompatibleLLM } from "../src/adapters/openai-compatible";

function env(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`Set ${name}`);
  return value;
}

const connection = { baseUrl: env("OPENAI_BASE_URL"), apiKey: env("OPENAI_API_KEY") };

const brain = await Brain.create({
  models: {
    system1: new OpenAICompatibleLLM({ ...connection, model: env("SYSTEM1_MODEL") }),
    system2: new OpenAICompatibleLLM({ ...connection, model: env("SYSTEM2_MODEL"), reasoning: process.env.SYSTEM2_REASONING === "1" }),
    embedder: new OpenAICompatibleEmbedder({ ...connection, model: env("EMBEDDING_MODEL") }),
  },
  storage: new FileStorage(process.env.BRAIN_DATA ?? ".brain-data"),
  onSay: (text) => console.log(`🧠 ${text}`),
});

function show(reports: TickReport[]) {
  for (const r of reports) {
    const acted = r.chosen.map((p) => `${p.source}→${p.tool}`).join(", ");
    console.log(`   tick ${r.tick}: woke[${r.woke.join(", ")}] acted[${acted}]`);
    if (r.assessment.changed && r.assessment.reasons.length) console.log(`   monitor: ${r.assessment.reasons.join("; ")}`);
  }
}

const rl = createInterface({ input: process.stdin, output: process.stdout });
while (true) {
  const line = (await rl.question("👤 ")).trim();
  if (line === "/quit") break;
  if (line === "/sleep") {
    console.log(await brain.idle());
    show(await brain.run()); // handle any reminders that came due
    continue;
  }
  if (!line) continue;
  await brain.hear({ source: "user", parts: [{ type: "text", text: line }] });
  show(await brain.run());
}
rl.close();
