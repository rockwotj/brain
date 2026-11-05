/**
 * Interactive chat with the harness over any OpenAI-compatible API.
 * Real provider calls only; nothing here is scripted. (Not run in CI.)
 *
 *   OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=... \
 *   FAST_MODEL=<small fast model> REASONING_MODEL=<reasoning model> bun run chat
 *
 * Optional env:
 *   EMBEDDING_MODEL=...           enables memory recall (cosine similarity)
 *   VISION=reasoning|all          which profiles accept images (default: none)
 *   SEND_REASONING_EFFORT=1       send reasoning_effort (only for models that accept it)
 *   FAST_SERVICE_MODE=standard|fast|ultrafast   service lane for the fast profile (default standard)
 *   SERVICE_TIER_FAST=...         provider `service_tier` value for the "fast" lane, if it has one
 *   SERVICE_TIER_ULTRAFAST=...    provider `service_tier` value for the "ultrafast" lane, if it has one
 *   CONTEXT_WINDOW=128000         context window of both models (tokens)
 *   BRAIN_DATA=.brain-data        directory for sessions, records and memory
 *
 * Commands: /image <path> (attach to the next message), /new (task boundary),
 * /sleep (consolidate episodes into knowledge), /quit.
 * Audio/video and decision models: no adapters ship with this PoC, so this
 * example runs without them (deterministic routing; no audio/video input).
 */
import { readdir, readFile } from "node:fs/promises";
import { extname, relative, resolve } from "node:path";
import { createInterface } from "node:readline/promises";
import { FileStorage } from "../src/adapters/file-storage.ts";
import { OpenAICompatibleEmbedder, OpenAICompatibleLlm } from "../src/adapters/openai-compatible.ts";
import { consolidate, FunctionTools, Harness, type ContentPart, type ModelProfile, type ToolSpec } from "../src/index.ts";

function env(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`Set ${name}`);
  return value;
}

const connection = { baseUrl: env("OPENAI_BASE_URL"), apiKey: env("OPENAI_API_KEY") };
const contextWindow = Number(process.env.CONTEXT_WINDOW ?? 128_000);
const vision = process.env.VISION ?? "";
const fastMode = process.env.FAST_SERVICE_MODE ?? "standard";
if (fastMode !== "standard" && fastMode !== "fast" && fastMode !== "ultrafast") throw new Error("FAST_SERVICE_MODE must be standard, fast or ultrafast");
const fast: ModelProfile = {
  id: "fast", model: env("FAST_MODEL"), tier: "fast", reasoningEffort: "none", serviceMode: fastMode,
  contextWindow, maxOutputTokens: 1_000, vision: vision === "all",
};
const reasoning: ModelProfile = {
  id: "reasoning", model: env("REASONING_MODEL"), tier: "reasoning", reasoningEffort: "medium", serviceMode: "standard",
  contextWindow, maxOutputTokens: 4_000, vision: vision === "all" || vision === "reasoning",
};

// Real, read-only tools scoped to the current directory.
const root = process.cwd();
const inside = (path: string) => {
  const full = resolve(root, path);
  if (relative(root, full).startsWith("..")) throw new Error("path is outside the working directory");
  return full;
};
const tools: ToolSpec[] = [
  { name: "current_time", description: "Current date and time (ISO 8601, UTC)", effect: "read", params: {} },
  { name: "list_files", description: "List entries of a directory under the working directory", effect: "read", params: { path: { type: "string", description: "relative directory path, e.g. ." } } },
  { name: "read_file", description: "Read a UTF-8 text file under the working directory", effect: "read", params: { path: { type: "string", description: "relative file path" } } },
];
const toolPort = new FunctionTools({
  current_time: () => new Date().toISOString(),
  list_files: async (args) => (await readdir(inside(String(args.path)), { withFileTypes: true })).map((e) => `${e.name}${e.isDirectory() ? "/" : ""}`).join("\n"),
  read_file: async (args) => readFile(inside(String(args.path)), "utf8"),
});

const serviceTiers = {
  ...(process.env.SERVICE_TIER_FAST ? { fast: process.env.SERVICE_TIER_FAST } : {}),
  ...(process.env.SERVICE_TIER_ULTRAFAST ? { ultrafast: process.env.SERVICE_TIER_ULTRAFAST } : {}),
};
const llm = new OpenAICompatibleLlm({ ...connection, sendReasoningEffort: process.env.SEND_REASONING_EFFORT === "1", serviceTiers });
const embeddingModel = process.env.EMBEDDING_MODEL;
const harness = new Harness({
  llm,
  profiles: { fast, reasoning },
  tools,
  toolPort,
  permissions: { allowEffects: ["read"] },
  specialists: [
    { name: "responder", instructions: "Answer conversational or general-knowledge questions directly with an answer-only finish. Stay silent when tools or files are needed.", tools: [], canFinish: true },
    { name: "files", instructions: "Answer questions about local files and the time using your tools. Read records in full when a preview is cut. Finish citing act-ids.", tools: ["current_time", "list_files", "read_file"], canFinish: true },
  ],
  storage: new FileStorage(process.env.BRAIN_DATA ?? ".brain-data"),
  ...(embeddingModel ? { embedder: new OpenAICompatibleEmbedder({ ...connection, model: embeddingModel }) } : {}),
});

const session = await harness.openSession("chat");
const rl = createInterface({ input: process.stdin, output: process.stdout });
const pending: ContentPart[] = [];
let newTask = false;
const IMAGE_TYPES: Record<string, string> = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif" };

while (true) {
  const line = (await rl.question("you> ")).trim();
  if (line === "/quit") break;
  if (line === "/new") {
    newTask = true;
    console.log("(next message starts a new task)");
    continue;
  }
  if (line === "/sleep") {
    const report = await consolidate({ llm, profile: reasoning, episodes: harness.episodes, knowledge: harness.knowledge, habits: harness.habits, tools });
    console.log(`replayed ${report.replayed.length} episodes, learned ${report.learned.length}, habits proposed ${report.habitsProposed.length} (pending approval)`);
    continue;
  }
  if (line.startsWith("/image ")) {
    const path = line.slice("/image ".length).trim();
    const mimeType = IMAGE_TYPES[extname(path).toLowerCase()];
    if (!mimeType) console.log("unsupported image type");
    else pending.push({ type: "image", mimeType, base64: (await readFile(path)).toString("base64") });
    continue;
  }
  if (!line) continue;
  try {
    const result = await session.turn({ input: [{ type: "text", text: line }, ...pending.splice(0)], newTask });
    newTask = false;
    console.log(`brain> ${result.answer.text}`);
    const tiers = result.modelCalls.map((c) => c.tier);
    console.log(`  (${result.status}${result.completion ? `, ${result.completion}` : ""}; ${result.steps} steps; ` +
      `calls fast=${tiers.filter((t) => t === "fast").length} reasoning=${tiers.filter((t) => t === "reasoning").length}; ` +
      `input tokens ${result.usage.inputTokens}, provider-reported cached ${result.usage.cachedInputTokens})`);
  } catch (error) {
    console.log(`error: ${error instanceof Error ? error.message : String(error)}`);
  }
}
rl.close();
