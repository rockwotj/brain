/**
 * Offline consolidation ("sleep"). Run explicitly between tasks, never inside
 * the live loop. Replays unconsolidated episodes (surprising ones first), asks a
 * model for lessons and habit candidates, and only accepts items whose cited
 * episodes were actually replayed. Habit candidates are registered as *pending*:
 * a human must approve them before they can fire.
 */
import type { LlmClient, ModelProfile } from "./llm.ts";
import type { EpisodicStore, HabitStore, KnowledgeStore } from "./memory.ts";
import { callText, clip } from "./state.ts";
import { validateArgs, type ToolSpec } from "./tools.ts";
import type { Episode, Habit, KnowledgeItem } from "./types.ts";
import { parseConsolidation } from "./validate.ts";

export type ConsolidationInput = {
  llm: LlmClient;
  profile: ModelProfile;
  episodes: EpisodicStore;
  knowledge: KnowledgeStore;
  habits: HabitStore;
  tools: readonly ToolSpec[];
  maxEpisodes?: number;
};

export type ConsolidationReport = { replayed: string[]; learned: KnowledgeItem[]; habitsProposed: Habit[]; rejected: string[] };

const SYSTEM = `You consolidate an agent's past episodes into durable knowledge.
Reply with exactly one JSON object:
{"lessons":[{"statement":"general, reusable lesson","episodes":["ep-ids that support it"]}],
 "habits":[{"trigger":"phrase in future goals","tool":"tool name","args":{...},"episodes":["ep-ids"]}]}
Only cite episodes listed in the request. Prefer lessons learned from surprises. Empty lists are fine.`;

export async function consolidate(input: ConsolidationInput): Promise<ConsolidationReport> {
  const surprises = (e: Episode) => e.actions.filter((a) => a.surprise).length;
  const batch = input.episodes.list()
    .filter((e) => !e.consolidated)
    .sort((a, b) => surprises(b) - surprises(a))
    .slice(0, input.maxEpisodes ?? 5);
  const report: ConsolidationReport = { replayed: batch.map((e) => e.id), learned: [], habitsProposed: [], rejected: [] };
  if (!batch.length) return report;

  const existing = input.knowledge.list().map((k) => `- ${k.id}: ${k.statement}`);
  const request = ["EPISODES", ...batch.map(digest), "EXISTING KNOWLEDGE", ...(existing.length ? existing : ["- (none)"])].join("\n");
  const res = await input.llm.complete({
    agent: "consolidator",
    profile: input.profile,
    messages: [{ role: "system", content: SYSTEM }, { role: "user", content: request }],
    stablePrefixLength: 1,
  });
  const parsed = parseConsolidation(res.text);
  report.rejected.push(...parsed.issues);
  const replayed = new Set(report.replayed);
  const cited = (ids: string[]) => ids.length > 0 && ids.every((id) => replayed.has(id));

  for (const lesson of parsed.value.lessons) {
    if (!cited(lesson.episodes)) report.rejected.push(`lesson without valid provenance: ${clip(lesson.statement, 60)}`);
    else report.learned.push(input.knowledge.add(lesson.statement, lesson.episodes));
  }
  for (const h of parsed.value.habits) {
    const spec = input.tools.find((t) => t.name === h.tool);
    const argError = spec ? validateArgs(spec, h.args) : `unknown tool ${h.tool}`;
    if (argError || !cited(h.episodes)) {
      report.rejected.push(`habit rejected (${argError ?? "invalid provenance"}): ${h.tool}`);
      continue;
    }
    report.habitsProposed.push(input.habits.register({ trigger: h.trigger, call: { tool: h.tool, args: h.args }, expect: { success: true }, sources: h.episodes }));
  }
  input.episodes.markConsolidated(report.replayed);
  return report;
}

function digest(e: Episode): string {
  const actions = e.actions.map((a) => {
    const out = a.actual.ok ? a.actual.output : `ERROR ${a.actual.error}`;
    return `    ${a.id} ${callText(a.call)} -> ${clip(out, 120)}${a.surprise ? " SURPRISE" : ""}`;
  });
  return [`- ${e.id} goal="${e.goal}" status=${e.status}`, ...actions, `    answer: ${clip(e.answer ?? "", 200)}`].join("\n");
}
