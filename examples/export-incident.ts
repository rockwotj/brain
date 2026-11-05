/**
 * A small fictional scenario used by the demo and the end-to-end tests:
 * "Why did last night's nightly-export job fail?" for an imaginary analytics team.
 * The models are scripted (ScriptedLlm) so every run is deterministic.
 */
import {
  FunctionTools,
  Harness,
  InMemoryEpisodicStore,
  InMemoryHabitStore,
  InMemoryKnowledgeStore,
  ScriptedLlm,
  type LlmRequest,
  type ModelProfile,
  type SpecialistSpec,
  type ToolSpec,
} from "../src/index.ts";

export const SYSTEM1: ModelProfile = {
  id: "s1-small", model: "small-fast-model", tier: "system1", reasoningEffort: "none",
  serviceMode: "standard", contextWindow: 8_000, maxOutputTokens: 400,
};
export const SYSTEM2: ModelProfile = {
  id: "s2-reasoner", model: "large-reasoning-model", tier: "system2", reasoningEffort: "high",
  serviceMode: "standard", contextWindow: 32_000, maxOutputTokens: 2_000,
};

const job = { job: { type: "string", description: "job name" } } as const;
export const TOOLS: ToolSpec[] = [
  { name: "read_job_logs", description: "Read the latest run log of a batch job", effect: "read", params: job },
  { name: "check_quota", description: "Show storage quota usage for a project", effect: "read", params: { project: { type: "string", description: "project id" } } },
  { name: "get_input_stats", description: "Compare last run's input volume with the 7-day average", effect: "read", params: job },
  { name: "restart_job", description: "Restart a batch job now", effect: "write", params: job },
];

export const SPECIALISTS: SpecialistSpec[] = [
  { name: "triage", instructions: "Form and test hypotheses about job failures. Finish when evidence explains the failure.", tools: ["read_job_logs", "check_quota"], canFinish: true },
  { name: "ops", instructions: "Suggest remediation actions for failed jobs.", tools: ["restart_job"] },
  { name: "data", instructions: "Interpret data-volume evidence. Speak only when you have data to interpret.", tools: ["get_input_stats"] },
];

export function makeTools(): FunctionTools {
  return new FunctionTools({
    read_job_logs: () => "02:14 nightly-export: worker exceeded 30m timeout while writing batch 7/7",
    check_quota: () => "storage quota usage 41% (healthy)",
    get_input_stats: () => "input rows 9.1M last night vs 3.0M 7-day average (about 3x)",
    restart_job: () => "restarted",
  });
}

const reply = (proposals: unknown[]) => JSON.stringify({ proposals });
const transcriptText = (r: LlmRequest) => r.messages.map((m) => m.content).join("\n");

/** Scripted models. Each agent decides from its own transcript, like a real model would. */
export function scriptedResponder(r: LlmRequest): string {
  const seen = transcriptText(r);
  switch (r.agent) {
    case "triage":
      if (seen.includes("act-3 get_input_stats")) {
        return reply([{ kind: "finish", summary: "The export hit its 30-minute timeout because input volume was about 3x normal; quota was healthy.", evidence: ["act-1", "act-2", "act-3"] }]);
      }
      if (seen.includes("act-2 check_quota")) {
        return reply([{ kind: "observation", type: "observation", content: "Quota is healthy, so quota is not the cause this time.", evidence: ["act-2"], confidence: "high" }]);
      }
      return reply([
        { kind: "observation", type: "hypothesis", content: "Storage quota may be exhausted again, as recalled from memory.", evidence: ["kn-1"], confidence: "medium" },
        { kind: "action", tool: "check_quota", args: { project: "acme-analytics" }, rationale: "test the quota hypothesis", expect: { success: true, contains: "exceeded" } },
      ]);
    case "ops":
      if (seen.includes("timeout") && !seen.includes("blocked restart_job")) {
        return reply([{ kind: "action", tool: "restart_job", args: { job: "nightly-export" }, rationale: "a restart usually clears timeouts", expect: { success: true } }]);
      }
      return reply([]);
    case "data":
      if (seen.includes("act-3 get_input_stats") && !seen.includes("Input volume roughly tripled")) {
        return reply([{ kind: "observation", type: "observation", content: "Input volume roughly tripled versus the weekly average.", evidence: ["act-3"], confidence: "high" }]);
      }
      return reply([]);
    case "executive":
      return reply([
        { kind: "observation", type: "focus", content: "Timeout with healthy quota: check whether input volume grew.", evidence: ["act-1", "act-2"], confidence: "medium" },
        { kind: "action", tool: "get_input_stats", args: { job: "nightly-export" }, rationale: "a volume spike would explain a timeout", expect: { success: true } },
      ]);
    case "narrator":
      return JSON.stringify({
        answer: "The nightly export timed out after 30 minutes [act-1]. Storage quota was healthy [act-2], " +
          "but input volume was about 3x the weekly average [act-3]. (inferred) The timeout is probably too tight for this volume.",
        citations: ["act-1", "act-2", "act-3"],
      });
    case "consolidator":
      return JSON.stringify({
        lessons: [
          { statement: "For export timeouts, compare input volume with the weekly average before blaming quota.", episodes: ["ep-2"] },
          { statement: "Exports never fail on weekends.", episodes: ["ep-99"] },
        ],
        habits: [{ trigger: "export", tool: "get_input_stats", args: { job: "nightly-export" }, episodes: ["ep-2"] }],
      });
    case "background":
      return reply([
        { kind: "observation", type: "observation", content: "Consider proposing a volume-based timeout for nightly-export.", evidence: ["kn-2"], confidence: "low" },
        { kind: "action", tool: "restart_job", args: { job: "nightly-export" }, rationale: "pre-emptive restart" },
      ]);
    default:
      return reply([]);
  }
}

/** Build a harness with seeded memory: one past episode, one knowledge item, one approved habit. */
export function makeScenario() {
  const llm = new ScriptedLlm(scriptedResponder);
  const tools = makeTools();
  const episodes = new InMemoryEpisodicStore();
  const knowledge = new InMemoryKnowledgeStore();
  const habits = new InMemoryHabitStore();
  const past = episodes.append({ goal: "Why did the nightly export fail?", status: "finished", actions: [], answer: "Storage quota was exceeded." });
  episodes.markConsolidated([past.id]);
  knowledge.add("Nightly export failures are usually caused by storage quota.", [past.id]);
  const habit = habits.register({ trigger: "export", call: { tool: "read_job_logs", args: { job: "nightly-export" } }, expect: { success: true }, sources: [past.id] });
  habits.approve(habit.id);
  const harness = new Harness({
    llm,
    profiles: { system1: SYSTEM1, system2: SYSTEM2 },
    tools: TOOLS,
    toolPort: tools,
    specialists: SPECIALISTS,
    memory: { episodes, knowledge, habits },
  });
  return { llm, tools, harness, episodes, knowledge, habits };
}

export const GOAL = "Why did last night's nightly export job fail?";
export const READ_ONLY = { id: "c-1", kind: "read_only", reason: "investigation only; do not change production" } as const;
