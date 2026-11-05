import { FunctionTools, Harness, type HarnessConfig, type LlmRequest, type Message, type SpecialistSpec, type ToolCall } from "../src/index.ts";
import { FAST, REASONING, TOOLS } from "./fixtures/export-incident.ts";
import { ScriptedLlm, sameMessage } from "./fixtures/scripted-llm.ts";

export const reply = (...proposals: unknown[]) => JSON.stringify({ proposals });
export const narration = (answer: string, citations: string[]) => JSON.stringify({ answer, citations });

export function quietTools() {
  return new FunctionTools({
    read_job_logs: () => "log: timeout",
    check_quota: () => "quota 10%",
    get_input_stats: () => "rows 3x",
    restart_job: () => "restarted",
  });
}

export function build(
  respond: (r: LlmRequest, i: number) => string | Promise<string>,
  overrides: Partial<HarnessConfig> & { specialists?: SpecialistSpec[] } = {},
) {
  const llm = new ScriptedLlm(respond);
  const tools = quietTools();
  const harness = new Harness({
    llm,
    profiles: { fast: FAST, reasoning: REASONING },
    tools: TOOLS,
    toolPort: tools,
    specialists: [{ name: "triage", instructions: "investigate", tools: ["read_job_logs", "check_quota", "restart_job"], canFinish: true }],
    ...overrides,
  });
  return { llm, tools, harness };
}

export async function approveHabit(harness: Harness, call: ToolCall, trigger = "export") {
  const habit = await harness.habits.register({ trigger, call, expect: { success: true }, sources: ["ep-1"] });
  await harness.habits.approve(habit.id);
  return habit;
}

/** True if `prefix` is exactly the first messages of `full`. */
export function isPrefix(prefix: readonly Message[], full: readonly Message[]): boolean {
  return prefix.length <= full.length && prefix.every((m, i) => sameMessage(m, full[i]));
}
