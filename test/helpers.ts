import { FunctionTools, Harness, ScriptedLlm, type HarnessConfig, type LlmRequest, type Message, type SpecialistSpec } from "../src/index.ts";
import { SYSTEM1, SYSTEM2, TOOLS } from "../examples/export-incident.ts";

export const reply = (...proposals: unknown[]) => JSON.stringify({ proposals });

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
    profiles: { system1: SYSTEM1, system2: SYSTEM2 },
    tools: TOOLS,
    toolPort: tools,
    specialists: [{ name: "triage", instructions: "investigate", tools: ["read_job_logs", "check_quota", "restart_job"], canFinish: true }],
    ...overrides,
  });
  return { llm, tools, harness };
}

/** True if `prefix` is exactly the first messages of `full`. */
export function isPrefix(prefix: readonly Message[], full: readonly Message[]): boolean {
  return prefix.length <= full.length && prefix.every((m, i) => m.role === full[i]?.role && m.content === full[i]?.content);
}
