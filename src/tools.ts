/**
 * Tools: a trusted registry (schemas + effect metadata), a port that actually
 * runs tools, a policy gate, and the Actuator — the single execution authority.
 * Models only ever *propose* calls; nothing reaches a ToolPort except via Actuator.
 */
import { isSurprise } from "./monitor.ts";
import type { ActionRecord, Constraint, Effect, JsonObject, Prediction, ToolCall } from "./types.ts";

export type ParamSpec = { type: "string" | "number" | "boolean"; description: string; optional?: boolean };
export type ToolSpec = { name: string; description: string; effect: Effect; params: Record<string, ParamSpec> };
export type ToolResult = { ok: true; output: string } | { ok: false; error: string };

export interface ToolPort {
  invoke(call: ToolCall, signal?: AbortSignal): Promise<ToolResult>;
}

/** Real tool port that dispatches to plain (sync or async) functions. Throwing = tool error. */
export class FunctionTools implements ToolPort {
  /** Every call received, in order (an audit log). */
  readonly calls: ToolCall[] = [];
  constructor(private readonly handlers: Record<string, (args: JsonObject) => string | Promise<string>>) {}
  async invoke(call: ToolCall): Promise<ToolResult> {
    this.calls.push(call);
    const handler = this.handlers[call.tool];
    if (!handler) return { ok: false, error: `no handler for ${call.tool}` };
    try {
      return { ok: true, output: await handler(call.args) };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
  }
}

/** Static permissions granted to this deployment (separate from per-task constraints). */
export type Permissions = { allowEffects: readonly Effect[] };

export type PolicyDecision = { allowed: true; spec: ToolSpec } | { allowed: false; reason: string };

export function checkPolicy(
  spec: ToolSpec | undefined,
  call: ToolCall,
  constraints: readonly Constraint[],
  permissions: Permissions,
): PolicyDecision {
  if (!spec) return { allowed: false, reason: `unknown tool ${call.tool}` };
  if (!permissions.allowEffects.includes(spec.effect)) return { allowed: false, reason: `permission: ${spec.effect} tools not granted` };
  for (const c of constraints) {
    if (c.kind === "read_only" && spec.effect !== "read") return { allowed: false, reason: `${c.id} read-only: ${c.reason}` };
    if (c.kind === "deny_tool" && c.tool === spec.name) return { allowed: false, reason: `${c.id} denies ${spec.name}: ${c.reason}` };
  }
  const argError = validateArgs(spec, call.args);
  return argError ? { allowed: false, reason: argError } : { allowed: true, spec };
}

export function validateArgs(spec: ToolSpec, args: JsonObject): string | undefined {
  for (const key of Object.keys(args)) if (!spec.params[key]) return `unexpected argument ${key}`;
  for (const [key, p] of Object.entries(spec.params)) {
    const value = args[key];
    if (value === undefined) {
      if (!p.optional) return `missing argument ${key}`;
    } else if (typeof value !== p.type) return `argument ${key} must be ${p.type}`;
  }
  return undefined;
}

export function renderToolSchemas(specs: readonly ToolSpec[]): string {
  if (!specs.length) return "TOOLS: none. You cannot propose actions.";
  // Sorted so the text is byte-identical regardless of registration order (cache-friendly).
  const lines = [...specs].sort((a, b) => a.name.localeCompare(b.name)).map((s) => {
    const params = Object.entries(s.params)
      .map(([k, p]) => `${k}${p.optional ? "?" : ""}: ${p.type} — ${p.description}`)
      .join("; ");
    return `- ${s.name} [${s.effect}]: ${s.description}. Args: { ${params} }`;
  });
  return ["TOOLS (effects are enforced by the runtime):", ...lines].join("\n");
}

export type Execution = { kind: "executed"; record: ActionRecord; effect: Effect } | { kind: "blocked"; reason: string };

export class Actuator {
  private readonly specs: Map<string, ToolSpec>;

  constructor(
    specs: readonly ToolSpec[],
    private readonly port: ToolPort,
    private readonly permissions: Permissions,
  ) {
    this.specs = new Map(specs.map((s) => [s.name, s]));
  }

  spec(name: string): ToolSpec | undefined {
    return this.specs.get(name);
  }

  check(call: ToolCall, constraints: readonly Constraint[]): PolicyDecision {
    return checkPolicy(this.specs.get(call.tool), call, constraints, this.permissions);
  }

  /** Re-checks policy immediately before running, regardless of who proposed the call. */
  async execute(
    call: ToolCall,
    meta: { id: string; step: number; source: string; predicted: Prediction; constraints: readonly Constraint[] },
    signal?: AbortSignal,
  ): Promise<Execution> {
    const decision = this.check(call, meta.constraints);
    if (!decision.allowed) return { kind: "blocked", reason: decision.reason };
    signal?.throwIfAborted(); // last check: a cancelled run must not start a side effect
    let actual: ToolResult;
    try {
      actual = await this.port.invoke(call, signal);
    } catch (error) {
      actual = { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
    const record: ActionRecord = {
      id: meta.id,
      step: meta.step,
      call,
      source: meta.source,
      predicted: meta.predicted,
      actual,
      surprise: isSurprise(meta.predicted, actual),
    };
    return { kind: "executed", record, effect: decision.spec.effect };
  }
}
