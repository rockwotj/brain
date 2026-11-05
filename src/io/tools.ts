export type Risk = "low" | "medium" | "high";
export type ToolArgs = Record<string, unknown>;

/**
 * Something the brain can do to the world.
 *
 * `effector` groups tools that can't be used at the same time (one voice, one keyboard...).
 * The selector releases at most one action per effector per tick.
 */
export interface Tool {
  name: string;
  description: string;
  /** JSON-schema-ish description of the arguments, shown to models. */
  parameters: Record<string, unknown>;
  effector: string;
  risk: Risk;
  run(args: ToolArgs): Promise<string>;
}

/** Render tools for a prompt. Sorted by name so the text is identical every time (cache-friendly). */
export function describeTools(tools: Tool[]): string {
  return [...tools]
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((t) => `- ${t.name} [effector: ${t.effector}, risk: ${t.risk}]: ${t.description} Args: ${JSON.stringify(t.parameters)}`)
    .join("\n");
}
