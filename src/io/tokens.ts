import type { Part } from "./content";

/** Rough token estimate (≈4 characters per token). Good enough for budgeting decisions. */
export function estimateTokens(value: string | Part[]): number {
  if (typeof value === "string") return Math.ceil(value.length / 4);
  return value.reduce((sum, p) => sum + (p.type === "text" ? estimateTokens(p.text) : 500), 0);
}
