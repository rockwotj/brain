import type { Part } from "./content";

/**
 * Provider-agnostic LLM interface. Real SDKs live behind this in src/adapters.
 */
export type Message = { role: "user" | "assistant"; content: Part[] };

export type Thinking = "none" | "low" | "medium" | "high";

/**
 * System 1: fast, cheap model (habits, judging predictions, risk tags, summaries, light loops).
 * System 2: slow, strong model (deliberation, planning, consolidation).
 */
export type Tier = "system1" | "system2";

export type GenerateRequest = {
  /** Stable instructions. Only changes when a context compacts. */
  system: string;
  messages: Message[];
  /**
   * The first `cacheUpTo` messages are byte-identical to the previous call on this context.
   * Adapters for providers with explicit cache breakpoints should put one here; providers
   * with automatic prefix caching can ignore it.
   */
  cacheUpTo?: number;
  thinking?: Thinking;
  temperature?: number;
  /** Ask the model to answer with a single JSON object. */
  json?: boolean;
  maxOutputTokens?: number;
};

export type Usage = { inputTokens: number; cachedInputTokens: number; outputTokens: number };

export type GenerateResponse = { text: string; usage: Usage };

export interface LLM {
  readonly name: string;
  generate(req: GenerateRequest): Promise<GenerateResponse>;
}

/** Ask for JSON and parse it. */
export async function generateJson<T>(llm: LLM, req: GenerateRequest): Promise<{ value: T; raw: string }> {
  const res = await llm.generate({ ...req, json: true });
  return { value: parseJson<T>(res.text), raw: res.text };
}

/** Parse a JSON object, tolerating prose or code fences around it. */
export function parseJson<T>(raw: string): T {
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start === -1 || end < start) throw new Error(`Expected a JSON object, got: ${raw.slice(0, 200)}`);
  return JSON.parse(raw.slice(start, end + 1)) as T;
}
