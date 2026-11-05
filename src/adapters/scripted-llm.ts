import type { GenerateRequest, GenerateResponse, LLM } from "../io/llm";
import { estimateTokens } from "../io/tokens";

export type Script = (req: GenerateRequest) => string | object;

/**
 * A fake model driven by a function, for tests and the offline demo.
 * It simulates provider prefix caching so you can see whether prefixes stay stable.
 */
export class ScriptedLLM implements LLM {
  readonly calls: GenerateRequest[] = [];
  readonly totals = { calls: 0, inputTokens: 0, cachedInputTokens: 0, outputTokens: 0 };
  private readonly cache = new FakePromptCache();

  constructor(
    readonly name: string,
    private readonly script: Script,
  ) {}

  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    this.calls.push(req);
    const out = this.script(req);
    const text = typeof out === "string" ? out : JSON.stringify(out);
    const { total, cached } = this.cache.lookup(req);
    const usage = { inputTokens: total, cachedInputTokens: cached, outputTokens: estimateTokens(text) };
    this.totals.calls++;
    this.totals.inputTokens += usage.inputTokens;
    this.totals.cachedInputTokens += usage.cachedInputTokens;
    this.totals.outputTokens += usage.outputTokens;
    return { text, usage };
  }
}

/** Remembers every prefix it has seen; a request is "cached" up to its longest seen prefix. */
class FakePromptCache {
  private seen = new Set<string>();

  lookup(req: GenerateRequest): { total: number; cached: number } {
    const chunks = [req.system, ...req.messages.map((m) => JSON.stringify(m))];
    let key = "";
    let total = 0;
    let cached = 0;
    for (const chunk of chunks) {
      key += "\u241e" + chunk;
      total += estimateTokens(chunk);
      if (this.seen.has(key)) cached = total;
      else this.seen.add(key);
    }
    return { total, cached };
  }
}
