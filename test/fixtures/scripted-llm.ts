/**
 * TEST FIXTURE ONLY. A deterministic stand-in for a model: replies come from a
 * script, every request is recorded, and the "cached" token count is simulated
 * as the prefix shared with the same agent's previous request (an upper bound
 * on what a prefix cache could reuse, not a measurement).
 */
import { estimateMessages, roughTokenEstimate, type LlmClient, type LlmRequest, type LlmResponse, type Message, type TokenEstimator } from "../../src/index.ts";

export type Responder = (request: LlmRequest, callIndex: number) => string | Promise<string>;

export class ScriptedLlm implements LlmClient {
  readonly requests: LlmRequest[] = [];
  private readonly previous = new Map<string, readonly Message[]>();
  /** Optional cache-expiry hint to return with every response. */
  cacheExpiresAt?: number;

  constructor(
    private readonly respond: Responder,
    private readonly estimate: TokenEstimator = roughTokenEstimate,
  ) {}

  async complete(request: LlmRequest): Promise<LlmResponse> {
    request.signal?.throwIfAborted();
    this.requests.push({ ...request, messages: [...request.messages] });
    const text = await this.respond(request, this.requests.length - 1);
    const prev = this.previous.get(request.agent) ?? [];
    let shared = 0;
    while (shared < prev.length && shared < request.messages.length && sameMessage(prev[shared], request.messages[shared])) shared++;
    this.previous.set(request.agent, [...request.messages, { role: "assistant", content: text }]);
    return {
      text,
      usage: {
        inputTokens: estimateMessages(request.messages, this.estimate),
        cachedInputTokens: estimateMessages(request.messages.slice(0, shared), this.estimate),
        outputTokens: this.estimate(text),
      },
      ...(this.cacheExpiresAt !== undefined ? { cacheExpiresAt: this.cacheExpiresAt } : {}),
    };
  }
}

export function sameMessage(a: Message | undefined, b: Message | undefined): boolean {
  return !!a && !!b && a.role === b.role && JSON.stringify(a.content) === JSON.stringify(b.content);
}

/** All text a request shows the model (media parts skipped). */
export function transcriptText(r: LlmRequest): string {
  return r.messages.map((m) => (typeof m.content === "string" ? m.content : m.content.map((p) => (p.type === "text" ? p.text : `[${p.type}]`)).join("\n"))).join("\n");
}

/** Text of the newest message of a request. */
export function lastText(r: LlmRequest): string {
  const m = r.messages.at(-1);
  if (!m) return "";
  return typeof m.content === "string" ? m.content : m.content.map((p) => (p.type === "text" ? p.text : `[${p.type}]`)).join("\n");
}
