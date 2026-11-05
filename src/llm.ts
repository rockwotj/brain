/**
 * Provider-neutral LLM port. Adapters for real providers implement `LlmClient`;
 * this file only ships a scripted fake for tests and the demo.
 */

export type Message = { role: "system" | "user" | "assistant"; content: string };

/**
 * Abstract model profile. `tier` + `reasoningEffort` describe *how much thinking*;
 * `serviceMode` is a separate provider latency/priority option (e.g. a "fast" lane)
 * and does not change the model's capability.
 */
export type ModelProfile = {
  id: string;
  model: string;
  tier: "system1" | "system2";
  reasoningEffort: "none" | "low" | "medium" | "high";
  serviceMode: "standard" | "fast";
  contextWindow: number;
  maxOutputTokens: number;
};

export type LlmRequest = {
  /** Which harness agent is calling (for logging / per-agent caching). */
  agent: string;
  profile: ModelProfile;
  messages: readonly Message[];
  /**
   * Number of leading messages identical to this agent's previous request.
   * Adapters MAY use it as a prompt-cache breakpoint. Whether a provider
   * actually serves a cache hit is provider-dependent and never guaranteed.
   */
  stablePrefixLength: number;
  signal?: AbortSignal;
};

export type Usage = { inputTokens: number; outputTokens: number; cachedInputTokens?: number };
export type LlmResponse = { text: string; usage: Usage };

export interface LlmClient {
  complete(request: LlmRequest): Promise<LlmResponse>;
}

/** Injectable token estimator. Swap in a real tokenizer for production use. */
export type TokenEstimator = (text: string) => number;
export const roughTokenEstimate: TokenEstimator = (text) => Math.ceil(text.length / 4);

export function estimateMessages(messages: readonly Message[], estimate: TokenEstimator): number {
  return messages.reduce((sum, m) => sum + estimate(m.content) + 4, 0);
}

export type Responder = (request: LlmRequest, callIndex: number) => string | Promise<string>;

/**
 * Deterministic fake model. Records every request and reports a *simulated*
 * cached-token count equal to the prefix shared with the same agent's previous
 * request — an upper bound on what a prefix cache could reuse, not a measurement.
 */
export class ScriptedLlm implements LlmClient {
  readonly requests: LlmRequest[] = [];
  private readonly previous = new Map<string, readonly Message[]>();

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
    };
  }
}

function sameMessage(a: Message | undefined, b: Message | undefined): boolean {
  return !!a && !!b && a.role === b.role && a.content === b.content;
}

/** Pick the profile for a role. Urgency changes only the service lane, never the tier. */
export function routeProfile(base: ModelProfile, urgency: "normal" | "high"): ModelProfile {
  return urgency === "high" && base.serviceMode !== "fast" ? { ...base, serviceMode: "fast" } : base;
}
