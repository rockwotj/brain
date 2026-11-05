/**
 * Provider-neutral LLM port. Real providers implement `LlmClient` in
 * src/adapters/; the core never imports an SDK.
 */
import { describeMedia, type ContentPart } from "./content.ts";

/** System and assistant messages are text; user messages may carry images. */
export type Message = { role: "system" | "user" | "assistant"; content: string | readonly ContentPart[] };

/**
 * Abstract profile of a generative model. Three independent settings:
 *  - `tier`: the model class, "fast" (small, cheap) or "reasoning" (deliberate);
 *  - `reasoningEffort`: how much the model thinks per call;
 *  - `serviceMode`: a provider latency/priority lane ("standard", "fast",
 *    "ultrafast"). It changes how quickly a request is served, not the model.
 *    Adapters map lanes only to provider options they are configured with.
 */
export type ModelProfile = {
  id: string;
  model: string;
  tier: "fast" | "reasoning";
  reasoningEffort: "none" | "low" | "medium" | "high";
  serviceMode: ServiceMode;
  contextWindow: number;
  maxOutputTokens: number;
  /** The model accepts image parts. Audio/video never go to chat models. */
  vision?: boolean;
};
export type ServiceMode = "standard" | "fast" | "ultrafast";

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
export type LlmResponse = {
  text: string;
  usage: Usage;
  /** Optional provider hint (ms since epoch): when this prompt's cached prefix expires. */
  cacheExpiresAt?: number;
};

export interface LlmClient {
  complete(request: LlmRequest): Promise<LlmResponse>;
}

export class UnsupportedContentError extends Error {}

/** Throws if a message carries media the target model cannot take. Adapters call this. */
export function assertContentSupported(messages: readonly Message[], accepts: { images: boolean }, target: string): void {
  for (const m of messages) {
    if (typeof m.content === "string") continue;
    for (const part of m.content) {
      if (part.type === "text") continue;
      if (part.type !== "image") throw new UnsupportedContentError(`${target}: ${describeMedia(part)} must go through a LiveModel, not a chat model`);
      if (!accepts.images) throw new UnsupportedContentError(`${target}: this model is text-only and cannot take ${describeMedia(part)}`);
    }
  }
}

/** Injectable token estimator. Swap in a real tokenizer for production use. */
export type TokenEstimator = (text: string) => number;
export const roughTokenEstimate: TokenEstimator = (text) => Math.ceil(text.length / 4);
/** Flat budget per image. Real costs depend on provider and resolution. */
export const MEDIA_TOKEN_ESTIMATE = 1_000;

export function estimateMessages(messages: readonly Message[], estimate: TokenEstimator): number {
  return messages.reduce((sum, m) => {
    if (typeof m.content === "string") return sum + estimate(m.content) + 4;
    return sum + 4 + m.content.reduce((s, p) => s + (p.type === "text" ? estimate(p.text) : MEDIA_TOKEN_ESTIMATE), 0);
  }, 0);
}

/** `fast` serves specialists (and the narrator by default); `reasoning` serves the executive. */
export type Profiles = { fast: ModelProfile; reasoning: ModelProfile; narrator?: ModelProfile };
export type AgentRole = "specialist" | "executive" | "narrator";

/**
 * The one place that picks a model per role. Specialists and the narrator use
 * the fast profile, the executive the reasoning profile. Each agent keeps its
 * profile for life (a transcript never changes model mid-stream); per call,
 * `withUrgency` may only switch the provider service lane.
 */
export function chooseProfile(profiles: Profiles, role: AgentRole): ModelProfile {
  return role === "executive" ? profiles.reasoning : role === "narrator" ? (profiles.narrator ?? profiles.fast) : profiles.fast;
}

/**
 * Urgency changes the provider service lane only, never the model, tier or effort.
 * High urgency raises "standard" to "fast"; "fast" and "ultrafast" stay as they are.
 */
export function withUrgency(profile: ModelProfile, urgency: "normal" | "high"): ModelProfile {
  return urgency === "high" && profile.serviceMode === "standard" ? { ...profile, serviceMode: "fast" } : profile;
}
