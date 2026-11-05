/**
 * Provider-neutral LLM port. Real providers implement `LlmClient` in
 * src/adapters/; the core never imports an SDK.
 */
import { describeMedia, type ContentPart } from "./content.ts";

/** System and assistant messages are text; user messages may carry images. */
export type Message = { role: "system" | "user" | "assistant"; content: string | readonly ContentPart[] };

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
  /** The model accepts image parts. Audio/video never go to chat models. */
  vision?: boolean;
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

export type Profiles = { system1: ModelProfile; system2: ModelProfile; narrator?: ModelProfile };
export type AgentRole = "specialist" | "executive" | "narrator";

/**
 * The one place that picks a model per role. Specialists and the narrator use
 * the cheap System 1 profile, the executive the System 2 profile. Each agent
 * keeps its profile for life (a transcript never changes model mid-stream);
 * per call, `withUrgency` may only switch the provider service lane.
 */
export function chooseProfile(profiles: Profiles, role: AgentRole): ModelProfile {
  return role === "executive" ? profiles.system2 : role === "narrator" ? (profiles.narrator ?? profiles.system1) : profiles.system1;
}

/** Urgency changes the provider service lane only, never the model, tier or effort. */
export function withUrgency(profile: ModelProfile, urgency: "normal" | "high"): ModelProfile {
  return urgency === "high" && profile.serviceMode !== "fast" ? { ...profile, serviceMode: "fast" } : profile;
}
