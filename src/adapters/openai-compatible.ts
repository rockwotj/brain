/**
 * Adapter for OpenAI-compatible HTTP APIs (chat completions + embeddings).
 * Plain `fetch`, no SDK. NOT TESTED AGAINST A LIVE ENDPOINT: the unit tests
 * only check request/response mapping with an injected `fetch`.
 */
import type { MediaPart } from "../io/content.ts";
import type { Embedder } from "../io/embeddings.ts";
import { assertContentSupported, type LlmClient, type LlmRequest, type LlmResponse, type Message } from "../io/llm.ts";

export type OpenAICompatibleOptions = {
  baseUrl: string;
  apiKey: string;
  /** Send the profile's `reasoningEffort` as `reasoning_effort` (only for models that support it). */
  sendReasoningEffort?: boolean;
  /**
   * `service_tier` value to send per service lane, only for lanes your provider
   * actually offers. A lane without a configured value sends no `service_tier`
   * (the provider's default lane); nothing is guessed.
   */
  serviceTiers?: { fast?: string; ultrafast?: string };
  /** Name of the output-limit field; older servers only know `max_tokens`. */
  tokenLimitField?: "max_completion_tokens" | "max_tokens";
  /** Ask for `response_format: json_object` (all harness prompts request JSON). Default true. */
  jsonMode?: boolean;
  fetch?: typeof fetch;
};

export class OpenAICompatibleLlm implements LlmClient {
  constructor(private readonly opts: OpenAICompatibleOptions) {}

  async complete(req: LlmRequest): Promise<LlmResponse> {
    // Image support is declared once, on the profile (`vision`); audio/video are always rejected.
    assertContentSupported(req.messages, { images: !!req.profile.vision }, req.profile.model);
    // These APIs cache identical prompt prefixes automatically, so `stablePrefixLength`
    // needs no marker here. Providers with explicit breakpoints would place one there.
    const body: Record<string, unknown> = {
      model: req.profile.model,
      messages: req.messages.map(toWire),
      [this.opts.tokenLimitField ?? "max_completion_tokens"]: req.profile.maxOutputTokens,
    };
    if (this.opts.jsonMode ?? true) body.response_format = { type: "json_object" };
    if (this.opts.sendReasoningEffort && req.profile.reasoningEffort !== "none") body.reasoning_effort = req.profile.reasoningEffort;
    const mode = req.profile.serviceMode;
    const tier = mode === "standard" ? undefined : this.opts.serviceTiers?.[mode];
    if (tier) body.service_tier = tier;

    const data = await post(this.opts, "/chat/completions", body, req.signal);
    const content = data?.choices?.[0]?.message?.content;
    if (typeof content !== "string") throw new Error("chat completion returned no text content");
    return {
      text: content,
      usage: {
        inputTokens: numberOr0(data?.usage?.prompt_tokens),
        outputTokens: numberOr0(data?.usage?.completion_tokens),
        cachedInputTokens: numberOr0(data?.usage?.prompt_tokens_details?.cached_tokens),
      },
    };
  }
}

export class OpenAICompatibleEmbedder implements Embedder {
  constructor(private readonly opts: OpenAICompatibleOptions & { model: string }) {}

  async embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]> {
    const data = await post(this.opts, "/embeddings", { model: this.opts.model, input: texts }, signal);
    const rows: unknown = data?.data;
    if (!Array.isArray(rows) || rows.length !== texts.length) throw new Error("embeddings response has the wrong number of rows");
    return rows.map((row: { embedding?: unknown }) => {
      const v = row?.embedding;
      if (!Array.isArray(v) || !v.every((x) => typeof x === "number")) throw new Error("embedding is not a number array");
      return v;
    });
  }
}

function toWire(m: Message): { role: string; content: unknown } {
  if (typeof m.content === "string") return { role: m.role, content: m.content };
  return {
    role: m.role,
    content: m.content.map((p) => (p.type === "text" ? { type: "text", text: p.text } : { type: "image_url", image_url: { url: imageUrl(p) } })),
  };
}

function imageUrl(p: MediaPart): string {
  if (p.url) return p.url;
  if (p.base64) return `data:${p.mimeType};base64,${p.base64}`;
  throw new Error("image part needs base64 data or a url");
}

function numberOr0(v: unknown): number {
  return typeof v === "number" ? v : 0;
}

// Response JSON is provider data; fields are checked where used above.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function post(opts: OpenAICompatibleOptions, path: string, body: unknown, signal?: AbortSignal): Promise<any> {
  const res = await (opts.fetch ?? fetch)(`${opts.baseUrl.replace(/\/$/, "")}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${opts.apiKey}` },
    body: JSON.stringify(body),
    ...(signal ? { signal } : {}),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status} ${(await res.text()).slice(0, 500)}`);
  return res.json();
}
