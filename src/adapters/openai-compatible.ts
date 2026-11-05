import type { Part } from "../io/content";
import type { Embedder } from "../io/embeddings";
import type { GenerateRequest, GenerateResponse, LLM } from "../io/llm";

/**
 * Example adapter for any OpenAI-compatible chat API (many providers and local servers speak it).
 * Plain fetch, no SDK. UNTESTED against live endpoints — a starting point, not a supported client.
 */
export type OpenAICompatibleOptions = {
  baseUrl: string;
  apiKey: string;
  model: string;
  /** Send `reasoning_effort` for models that support it. */
  reasoning?: boolean;
};

export class OpenAICompatibleLLM implements LLM {
  readonly name: string;

  constructor(private readonly opts: OpenAICompatibleOptions) {
    this.name = opts.model;
  }

  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    // These APIs cache stable prefixes automatically, so req.cacheUpTo needs no marker here.
    // Providers with explicit cache breakpoints would put one on message[req.cacheUpTo - 1].
    const body = {
      model: this.opts.model,
      messages: [{ role: "system", content: req.system }, ...req.messages.map((m) => ({ role: m.role, content: m.content.map(toPart) }))],
      temperature: req.temperature,
      max_tokens: req.maxOutputTokens,
      response_format: req.json ? { type: "json_object" } : undefined,
      reasoning_effort: this.opts.reasoning && req.thinking && req.thinking !== "none" ? req.thinking : undefined,
    };
    const data = await post(this.opts, "/chat/completions", body);
    return {
      text: data.choices?.[0]?.message?.content ?? "",
      usage: {
        inputTokens: data.usage?.prompt_tokens ?? 0,
        cachedInputTokens: data.usage?.prompt_tokens_details?.cached_tokens ?? 0,
        outputTokens: data.usage?.completion_tokens ?? 0,
      },
    };
  }
}

export class OpenAICompatibleEmbedder implements Embedder {
  constructor(private readonly opts: OpenAICompatibleOptions) {}

  async embed(texts: string[]): Promise<number[][]> {
    const data = await post(this.opts, "/embeddings", { model: this.opts.model, input: texts });
    return data.data.map((d: { embedding: number[] }) => d.embedding);
  }
}

function toPart(p: Part) {
  if (p.type === "text") return { type: "text", text: p.text };
  if (p.type === "image") {
    const url = p.url ?? `data:${p.mimeType};base64,${Buffer.from(p.data ?? new Uint8Array()).toString("base64")}`;
    return { type: "image_url", image_url: { url } };
  }
  throw new Error(`${p.type} should go through the live model, not the chat model`);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function post(opts: OpenAICompatibleOptions, path: string, body: unknown): Promise<any> {
  const res = await fetch(`${opts.baseUrl.replace(/\/$/, "")}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${opts.apiKey}` },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status} ${await res.text()}`);
  return res.json();
}
