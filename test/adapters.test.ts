/**
 * Adapters. The OpenAI-compatible client is exercised only with an injected
 * `fetch` (request/response mapping); it has NOT been run against a live endpoint.
 * FileStorage is exercised for real on a temporary directory.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FileStorage } from "../src/adapters/file-storage.ts";
import { OpenAICompatibleEmbedder, OpenAICompatibleLlm } from "../src/adapters/openai-compatible.ts";
import { UnsupportedContentError, type LlmRequest } from "../src/index.ts";
import { FAST } from "./fixtures/export-incident.ts";
import { build, reply } from "./helpers.ts";

type Sent = { url: string; body: Record<string, unknown>; headers: Record<string, string> };
function fakeFetch(response: unknown, sent: Sent[]): typeof fetch {
  return (async (url: string, init: RequestInit) => {
    sent.push({ url, body: JSON.parse(String(init.body)), headers: init.headers as Record<string, string> });
    return new Response(JSON.stringify(response), { status: 200 });
  }) as unknown as typeof fetch;
}
const request = (over: Partial<LlmRequest> = {}): LlmRequest => ({
  agent: "a", profile: FAST, stablePrefixLength: 1,
  messages: [{ role: "system", content: "sys" }, { role: "user", content: "hi" }], ...over,
});

describe("OpenAICompatibleLlm (mapping only, injected fetch)", () => {
  test("maps profile and messages to a chat request and reads usage", async () => {
    const sent: Sent[] = [];
    const llm = new OpenAICompatibleLlm({
      baseUrl: "https://example.invalid/v1/", apiKey: "k", sendReasoningEffort: true, serviceTiers: { fast: "priority" },
      fetch: fakeFetch({ choices: [{ message: { content: "{\"proposals\":[]}" } }], usage: { prompt_tokens: 50, completion_tokens: 5, prompt_tokens_details: { cached_tokens: 40 } } }, sent),
    });
    const profile = { ...FAST, reasoningEffort: "low" as const, serviceMode: "fast" as const, vision: true };
    const res = await llm.complete(request({
      profile,
      messages: [{ role: "system", content: "sys" }, { role: "user", content: [{ type: "text", text: "look" }, { type: "image", mimeType: "image/png", base64: "AAAA" }] }],
    }));
    expect(res).toEqual({ text: "{\"proposals\":[]}", usage: { inputTokens: 50, outputTokens: 5, cachedInputTokens: 40 } });
    const { url, body, headers } = sent[0]!;
    expect(url).toBe("https://example.invalid/v1/chat/completions");
    expect(headers.authorization).toBe("Bearer k");
    expect(body).toEqual({
      model: FAST.model,
      messages: [
        { role: "system", content: "sys" },
        { role: "user", content: [{ type: "text", text: "look" }, { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }] },
      ],
      max_completion_tokens: FAST.maxOutputTokens,
      response_format: { type: "json_object" },
      reasoning_effort: "low",
      service_tier: "priority",
    });
  });

  test("text-only profiles reject images and every chat model rejects audio/video, before any request", async () => {
    const sent: Sent[] = [];
    const llm = new OpenAICompatibleLlm({ baseUrl: "https://example.invalid", apiKey: "k", fetch: fakeFetch({}, sent) });
    const image = request({ messages: [{ role: "user", content: [{ type: "image", mimeType: "image/png", url: "https://x/y.png" }] }] });
    await expect(llm.complete(image)).rejects.toThrow(UnsupportedContentError);
    const audio = request({ profile: { ...FAST, vision: true }, messages: [{ role: "user", content: [{ type: "audio", mimeType: "audio/wav", base64: "AA" }] }] });
    await expect(llm.complete(audio)).rejects.toThrow("must go through a LiveModel");
    expect(sent).toHaveLength(0);
  });

  test("embedder validates the response shape", async () => {
    const ok = new OpenAICompatibleEmbedder({ baseUrl: "https://e.invalid", apiKey: "k", model: "emb", fetch: fakeFetch({ data: [{ embedding: [1, 2] }, { embedding: [3, 4] }] }, []) });
    expect(await ok.embed(["a", "b"])).toEqual([[1, 2], [3, 4]]);
    const bad = new OpenAICompatibleEmbedder({ baseUrl: "https://e.invalid", apiKey: "k", model: "emb", fetch: fakeFetch({ data: [{ embedding: [1] }] }, []) });
    await expect(bad.embed(["a", "b"])).rejects.toThrow("wrong number of rows");
  });

  test("service lanes map only to configured service_tier values (ultrafast included); nothing is guessed", async () => {
    const ok = { choices: [{ message: { content: "{}" } }], usage: {} };
    const sent: Sent[] = [];
    const configured = new OpenAICompatibleLlm({ baseUrl: "https://x.invalid", apiKey: "k", serviceTiers: { ultrafast: "turbo" }, fetch: fakeFetch(ok, sent) });
    await configured.complete(request({ profile: { ...FAST, serviceMode: "ultrafast" } }));
    await configured.complete(request({ profile: { ...FAST, serviceMode: "fast" } })); // no "fast" value configured
    const bare = new OpenAICompatibleLlm({ baseUrl: "https://x.invalid", apiKey: "k", fetch: fakeFetch(ok, sent) });
    await bare.complete(request({ profile: { ...FAST, serviceMode: "ultrafast" } }));
    expect(sent.map((s) => s.body.service_tier)).toEqual(["turbo", undefined, undefined]);
  });
});

describe("FileStorage (real file system)", () => {
  const dirs: string[] = [];
  const tempDir = async () => {
    const dir = await mkdtemp(join(tmpdir(), "brain-test-"));
    dirs.push(dir);
    return dir;
  };
  afterAll(async () => {
    for (const dir of dirs) await rm(dir, { recursive: true, force: true });
  });

  test("round-trips values and logs across instances", async () => {
    const dir = await tempDir();
    const a = new FileStorage(dir);
    await a.put("knowledge/000001", { item: { id: "kn-1" }, embedding: [0.5] });
    await a.put("episodes/000001", { x: 1 });
    await a.append("sessions/s/events", { seq: 1 });
    await a.append("sessions/s/events", { seq: 2 });
    const b = new FileStorage(dir);
    expect(await b.get("knowledge/000001")).toEqual({ item: { id: "kn-1" }, embedding: [0.5] });
    expect(await b.get("missing")).toBeUndefined();
    expect(await b.keys("knowledge/")).toEqual(["knowledge/000001"]);
    expect(await b.read("sessions/s/events")).toEqual([{ seq: 1 }, { seq: 2 }]);
    expect(await b.read("nope")).toEqual([]);
  });

  test("a session, its records and memory survive a restart", async () => {
    const dir = await tempDir();
    const script = (r: LlmRequest, i: number) =>
      r.agent === "narrator" ? JSON.stringify({ answer: "Timed out [act-1].", citations: ["act-1"] })
        : i === 0 ? reply({ kind: "action", tool: "read_job_logs", args: { job: "j" } })
          : reply({ kind: "finish", summary: "timed out", evidence: ["act-1"] });
    const first = build(script, { storage: new FileStorage(dir) });
    const s1 = await first.harness.openSession("s-1");
    const r1 = await s1.turn({ input: "why did it fail?" });
    expect(r1.status).toBe("finished");

    // New process: same directory, fresh objects.
    const second = build(() => reply({ kind: "action", tool: "check_quota", args: { project: "p" } }), { storage: new FileStorage(dir), limits: { maxSteps: 1 } });
    const s2 = await second.harness.openSession("s-1");
    expect(s2.state.goal).toBe("why did it fail?");
    expect(s2.state.records.get("act-1")?.actual).toEqual({ ok: true, output: "log: timeout" });
    expect((await second.harness.episodes.list()).map((e) => e.id)).toEqual(["ep-1"]);
    const r2 = await s2.turn({ input: "and the quota?" });
    expect(r2.actions[0]!.id).toBe("act-2"); // ids continue after the restart
    const reopened = await new FileStorage(dir).read<{ seq: number }>("sessions/s-1/events");
    expect(reopened.map((e) => e.seq)).toEqual(reopened.map((_e, i) => i + 1));
  });
});
