# brain

A small, readable TypeScript proof of concept for an LLM agent harness where
several models work at different speeds: cheap, narrow **specialists** run in
parallel on a fast model; a slower **executive** on a reasoning model is
consulted only when something calls for thought; and a deterministic
**runtime** decides what actually happens.

It is a place to try out ideas in plain code, not a product. There are no
benchmarks, and the test suite never calls a live model provider.

## Why it exists

Many agent loops send every step to one large model with one ever-growing
prompt. This project tries a different split:

- **Cheap by default, expensive on demand.** Specialists use the fast profile.
  The reasoning model is woken for a reason: a tool result that contradicts a
  prediction, conflicting proposals, a stall, a new goal nobody could act on,
  images only it can see, or a judgment that it is needed.
- **Models propose, the runtime decides.** Agents only reply with JSON
  proposals. A deterministic selector releases at most one action per step, and
  every tool call passes a policy gate. No model can get around a read-only
  constraint.
- **Judging is not generating.** "Should this specialist look?" and "does the
  executive need to think?" are typed questions (yes/no, choice, score) answered
  with probabilities by a separate *decision model* port: a fast classifier,
  sometimes called a "System One" model, that never writes text. It only
  steers routing; hard rules never depend on it.
- **Prompts that stay cacheable.** Each agent keeps its own append-only
  transcript. New information arrives as deltas, so each of an agent's requests
  extends its previous one exactly and a provider can reuse the cached prefix. The
  transcript is compacted into a fresh snapshot near the context limit or when
  an optional provider-supplied cache-expiry hint calls for it.
- **No provider lock-in.** The core talks to small ports (LLM, decision model,
  embeddings, live audio/video, storage, tools). One OpenAI-compatible HTTP
  adapter and a file-storage adapter ship with it; the rest is yours to plug in.

Names like *workspace*, *executive*, *habit* and *sleep* are borrowed from
cognitive science as design metaphors. This is not a model of the brain.

## What a run does

A **session** is one conversation. Each `session.turn()` (or `harness.run()`,
a single turn in a fresh session) goes like this:

1. **Perceive.** The input text (plus any images) becomes the goal, and
   constraints are pinned. Audio and video streams, if any, are read by your
   `LiveModel` into one perception record. This happens before the first step
   and stops after `maxLiveObservations`. With an embedder, related knowledge
   and past episodes are recalled into the workspace, marked as unverified
   claims.
2. **Step**, up to `maxSteps` times:
   - **Reflex.** If exactly one approved, read-only habit matches the goal and
     nothing needs attention, it runs with no model call.
   - **Route.** Otherwise, specialists that have news visible to them are woken
     and called in parallel. If you configure a decision model, it can skip
     some of them. Each replies with zero or more proposals: an observation, a
     tool action with a predicted outcome, a request to read a full record, or
     a finish.
   - **Escalate.** If the monitor has a reason, or the decision model judges it
     needed, the executive is called and its proposals join the pool.
   - **Select.** Proposals blocked by policy are dropped and broadcast. The
     rest are ranked by origin (executive > habit > specialist), then kind
     (verified finish > read > write > answer-only finish), then how many
     distinct model profiles agree. If the top two tie and one of them writes,
     nothing runs and the executive is consulted.
   - **Act.** The actuator checks policy again and calls your `ToolPort`. If
     the outcome contradicts the prediction, that is a *surprise*: the
     executive is consulted on the next step, and writes need its backing for
     a few steps.
3. **Stop** on a finish, after `maxSteps`, when the model-call budget runs
   out, when nothing moves even after the executive was consulted, or when the
   `AbortSignal` fires.
4. **Answer.** A finish that cites a successful action is *verified*: a
   narrator writes the answer from the recorded trace, and its citations are
   checked against real record ids. A finish with no tool evidence is
   *answer-only* and returned as written. Every turn is stored as an episode.

An abbreviated trace from the scripted test scenario
(`test/fixtures/export-incident.ts`). A remembered claim says export failures
are usually quota problems, the goal is "Why did last night's nightly export
job fail?", and the task is pinned read-only:

```text
step 1
  reflex read_job_logs({"job":"nightly-export"}): approved read-only habit hb-1; deliberation skipped
  #5 [actuator] act-1 read_job_logs(…) -> ok (73 chars): 02:14 nightly-export: worker exceeded 30m timeout …
step 2
  wake [triage, ops, data] in parallel on fast-small
  #8 [policy] blocked restart_job({"job":"nightly-export"}): c-1 read-only: investigation only; …
  select check_quota({"project":"acme-analytics"}): origin=specialist, kind=read, …
  #9 [actuator] act-2 check_quota(…) -> ok (33 chars): storage quota usage 41% (healthy) (SURPRISE)
step 3
  wake [triage, ops, data] in parallel on fast-small
  escalate (surprise) -> executive on reasoner
  select get_input_stats({"job":"nightly-export"}): origin=executive, kind=read, …
step 4
  wake [triage, ops, data] in parallel on fast-small
  select finish: origin=specialist, kind=verified finish, …
end: finished (verified) after 4 steps; episode ep-2
```

The triage specialist predicted that the quota check would show "exceeded".
It did not, so the executive was consulted and redirected the investigation to
input volume. The restart proposed by `ops` was blocked by the read-only
constraint and never reached the tool.

### Between turns: sleep and idle

Two more functions work on stored memory. Both are ordinary `async` functions
that **you call**, for example between tasks or from your own job. The harness
never schedules them and never runs them in the background.

Run sleep between turns, with one writer at a time. Concurrent memory updates
are not protected by transactions or locks: ID allocation and read–modify–write
updates can race, and the file adapter does not support concurrent writers.
An async function is not, by itself, safe to run alongside another writer.

- `consolidate()` ("sleep") replays unconsolidated episodes, with surprising
  ones first, and asks a model for lessons and habit candidates. A lesson is
  kept only if it cites episodes that were actually replayed. Habits are
  registered as *pending* until someone calls `harness.habits.approve(id)`
  (an opt-in flag, `autoApproveReadOnlyHabits`, approves read-only ones). A
  habit whose outcome surprises is suspended.
- `idle()` asks a model for at most a couple of ideas, using recent knowledge
  and episodes. It has no tools. You can pass its ideas to a later turn as
  `preparedIdeas`, and they enter the workspace at low priority.

## Quick start

Requires [Bun](https://bun.sh) (checked with 1.4.4).

```sh
bun install
bun run check   # tsc --noEmit, then bun test (offline: models are scripted test doubles)
```

To chat with real models over any OpenAI-compatible API:

```sh
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=... \
FAST_MODEL='your-fast-model' REASONING_MODEL='your-reasoning-model' \
bun run chat
```

The chat example has two specialists (a direct responder and a file reader)
and three read-only tools scoped to the current directory: `current_time`,
`list_files` and `read_file`. It stores sessions and memory in `.brain-data/`.
Commands:

- `/image <path>` attaches an image to your next message.
- `/new` makes the next message start a new task.
- `/sleep` runs `consolidate()`.
- `/quit` exits.

Optional settings (embeddings, vision, service tiers, context window) are
listed at the top of `examples/chat.ts`. It runs without a decision model and
without audio/video, because no adapters for those ship here.

## Using it in code

Save this as `examples/minimal.ts` and run `bun examples/minimal.ts` with the
same environment variables as the chat example:

```ts
import { OpenAICompatibleLlm } from "../src/adapters/openai-compatible.ts";
import { FunctionTools, Harness, type ModelProfile } from "../src/index.ts";

const fast: ModelProfile = {
  id: "fast", model: process.env.FAST_MODEL!, tier: "fast", reasoningEffort: "none",
  serviceMode: "standard", contextWindow: 128_000, maxOutputTokens: 1_000,
};
const reasoning: ModelProfile = {
  id: "reasoning", model: process.env.REASONING_MODEL!, tier: "reasoning", reasoningEffort: "medium",
  serviceMode: "standard", contextWindow: 128_000, maxOutputTokens: 4_000,
};

const harness = new Harness({
  llm: new OpenAICompatibleLlm({ baseUrl: process.env.OPENAI_BASE_URL!, apiKey: process.env.OPENAI_API_KEY! }),
  profiles: { fast, reasoning },
  tools: [{ name: "current_time", description: "Current date and time (ISO 8601, UTC)", effect: "read", params: {} }],
  toolPort: new FunctionTools({ current_time: () => new Date().toISOString() }),
  permissions: { allowEffects: ["read"] },
  specialists: [{
    name: "clock",
    instructions: "Answer questions about the time with current_time. Finish citing the act-id.",
    tools: ["current_time"],
    canFinish: true,
  }],
});

const result = await harness.run({ input: "What time is it in UTC?" });
console.log(result.answer.text);
console.log(result.status, result.completion);
console.log(result.trace.join("\n"));
```

Each tool declares whether its `effect` is `read` or `write`. `permissions`
sets which effects this deployment may run at all; **the default allows
both**. Constraints passed to a turn (`read_only`, `deny_tool`, or a plain
`instruction`) narrow that further for one task. `RunResult` also returns the
executed `actions`, the appended `events`, every `modelCalls` entry with its
tier, the `decisions`, and token `usage`.

For a conversation, open a session and keep calling `turn`. Agents keep their
transcripts between turns. `newTask: true` clears the workspace and
constraints and starts fresh transcripts:

```ts
import { consolidate, idle } from "../src/index.ts";

const session = await harness.openSession("demo");
await session.turn({ input: "Why did the nightly export fail?" });
await session.turn({ input: "Summarise that in one sentence." });
await session.turn({ input: "Different question: what files are here?", newTask: true });

// Later, when you decide to (consolidate and idle are exported from src/index.ts):
const { llm } = harness.config;
const sleep = await consolidate({
  llm, profile: reasoning, tools: harness.config.tools,
  episodes: harness.episodes, knowledge: harness.knowledge, habits: harness.habits,
});
console.log(sleep.learned, sleep.habitsProposed); // habits stay pending until approved
const { ideas } = await idle({ llm, profile: fast, knowledge: harness.knowledge, episodes: harness.episodes });
await session.turn({ input: "Did tonight's export finish?", preparedIdeas: ideas });
```

### Model profiles and service modes

A `ModelProfile` sets three independent things. `tier` is the model class
(`fast` for specialists and the narrator, `reasoning` for the executive).
`reasoningEffort` is how much the model thinks per call. `serviceMode` is the
provider latency lane: `standard`, `fast` or `ultrafast`, which changes how
quickly a request is served, not which model serves it.

A turn with `urgency: "high"` moves `standard` profiles to `fast` and leaves
the model, tier and effort alone. The OpenAI-compatible adapter sends a
`service_tier` only for lanes you map in `serviceTiers`. Otherwise the
provider's default lane is used.

## Architecture

All session state is a fold over an append-only event log stored through the
`Storage` port, so a session can be reopened by id. Agents do not share a
transcript: each receives the events it is allowed to see as text appended to
its own. If a specialist's action turns out as predicted, the outcome
goes only to that specialist. Surprises and failures are broadcast. Long tool
outputs are shown as previews, and an agent can ask to read the full record in
private.

| Port | You provide | Shipped |
| --- | --- | --- |
| `LlmClient` | chat completions for every agent | `OpenAICompatibleLlm` |
| `DecisionModel` | typed probabilities for routing/escalation (optional) | none |
| `Embedder` | vectors for memory recall (optional) | `OpenAICompatibleEmbedder` |
| `LiveModel` | `AsyncIterable` media chunks → text observations (optional) | none |
| `Storage` | key/value + append-only logs | `MemoryStorage`, `FileStorage` |
| `ToolPort` | runs a tool call | `FunctionTools` |

Source map (`src/`):

- `harness.ts` holds configuration, limits and shared services. `session.ts`
  holds the turn loop.
- `agents.ts` builds agent prompts, with a shared block first and the role
  last so prefixes match. `transcript.ts` keeps per-agent append-only
  transcripts and handles compaction.
- `state.ts` holds the event log, workspace and evidence index. `types.ts`
  holds the shared data shapes.
- `selector.ts` ranks proposals. `tools.ts` has the policy gate and the
  actuator. `monitor.ts` tracks surprise, stalls and escalation reasons.
- `validate.ts` checks model output at the boundary. `narrator.ts` writes the
  grounded answer.
- `memory.ts` has the episode, knowledge and habit stores. `consolidation.ts`
  is sleep and `background.ts` is idle.
- `io/` holds the provider-neutral ports. `adapters/` holds the
  OpenAI-compatible and file-storage implementations.

Tests in `test/` use scripted models, a scripted decision model and a
word-hash embedder. They check behaviour (policy, routing, prefix stability,
compaction, persistence, media handling), not model quality.

## Current limitations

- **No live provider tests.** The OpenAI-compatible adapter is only checked
  against an injected `fetch`. Cache reuse is never measured: the
  "cached" token counts in tests are simulated, and real cache hits depend on
  the provider.
- **No decision-model or live-model adapter.** Without a decision model,
  routing is rule-based: every specialist with visible news wakes on each
  step. In the trace above, `ops` and `data` are called every step even when
  they have nothing to say.
- **Only the turn loop runs by itself.** Sleep and idle are explicit calls.
  There is no scheduler or background worker, and the harness runs nothing
  alongside a turn. The only concurrency is the parallel specialist calls
  within a step. Escalation is not an interrupt: a surprise is handled on the
  next step, after another specialist round. Audio/video perception is a
  bounded phase before the steps, not a stream that runs alongside them.
- **Budgets cover chat calls only.** Decision-model calls and
  consolidate/idle calls do not count toward `maxModelCalls` or a turn's
  `usage`.
- **Persistence is simple.** Transcripts live in memory, so a reopened
  session rebuilds each agent's prompt from a fresh snapshot. Each session
  runs one turn at a time; callers must serialize turns on a shared session.
  Memory updates are not concurrency-safe, and `FileStorage` is intended for a
  single process without concurrent writers. If a storage write
  fails, the turn ends, and the log may be missing an event the in-memory
  view already saw.
- **Rough estimates and simple matching.** Tokens are estimated as
  characters ÷ 4, and each image counts as a flat 1,000 tokens. Habits fire on
  a case-insensitive substring of the goal. Recall is brute-force cosine
  similarity with a `minSimilarity` you need to tune for your embedder.
- **"Grounded" means only that the cited ids exist.** It does not check that
  the evidence supports the prose. Approving habits is left to you; the chat
  example has no approval command.
