# brain

An LLM agent harness structured around how the brain organizes thought. **Proof of concept:** it's
built to be read and tinkered with, not tuned or hardened. The design rationale is in
[`docs/architecture.md`](docs/architecture.md).

```sh
bun install
bun test            # unit tests for the context, workspace and selector
bun run typecheck

# chat with it over any OpenAI-compatible API
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=... \
SYSTEM1_MODEL=... SYSTEM2_MODEL=... EMBEDDING_MODEL=... bun run chat
```

## The idea in one paragraph

Each LLM call acts like a patch of cortex: good at completing patterns, with no sense of priorities.
The harness adds what the rest of the brain provides around it:
- a tiny shared **workspace** that loops compete to get into
- **predictions** on every action, so only surprises get attention
- an action **selector** that defaults to "no"
- a **monitor** that notices when things go wrong
- global **modulators** that retune the whole system
- separate **memory** systems with offline **sleep**
- a **fast habit path** on a cheap model

## Map: brain → code

| Brain | Code | What it does |
|---|---|---|
| Senses (+ top-down attention) | `src/perception/senses.ts` | Text and images pass through. Audio and video go to a live model, steered by the current goal. |
| Thalamus + predictive coding | `src/core/gateway.ts` | Scores each event. A cheap System 1 judge compares tool results with their predictions. |
| Global workspace | `src/core/workspace.ts` | About 7 slots. Old items decay, contradicting bids suppress each other and each source is capped. |
| Prefrontal cortex | `src/core/executive.ts` | Goal stack and top-down bias toward the current goal. |
| Cortical areas | `src/loops/specialist.ts`, `src/loops/roster.ts` | Narrow LLM loops (planner, critic, user model, overseer), each with its own cache-stable context. |
| Habits (striatum) | `src/loops/reflex.ts` | A familiar input runs a learned skill on System 1 and skips the workspace. |
| Amygdala | `src/core/risk.ts` | Quick risk tag on proposals. It can only raise risk. |
| Basal ganglia | `src/core/selector.ts` | Default no-go, one action per effector per tick, refractory period and learned tool values. |
| Anterior cingulate | `src/core/monitor.ts` | Detects repetition, error streaks, disagreement and stalls, then escalates. |
| Neuromodulators + interoception | `src/core/modulators.ts` | Urgency, explore, precision, progress and patience; budget is tracked as the body's state. These set model tier, thinking, temperature and parallelism. |
| Hippocampus | `src/memory/episodic.ts` | Append-only episode log. |
| Neocortex | `src/memory/semantic.ts` | Facts. Near-duplicates revise an existing fact and keep its history. |
| Habits / procedures | `src/memory/skills.ts` | Trigger → steps, plus a success rate. |
| Prospective memory | `src/memory/prospective.ts` | "When X happens, do Y" and "at time T, do Y". |
| Sleep | `src/memory/consolidation.ts` | Replays episodes into facts and skills on System 2. |

## One tick (`Brain.tick()` in `src/brain.ts`)

```
events ──► gateway ─┬─ familiar input ─────────► reflex (System 1 habit)
                    ├─ result matched prediction ► back to the loop that acted only
                    └─ salient / surprising ─────► workspace competition (with goal bias)
workspace winners ──► loops that subscribe to them ──► thoughts (bids) + action proposals
proposals ──► risk tagger ──► selector (one per effector) ──► tools ──► result events (next tick)
monitor + modulators ──► escalate model tier / thinking, alert the overseer, retune
```

Problems flow **up**: only the project-level `overseer` listens to errors and alerts.
Goals flow **down**: the overseer posts goal bids for the planner, and goal changes are published to every loop.

## Keeping prompt caches warm

`src/context/stable-context.ts` gives every loop a context whose prefix only grows:

- **Append-only.** Nothing that was already sent is edited. `cacheUpTo` tells adapters where the unchanged prefix ends, so they can place a cache breakpoint.
- **Deltas instead of rewrites.** Changes to shared facts, such as a new goal, are appended as `[context update]` notes on the next user message. The system prompt stays the same.
- **Compact only when the cache would break anyway.** The prefix is rewritten in two cases:
  - the context is about 80% full, just before it overflows
  - the provider cache TTL has passed since the last call and there are deltas to fold or a large log, so the next call would be uncached anyway
  
  Compaction folds the deltas into a new system prompt and summarizes the log on System 1.
- **Stable rendering.** Tools and facts are sorted, and each loop's instructions never change.
- **Shared prompts for fixed tasks.** The reflex, judge, risk and summarizer prompts are constant, so they stay cached across calls.
- **Tier changes are cache misses.** Caches are per model, so switching tiers forces an uncached call. Each loop stays on its default tier unless the monitor escalates it.

## System 1 / System 2

`Models` has two tiers plus optional live and embedding models:

- **System 1** (`Models.system1`): a small, fast model. It handles:
  - habits
  - prediction checks
  - risk tags
  - summaries
  - the critic and user-model loops
- **System 2** (`Models.system2`): a strong model for:
  - the planner and overseer
  - escalations
  - sleep
- `chooseRuntime()` picks the tier and thinking level for each call:
  - escalation or a high explore level moves a call up to System 2
  - a low budget moves calls down to System 1
  - urgency limits thinking

## Bring your own providers

Everything external sits behind small interfaces in `src/io/`:

| Interface | File | Notes |
|---|---|---|
| `LLM` | `io/llm.ts` | `generate({ system, messages, cacheUpTo, thinking, temperature, json })` |
| `Embedder` | `io/embeddings.ts` | `embed(texts)` |
| `LiveModel` | `io/live.ts` | `watch(media, focus)` returns an async stream of text observations |
| `Storage` | `io/storage.ts` | `get` / `put` / `delete` / `list(prefix)`. All state goes through this. |
| `Tool` | `io/tools.ts` | `run(args)`, plus an `effector` and `risk` |

Adapters in `src/adapters/`:

- `file-storage.ts`: one JSON file per key
- `openai-compatible.ts`: chat and embeddings over plain `fetch`. Untested against live endpoints.

`MemoryStorage` (in `io/storage.ts`) keeps everything in process memory. There's no live-model
adapter yet, so audio/video input needs one before it works.

```ts
const brain = await Brain.create({
  models: { system1, system2, live, embedder },
  storage: new FileStorage(".brain-data"),
  tools: [myTool],
  onSay: (text) => console.log(text),
});
await brain.hear({ source: "user", parts: [{ type: "text", text: "..." }] });
await brain.run();   // tick until things settle
await brain.idle();  // reminders + sleep
```

## Status and known gaps

- Verified by type-checking and unit tests of the stable context, workspace and selector. The full brain loop and the OpenAI-compatible adapter have not been run against real models, and nothing is benchmarked.
- No live-model adapter, so audio/video input isn't usable yet.
- Memory search compares against every stored item. There is no vector index and no forgetting or pruning yet.
- Actions run one tick at a time. There are no long-running or streaming tools.
- The live model returns text observations only. There is no keyframe pass-through or barge-in.
- Specialist output is JSON from a prompt. Provider-native structured output or tool calling would be sturdier.
- The thresholds and constants are first guesses.
