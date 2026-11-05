import type { Embedder } from "./embeddings";
import type { LiveModel } from "./live";
import type { LLM } from "./llm";

/** Everything the brain needs from model providers. */
export type Models = {
  /** Fast and cheap: habits, judging predictions, risk tags, summaries, light specialists. */
  system1: LLM;
  /** Slow and strong: deliberation, planning, consolidation ("sleep"). */
  system2: LLM;
  /** Realtime audio/video specialist. Optional if you only feed text and images. */
  live?: LiveModel;
  embedder: Embedder;
};
