/**
 * Harness = configuration + shared services (tools, memory, storage).
 * Work happens in sessions (session.ts): `openSession()` for multi-turn use,
 * `run()` for a one-shot task in a fresh session.
 */
import type { SpecialistSpec } from "./agents.ts";
import type { Embedder } from "./io/embeddings.ts";
import { roughTokenEstimate, type LlmClient, type Profiles, type TokenEstimator } from "./io/llm.ts";
import type { LiveModel } from "./io/live.ts";
import { MemoryStorage, type Storage } from "./io/storage.ts";
import { EpisodicStore, HabitStore, KnowledgeStore } from "./memory.ts";
import { Session, type RunResult, type TurnOptions } from "./session.ts";
import { StateStore, type WorkspaceBudget } from "./state.ts";
import { Actuator, type Permissions, type ToolPort, type ToolSpec } from "./tools.ts";

export type Limits = {
  maxSteps: number;
  maxModelCalls: number;
  maxExecutiveCalls: number;
  stallSteps: number;
  carefulSteps: number;
  maxLiveObservations: number;
};
const DEFAULT_LIMITS: Limits = { maxSteps: 12, maxModelCalls: 60, maxExecutiveCalls: 3, stallSteps: 2, carefulSteps: 2, maxLiveObservations: 10 };

export type EvidenceSettings = {
  /** Characters of a tool output shown inline in updates (rest via read_record). */
  previewChars: number;
  /** Characters delivered per read_record request. */
  recordChunkChars: number;
  /** Characters of each output shown to the narrator. */
  narratorRecordChars: number;
  /** Most recent records listed in a snapshot's evidence index. */
  indexSize: number;
};
const DEFAULT_EVIDENCE: EvidenceSettings = { previewChars: 240, recordChunkChars: 6_000, narratorRecordChars: 2_000, indexSize: 30 };

export type RecallSettings = { knowledge: number; episodes: number; minSimilarity: number };
/** minSimilarity depends on the embedding model; tune it for yours. */
const DEFAULT_RECALL: RecallSettings = { knowledge: 2, episodes: 1, minSimilarity: 0.3 };

export type HarnessConfig = {
  llm: LlmClient;
  profiles: Profiles;
  tools: readonly ToolSpec[];
  toolPort: ToolPort;
  permissions?: Permissions;
  specialists: readonly SpecialistSpec[];
  executiveInstructions?: string;
  /** Persistence for session events, records and memory. Default: in-process MemoryStorage. */
  storage?: Storage;
  /** Enables memory recall by cosine similarity. Without it nothing is recalled. */
  embedder?: Embedder;
  /** Required for audio/video input; without it such input is rejected. */
  live?: LiveModel;
  workspace?: WorkspaceBudget;
  evidence?: Partial<EvidenceSettings>;
  recall?: Partial<RecallSettings>;
  estimate?: TokenEstimator;
  clock?: () => number;
  limits?: Partial<Limits>;
};

export class Harness {
  readonly storage: Storage;
  readonly episodes: EpisodicStore;
  readonly knowledge: KnowledgeStore;
  readonly habits: HabitStore;
  readonly actuator: Actuator;
  readonly limits: Limits;
  readonly evidence: EvidenceSettings;
  readonly recall: RecallSettings;
  readonly estimate: TokenEstimator;

  constructor(readonly config: HarnessConfig) {
    this.storage = config.storage ?? new MemoryStorage();
    this.episodes = new EpisodicStore(this.storage, config.embedder);
    this.knowledge = new KnowledgeStore(this.storage, config.embedder);
    this.habits = new HabitStore(this.storage);
    this.actuator = new Actuator(config.tools, config.toolPort, config.permissions ?? { allowEffects: ["read", "write"] });
    this.limits = { ...DEFAULT_LIMITS, ...config.limits };
    this.evidence = { ...DEFAULT_EVIDENCE, ...config.evidence };
    this.recall = { ...DEFAULT_RECALL, ...config.recall };
    this.estimate = config.estimate ?? roughTokenEstimate;
  }

  /** Open (or reopen from storage) a session. Agents' transcripts persist across its turns. */
  async openSession(id: string = crypto.randomUUID()): Promise<Session> {
    const state = await StateStore.open({
      storage: this.storage,
      log: `sessions/${id}/events`,
      previewChars: this.evidence.previewChars,
      indexSize: this.evidence.indexSize,
      ...(this.config.workspace ? { budget: this.config.workspace } : {}),
    });
    return new Session(id, state, this);
  }

  /** One task in a fresh session. */
  async run(opts: TurnOptions): Promise<RunResult> {
    return (await this.openSession()).turn(opts);
  }
}
