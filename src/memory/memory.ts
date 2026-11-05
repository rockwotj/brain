import type { Clock } from "../io/clock";
import type { Embedder } from "../io/embeddings";
import type { Storage } from "../io/storage";
import { EpisodicMemory } from "./episodic";
import { ProspectiveMemory } from "./prospective";
import { SemanticMemory } from "./semantic";
import { SkillLibrary } from "./skills";

/** The long-term memory systems, kept separate so new experience can't overwrite old knowledge. */
export class Memory {
  readonly episodic: EpisodicMemory;
  readonly semantic: SemanticMemory;
  readonly skills: SkillLibrary;
  readonly prospective: ProspectiveMemory;

  constructor(storage: Storage, embedder: Embedder, clock: Clock) {
    this.episodic = new EpisodicMemory(storage, embedder, clock);
    this.semantic = new SemanticMemory(storage, embedder, clock);
    this.skills = new SkillLibrary(storage, embedder);
    this.prospective = new ProspectiveMemory(storage, embedder);
  }

  /** Cue-based recall across facts and past episodes, as lines for a prompt. */
  async recall(cue: string): Promise<string[]> {
    const facts = await this.semantic.recall(cue);
    const episodes = await this.episodic.recall(cue);
    return [
      ...facts.map((f) => `fact (v${f.version}): ${f.text}`),
      ...episodes.map((e) => `earlier (${new Date(e.at).toISOString()}): ${e.what}`),
    ];
  }
}
