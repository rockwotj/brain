import { cosine, embedOne, type Embedder } from "../io/embeddings";
import type { Storage } from "../io/storage";

export type Skill = {
  id: string;
  /** The kind of situation that should trigger this skill. */
  trigger: string;
  /** Plain-language steps; the System 1 model fills in the details. */
  steps: string[];
  embedding: number[];
  uses: number;
  successes: number;
};

/**
 * Procedures compiled from repeated successful episodes. A matching cue lets the cheap
 * model run the skill instead of the full workspace deliberation.
 *
 * Brain analogue: basal ganglia / cerebellum habits.
 */
export class SkillLibrary {
  constructor(
    private readonly storage: Storage,
    private readonly embedder: Embedder,
  ) {}

  async add(trigger: string, steps: string[]): Promise<Skill> {
    const embedding = await embedOne(this.embedder, trigger);
    const same = await this.match(trigger, 0.95, false);
    const skill: Skill = same
      ? { ...same, trigger, steps, embedding }
      : { id: crypto.randomUUID(), trigger, steps, embedding, uses: 0, successes: 0 };
    await this.storage.put(`skills/${skill.id}`, skill);
    return skill;
  }

  /** Best skill whose trigger looks like this cue (and that hasn't been failing). */
  async match(cue: string, minScore = 0.8, requireReliable = true): Promise<Skill | undefined> {
    const query = await embedOne(this.embedder, cue);
    const skills = (await this.storage.list<Skill>("skills/")).map((s) => s.value);
    let best: { skill: Skill; score: number } | undefined;
    for (const skill of skills) {
      const score = cosine(query, skill.embedding);
      const reliable = skill.uses < 3 || skill.successes / skill.uses >= 0.5;
      if (score >= minScore && (!requireReliable || reliable) && (!best || score > best.score)) best = { skill, score };
    }
    return best?.skill;
  }

  async reinforce(id: string, success: boolean) {
    const skill = await this.storage.get<Skill>(`skills/${id}`);
    if (!skill) return;
    await this.storage.put(`skills/${id}`, { ...skill, uses: skill.uses + 1, successes: skill.successes + (success ? 1 : 0) });
  }
}
