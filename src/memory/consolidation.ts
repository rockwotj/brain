import type { Clock } from "../io/clock";
import { generateJson, type LLM } from "../io/llm";
import type { Storage } from "../io/storage";
import type { Memory } from "./memory";

const SLEEP_PROMPT = `You consolidate an agent's recent experience into long-term memory.
You get a chronological log of episodes. Extract:
- "facts": durable, general knowledge worth remembering (about the user, the world, tools). Skip one-off details.
- "skills": procedures that worked and are likely to recur. Each has a "trigger" (an example of the
  situation, phrased like the input that would trigger it) and short plain-language "steps".
Answer with ONLY a JSON object: {"facts": string[], "skills": [{"trigger": string, "steps": string[]}]}`;

type SleepOutput = { facts?: string[]; skills?: { trigger: string; steps: string[] }[] };

/**
 * Offline replay: turns recent episodes into semantic facts and skills.
 * Episodes themselves are never modified.
 *
 * Brain analogue: sleep-dependent memory consolidation (hippocampal replay → neocortex).
 */
export class Consolidation {
  constructor(
    private readonly memory: Memory,
    private readonly system2: LLM,
    private readonly storage: Storage,
    private readonly clock: Clock,
  ) {}

  async sleep(): Promise<{ episodes: number; facts: number; skills: number }> {
    const lastSlept = (await this.storage.get<number>("sleep/last")) ?? 0;
    const episodes = await this.memory.episodic.since(lastSlept);
    if (!episodes.length) return { episodes: 0, facts: 0, skills: 0 };

    const log = episodes
      .map((e) => `[${new Date(e.at).toISOString()}] (${e.source}, surprise ${e.surprise.toFixed(2)}, valence ${e.valence.toFixed(2)}${e.goal ? `, goal: ${e.goal}` : ""}) ${e.what}`)
      .join("\n");

    const { value } = await generateJson<SleepOutput>(this.system2, {
      system: SLEEP_PROMPT,
      messages: [{ role: "user", content: [{ type: "text", text: log }] }],
      thinking: "medium",
    });

    for (const fact of value.facts ?? []) await this.memory.semantic.learn(fact);
    for (const skill of value.skills ?? []) await this.memory.skills.add(skill.trigger, skill.steps);

    await this.storage.put("sleep/last", this.clock());
    return { episodes: episodes.length, facts: value.facts?.length ?? 0, skills: value.skills?.length ?? 0 };
  }
}
