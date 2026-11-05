import type { Part } from "../io/content";
import { generateJson, type LLM } from "../io/llm";
import type { Modulators, Proposal } from "./types";

/** Everything that can happen to the brain. */
export type BrainEvent =
  | { kind: "percept"; source: string; text: string; attachments?: Part[] }
  | { kind: "result"; proposal: Proposal; ok: boolean; output: string }
  | { kind: "timer"; text: string };

export type Appraisal = {
  event: BrainEvent;
  /** One-line description used for logging, memory and bids. */
  text: string;
  /** 0..1 prediction error. */
  surprise: number;
  /** 0..1 how strongly this should bid for attention. */
  salience: number;
};

const JUDGE_PROMPT = `You compare a prediction with what actually happened.
Answer with ONLY a JSON object: {"match": number between 0 and 1}.
1 means the outcome is what was predicted (wording may differ), 0 means it is completely different.`;

/**
 * The gateway scores every incoming event before anything else sees it.
 *
 * Brain analogue: thalamic gating + predictive processing. Every action carries a prediction
 * (its efference copy); results that match are boring and stay local, mismatches are loud.
 * Judging the match is a System 1 job.
 */
export class Gateway {
  constructor(private readonly judge: LLM) {}

  async appraise(event: BrainEvent, mods: Modulators): Promise<Appraisal> {
    switch (event.kind) {
      case "percept":
        // Input from the user always matters; other sensors matter a bit less.
        return { event, text: `${event.source}: ${event.text}`, surprise: 1, salience: event.source === "user" ? 1 : 0.6 };

      case "timer":
        return { event, text: event.text, surprise: 0.5, salience: 0.6 };

      case "result": {
        const { proposal: p, ok, output } = event;
        const text = `${p.tool}(${JSON.stringify(p.args)}) → ${ok ? "" : "ERROR: "}${output.slice(0, 2000)}`;
        const match = ok ? await this.match(p.expect, output) : 0;
        // Precision weighting: a confident prediction that misses is more surprising.
        const surprise = ok ? (1 - match) * (0.5 + 0.5 * p.confidence) : 1;
        // How loud the surprise is depends on how much we currently trust new evidence.
        const salience = surprise * (0.5 + 0.5 * mods.precision);
        return { event, text, surprise, salience };
      }
    }
  }

  private async match(expected: string, actual: string): Promise<number> {
    if (expected.trim() === actual.trim()) return 1;
    const { value } = await generateJson<{ match: number }>(this.judge, {
      system: JUDGE_PROMPT,
      messages: [{ role: "user", content: [{ type: "text", text: `Predicted: ${expected}\nActual: ${actual.slice(0, 2000)}` }] }],
      thinking: "none",
      temperature: 0,
    });
    return Math.min(1, Math.max(0, Number(value.match) || 0));
  }
}
