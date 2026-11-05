import { textOf, type MediaPart, type Part } from "../io/content";
import type { LiveModel } from "../io/live";
import type { BrainEvent } from "../core/gateway";

export type Input = { source: string; parts: Part[] };

/**
 * Turns raw input into percepts.
 *  - text and images go straight through (the main models are multimodal);
 *  - audio and video go to the live model, which streams back short text observations,
 *    steered by `focus` (top-down attention from the current goal).
 *
 * Brain analogue: sensory cortices, with attention shaping what gets reported.
 */
export class Senses {
  constructor(private readonly live?: LiveModel) {}

  async *perceive(input: Input, focus: string): AsyncIterable<BrainEvent> {
    const texts = input.parts.filter((p) => p.type === "text");
    const images = input.parts.filter((p) => p.type === "image");
    const streams = input.parts.filter((p): p is MediaPart => p.type === "audio" || p.type === "video");

    if (texts.length || images.length) {
      yield { kind: "percept", source: input.source, text: textOf(texts) || "(image)", attachments: images.length ? images : undefined };
    }

    for (const media of streams) {
      if (!this.live) throw new Error(`Got ${media.type} input but no live model is configured`);
      for await (const observation of this.live.watch(media, focus)) {
        yield { kind: "percept", source: input.source, text: `[${media.type}] ${observation}` };
      }
    }
  }
}
