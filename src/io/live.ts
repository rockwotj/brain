import type { MediaPart } from "./content.ts";

/**
 * Specialist audio/video model port (e.g. a realtime speech or vision model).
 * It turns one media input into short text observations ("user says: ...").
 * `focus` is top-down attention: the current goal, so it reports what matters.
 *
 * No concrete adapter ships with this PoC; inject one to accept audio/video.
 */
export interface LiveModel {
  readonly name: string;
  observe(media: MediaPart, focus: string, signal?: AbortSignal): AsyncIterable<string>;
}
