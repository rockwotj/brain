import type { MediaPart } from "./content";

/**
 * A specialist realtime model for audio/video. It turns a media stream into a stream of
 * short text observations ("user says: ...", "a person enters the room").
 *
 * `focus` is top-down attention: what the brain currently cares about, so the model can
 * report what matters instead of everything it hears or sees.
 */
export interface LiveModel {
  readonly name: string;
  watch(media: MediaPart, focus: string): AsyncIterable<string>;
}
