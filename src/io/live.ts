import type { MediaPart } from "./content.ts";

/**
 * Port for a specialized audio/video model (e.g. a realtime speech or video model).
 * It reads an async stream of media chunks and yields short text observations
 * ("user says: ..."). `focus` is top-down attention: the current goal.
 *
 * Implementations should pull input lazily, stop when their output iterator is
 * closed, and honor `signal`. No concrete adapter ships with this PoC.
 */
export interface LiveModel {
  readonly name: string;
  observe(media: AsyncIterable<MediaPart>, focus: string, signal?: AbortSignal): AsyncIterable<string>;
}

/**
 * One audio or video input to a turn: a live source (microphone, camera) or a
 * recorded clip. The caller ends the stream (e.g. at the end of an utterance);
 * the harness also stops after `limits.maxLiveObservations` observations.
 */
export type LiveInput = { medium: "audio" | "video"; mimeType: string; chunks: AsyncIterable<MediaPart> };

/** A recorded clip as a one-chunk stream. */
export function clipInput(part: MediaPart): LiveInput {
  if (part.type === "image") throw new Error("images are not live input");
  return { medium: part.type, mimeType: part.mimeType, chunks: (async function* () { yield part; })() };
}

/**
 * Collect at most `max` observations. Closes the model's stream afterwards (so a
 * well-behaved model stops reading input) and rejects promptly when `signal`
 * aborts, even if the source is waiting for more input.
 */
export async function takeObservations(source: AsyncIterable<string>, max: number, signal?: AbortSignal): Promise<string[]> {
  signal?.throwIfAborted();
  const it = source[Symbol.asyncIterator]();
  const out: string[] = [];
  let onAbort: (() => void) | undefined;
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => reject(signal?.reason);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
  aborted.catch(() => {});
  let done = false;
  try {
    while (out.length < max) {
      const next = await Promise.race([it.next(), aborted]);
      if (next.done) {
        done = true;
        break;
      }
      out.push(next.value);
    }
  } catch (error) {
    void it.return?.()?.catch(() => {}); // ask the model to stop, without waiting on a stuck source
    throw error;
  } finally {
    if (onAbort) signal?.removeEventListener("abort", onAbort);
  }
  // Stopped early by the bound: close the model's stream so it stops reading input.
  if (!done) await it.return?.();
  return out;
}
