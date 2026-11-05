/**
 * Multimodal input as a list of parts. Media is carried inline (base64) or by
 * URL so that it survives JSON storage unchanged.
 *
 * Routing: text and images go to chat models (images only to profiles with
 * `vision: true`); audio and video go through a `LiveModel` (see live.ts).
 */
export type TextPart = { type: "text"; text: string };
export type MediaPart = { type: "image" | "audio" | "video"; mimeType: string; base64?: string; url?: string };
export type ContentPart = TextPart | MediaPart;

export const text = (value: string): TextPart => ({ type: "text", text: value });

export function isMedia(part: ContentPart): part is MediaPart {
  return part.type !== "text";
}

export function toParts(input: string | readonly ContentPart[]): ContentPart[] {
  return typeof input === "string" ? [text(input)] : [...input];
}

/** The text of the parts, joined by newlines. Media is skipped. */
export function textOf(parts: readonly ContentPart[]): string {
  return parts.flatMap((p) => (p.type === "text" ? [p.text] : [])).join("\n");
}

export function describeMedia(part: MediaPart): string {
  return `${part.type} (${part.mimeType})`;
}
