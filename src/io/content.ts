/**
 * Multimodal content. Everything that flows into or out of a model is a list of parts.
 */
export type TextPart = { type: "text"; text: string };

export type MediaPart = {
  type: "image" | "audio" | "video";
  mimeType: string;
  /** Either inline bytes or a URL the adapter can fetch. */
  data?: Uint8Array;
  url?: string;
};

export type Part = TextPart | MediaPart;

export const text = (value: string): TextPart => ({ type: "text", text: value });

/** Flatten parts to text (media becomes a placeholder like "[image]"). */
export function textOf(parts: Part[]): string {
  return parts.map((p) => (p.type === "text" ? p.text : `[${p.type}]`)).join("\n");
}
