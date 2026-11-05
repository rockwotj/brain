import { text, textOf, type Part } from "../io/content";
import type { LLM, Message } from "../io/llm";
import type { Bid } from "../core/types";

/** How a broadcast looks to a loop. Attachments (images) ride along as real parts. */
export function renderBroadcast(tick: number, items: Bid[], recalled: string[]): Part[] {
  const lines = items.map((b) => `- [${b.id}] ${b.kind} from ${b.source} (salience ${b.salience.toFixed(2)}): ${b.content}`);
  let body = `Tick ${tick}. New in your attention:\n${lines.join("\n")}`;
  if (recalled.length) body += `\n\nRecalled from memory (reconstructions, may be wrong):\n${recalled.map((r) => `- ${r}`).join("\n")}`;
  return [text(body), ...items.flatMap((b) => b.attachments ?? [])];
}

const SUMMARIZE_PROMPT = `Summarize this agent loop's working history so it can continue without the full log.
Keep: open goals, decisions and why, results that still matter, unresolved problems. Drop chatter. Be brief.`;

/** Compaction summarizer (a System 1 job). */
export async function summarize(llm: LLM, messages: Message[], previous: string): Promise<string> {
  const transcript = messages.map((m) => `${m.role}: ${textOf(m.content)}`).join("\n\n");
  const res = await llm.generate({
    system: SUMMARIZE_PROMPT,
    messages: [{ role: "user", content: [text(`${previous ? `Previous summary:\n${previous}\n\n` : ""}New history:\n${transcript}`)] }],
    thinking: "none",
    temperature: 0,
  });
  return res.text.trim();
}

export const ACTION_FORMAT = `"actions": [{"tool": string, "args": object, "expect": "what you predict the result will be", "confidence": 0-1, "why": string}]`;
