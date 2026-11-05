/**
 * Toy "models" for the offline demo and tests. They pattern-match prompts to fake just enough
 * behavior to exercise the harness. They are NOT intelligent; swap in real adapters for that.
 */
import { ScriptedLLM } from "../src/adapters/scripted-llm";
import { textOf, type MediaPart } from "../src/io/content";
import type { GenerateRequest } from "../src/io/llm";
import type { LiveModel } from "../src/io/live";
import type { Tool } from "../src/io/tools";

/** The part of the latest message that's new (ignores skill text and recalled memories). */
function attention(req: GenerateRequest): string {
  const last = req.messages.at(-1);
  if (!last) return "";
  const full = textOf(last.content);
  const start = full.indexOf("New in your attention:");
  return (start === -1 ? full : full.slice(start)).split("Recalled from memory")[0]!;
}

const reply = (text: string) => ({ tool: "reply", args: { text }, expect: `said: ${text}`, confidence: 0.9 });

function calculatorStep(now: string) {
  const ask = now.match(/what is (\d+)\s*\*\s*(\d+)/i);
  if (!ask) return undefined;
  const [a, b] = [Number(ask[1]), Number(ask[2])];
  return { tool: "calculator", args: { expression: `${a} * ${b}` }, expect: String(a * b), confidence: 0.9, why: "multiply" };
}

export function toySystem2() {
  return new ScriptedLLM("toy-system2", (req) => {
    const now = attention(req);

    if (req.system.includes('You are the "planner"')) {
      if (/calculator\(.*\) → (ERROR: )?NaN/.test(now)) return { actions: [reply("My calculator choked on that one. Let me know if you want me to work it out by hand.")] };
      const result = now.match(/calculator\(.*\) → (.+)/);
      if (result) return { actions: [reply(`The answer is ${result[1]}.`)] };
      const step = calculatorStep(now);
      return step ? { actions: [step] } : {};
    }

    if (req.system.includes("You consolidate")) {
      const question = now.match(/user: (What is [^\n]*)/);
      return {
        facts: ["The user asks quick arithmetic questions and wants short answers."],
        skills: question ? [{ trigger: question[1], steps: ["Call calculator with the expression.", "Reply with 'The answer is <result>.'"] }] : [],
      };
    }

    return {};
  });
}

export function toySystem1() {
  return new ScriptedLLM("toy-system1", (req) => {
    const now = attention(req);

    if (req.system.includes("You compare a prediction")) {
      const [, predicted = "", actual = ""] = now.match(/Predicted: (.*)\nActual: ([\s\S]*)/) ?? [];
      const numeric = /^-?\d+(\.\d+)?$/.test(predicted.trim());
      return { match: !numeric || actual.trim() === predicted.trim() ? 1 : 0 };
    }

    if (req.system.includes("You run a known habit")) {
      const result = now.match(/calculator\(.*\) → (.+)/);
      if (result) return { actions: [reply(`The answer is ${result[1]}.`)], done: true };
      const step = calculatorStep(now);
      return step ? { actions: [step], done: false } : { done: true };
    }

    if (req.system.includes("Summarize this agent loop")) return "Earlier: answered arithmetic questions.";

    return {}; // critic and user-model have nothing to add
  });
}

/** Pretends to transcribe audio: the "audio" bytes are just UTF-8 text. */
export const toyLive: LiveModel = {
  name: "toy-live",
  async *watch(media: MediaPart) {
    yield new TextDecoder().decode(media.data);
  },
};

/** A calculator with a deliberate bug: big results come back as NaN (to show surprise handling). */
export const calculator: Tool = {
  name: "calculator",
  description: "Evaluate an arithmetic expression.",
  parameters: { expression: "string, e.g. '17 * 23'" },
  effector: "compute",
  risk: "low",
  run: async ({ expression }) => {
    const expr = String(expression);
    if (!/^[\d\s+\-*/().]+$/.test(expr)) throw new Error(`not arithmetic: ${expr}`);
    const value = Function(`return (${expr})`)() as number;
    return value > 100_000 ? "NaN" : String(value);
  },
};
