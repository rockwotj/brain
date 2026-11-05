import type { Clock } from "../io/clock";
import type { Tool } from "../io/tools";
import type { Memory } from "../memory/memory";
import type { Executive } from "./executive";

/** Tools every brain has: speaking, managing goals, and remembering. */
export function builtinTools(deps: { say: (text: string) => void; executive: Executive; memory: Memory; clock: Clock }): Tool[] {
  const { say, executive, memory, clock } = deps;
  return [
    {
      name: "reply",
      description: "Say something to the user.",
      parameters: { text: "string" },
      effector: "voice",
      risk: "low",
      run: async ({ text }) => {
        say(String(text));
        return `said: ${text}`;
      },
    },
    {
      name: "set_goal",
      description: "Push a goal onto the goal stack. Every loop is told about it.",
      parameters: { text: "string" },
      effector: "executive",
      risk: "low",
      run: async ({ text }) => `goal ${(await executive.push(String(text))).id} set`,
    },
    {
      name: "complete_goal",
      description: "Mark a goal as done and remove it.",
      parameters: { id: "string" },
      effector: "executive",
      risk: "low",
      run: async ({ id }) => (executive.complete(String(id)) ? `goal ${id} completed` : `no goal ${id}`),
    },
    {
      name: "intend",
      description: "Remember to do something later: when a situation comes up and/or after some minutes.",
      parameters: { then: "string", when: "string (optional)", inMinutes: "number (optional)" },
      effector: "memory",
      risk: "low",
      run: async ({ then, when, inMinutes }) => {
        await memory.prospective.add({
          then: String(then),
          when: when ? String(when) : undefined,
          dueAt: inMinutes ? clock() + Number(inMinutes) * 60_000 : undefined,
        });
        return "intention stored";
      },
    },
  ];
}
