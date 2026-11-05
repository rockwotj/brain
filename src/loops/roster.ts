import type { SpecialistConfig } from "./specialist";

/**
 * The default society of loops. Add your own (researcher, coder...) the same way.
 * Levels form the timescale hierarchy: goals flow down as "goal" bids, problems flow up as
 * "error"/"alert" bids that only the higher loop listens to.
 */
export const defaultRoster: SpecialistConfig[] = [
  {
    id: "planner",
    level: "task",
    tier: "system2",
    canAct: true,
    listensTo: ["percept", "result", "error", "goal", "intention", "thought"],
    role: [
      "You turn what the user wants into the next concrete step and propose tool actions.",
      "When the request is satisfied, reply to the user. Use set_goal/complete_goal for work that spans many steps.",
      "Post a short thought only when other loops need to know (e.g. a non-obvious plan worth critiquing).",
    ].join(" "),
  },
  {
    id: "critic",
    level: "task",
    tier: "system1",
    canAct: false,
    listensTo: ["thought"],
    role: "You look for mistakes in other loops' plans and claims. If something is wrong, post a thought that contradicts it (include its id) with high salience. Otherwise return {}.",
  },
  {
    id: "user-model",
    level: "task",
    tier: "system1",
    canAct: false,
    listensTo: ["percept"],
    role: "You model the user: what they want, know and expect. Post one short inference only when it adds something non-obvious. Otherwise return {}.",
  },
  {
    id: "overseer",
    level: "project",
    tier: "system2",
    canAct: true,
    listensTo: ["error", "alert"],
    role: [
      "You only see problems that lower loops could not handle.",
      "Decide whether to change the goal (set_goal/complete_goal), change strategy (post a 'goal' thought for the planner), or ask the user for help (reply).",
    ].join(" "),
  },
];
