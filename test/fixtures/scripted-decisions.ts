/**
 * TEST FIXTURE ONLY. A stand-in decision model: answers come from a script and
 * every request is recorded. It is not a classifier and makes no judgments.
 */
import type { DecisionModel, DecisionRequest } from "../../src/index.ts";

export class ScriptedDecisions implements DecisionModel {
  readonly name = "scripted-decisions";
  readonly requests: DecisionRequest[] = [];

  constructor(private readonly answer: (request: DecisionRequest) => Record<string, unknown> | Promise<Record<string, unknown>>) {}

  async decide(request: DecisionRequest): Promise<Record<string, unknown>> {
    this.requests.push(request);
    return this.answer(request);
  }
}

export const yes = (probability: number) => ({ type: "boolean", probability });
