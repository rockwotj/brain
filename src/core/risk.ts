import { generateJson, type LLM } from "../io/llm";
import { describeTools, type Risk, type Tool } from "../io/tools";
import type { Proposal } from "./types";

const RISK_PROMPT = `You tag proposed agent actions by risk of harm, data loss, spending money or acting
on someone's behalf without permission.
Answer with ONLY a JSON object mapping proposal id to "low", "medium" or "high", e.g. {"p1": "low"}.`;

const rank: Record<Risk, number> = { low: 0, medium: 1, high: 2 };

/**
 * A quick risk check that runs before the selector. Tools declare a base risk; for "medium" tools,
 * a System 1 model looks at the specific call and may raise it. Risk only ever goes up.
 *
 * Brain analogue: the amygdala's fast, coarse threat pathway.
 */
export class RiskTagger {
  constructor(private readonly system1: LLM) {}

  async tag(proposals: Proposal[], tools: Map<string, Tool>): Promise<Proposal[]> {
    const tagged = proposals.map((p) => ({ ...p, risk: tools.get(p.tool)?.risk ?? "high" }));
    const unsure = tagged.filter((p) => p.risk === "medium");
    if (!unsure.length) return tagged;

    const used = [...new Set(unsure.map((p) => p.tool))].map((name) => tools.get(name)!);
    const { value } = await generateJson<Record<string, Risk>>(this.system1, {
      system: RISK_PROMPT,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: `Tools:\n${describeTools(used)}\n\nProposals:\n${unsure.map((p) => `${p.id}: ${p.tool}(${JSON.stringify(p.args)}) — ${p.why ?? ""}`).join("\n")}`,
            },
          ],
        },
      ],
      thinking: "none",
      temperature: 0,
    });

    return tagged.map((p) => {
      const judged = value[p.id];
      return judged && rank[judged] > rank[p.risk] ? { ...p, risk: judged } : p;
    });
  }
}
