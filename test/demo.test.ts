import { expect, test } from "bun:test";

test("demo runs deterministically and prints a clear trace", async () => {
  const run = async () => {
    const proc = Bun.spawn(["bun", "examples/demo.ts"], { cwd: `${import.meta.dir}/..`, stdout: "pipe", stderr: "pipe" });
    const out = await new Response(proc.stdout).text();
    expect(await proc.exited).toBe(0);
    return out;
  };
  const out = await run();
  for (const expected of [
    "wake [triage, ops, data] in parallel on s1-small",
    "reflex read_job_logs",
    "deliberation skipped",
    "(SURPRISE)",
    "escalate (surprise) -> executive on s2-reasoner",
    "blocked restart_job",
    "STATUS: finished in 4 steps",
    "grounded=true",
    "learned kn-2",
    "[pending]",
    "rejected: lesson without valid provenance",
    "tool calls during idle: 0",
  ]) expect(out).toContain(expected);
  expect(await run()).toBe(out);
});
