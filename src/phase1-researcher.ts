import { $ } from "bun";
import { existsSync } from "fs";
import { PHASE1_PROMPT } from "./prompts";

const RESEARCH_DIR = "research";
const PROBLEM_FILE = `${RESEARCH_DIR}/problem.md`;

export async function runPhase1(): Promise<boolean> {
  console.log("═══════════════════════════════════════");
  console.log("  Phase 1: Research Problem Verification");
  console.log("═══════════════════════════════════════\n");

  if (!existsSync(PROBLEM_FILE)) {
    console.error(`Error: ${PROBLEM_FILE} not found.`);
    console.error("Create your research problem first:");
    console.error("  cp templates/problem_template.md research/problem.md");
    process.exit(1);
  }

  await $`mkdir -p output`.quiet();

  console.log("Spawning researcher agent (Claude YOLO mode)...\n");

  const proc = Bun.spawn(
    [
      "claude",
      "--dangerously-skip-permissions",
      "-p",
      PHASE1_PROMPT,
      "--output-format",
      "text",
    ],
    {
      cwd: process.cwd(),
      stdout: "inherit",
      stderr: "inherit",
    }
  );

  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    console.error("\nPhase 1 agent failed with exit code:", exitCode);
    return false;
  }

  if (!existsSync("output/confirmed_statement.md")) {
    console.error("\nPhase 1 agent did not produce confirmed_statement.md");
    return false;
  }

  console.log("\n✓ Phase 1 complete: output/confirmed_statement.md created");
  return true;
}

// Run directly if this is the entry point
if (import.meta.main) {
  const success = await runPhase1();
  process.exit(success ? 0 : 1);
}
