import { $ } from "bun";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "fs";
import { runPhase1 } from "./phase1-researcher";
import {
  ORCHESTRATOR_IDEA_PROMPT,
  FINAL_REPORT_PROMPT,
  getSubAgentPrompt,
} from "./prompts";

const MAX_PARALLEL_AGENTS = 4;
const skipPhase1 = process.argv.includes("--skip-phase1");

// ── Phase 2: Generate Research Ideas ──────────────────────────────

async function runPhase2Ideas(): Promise<boolean> {
  console.log("\n═══════════════════════════════════════");
  console.log("  Phase 2: Generating Research Ideas");
  console.log("═══════════════════════════════════════\n");

  await $`mkdir -p output/ideas`.quiet();

  console.log("Spawning orchestrator idea-generation agent...\n");

  const proc = Bun.spawn(
    [
      "claude",
      "--dangerously-skip-permissions",
      "-p",
      ORCHESTRATOR_IDEA_PROMPT,
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
    console.error("\nIdea generation agent failed with exit code:", exitCode);
    return false;
  }

  const ideaFiles = readdirSync("output/ideas").filter((f) =>
    f.startsWith("idea_")
  );
  if (ideaFiles.length === 0) {
    console.error("\nNo idea files were generated.");
    return false;
  }

  console.log(`\n✓ Phase 2 (ideas): ${ideaFiles.length} ideas generated`);
  return true;
}

// ── Phase 3: Dispatch Sub-Agents ──────────────────────────────────

async function runSubAgent(ideaId: string): Promise<boolean> {
  console.log(`  → Starting sub-agent for idea ${ideaId}`);

  const prompt = getSubAgentPrompt(ideaId);

  const proc = Bun.spawn(
    [
      "claude",
      "--dangerously-skip-permissions",
      "-p",
      prompt,
      "--output-format",
      "text",
    ],
    {
      cwd: process.cwd(),
      stdout: "pipe",
      stderr: "pipe",
    }
  );

  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    console.error(`  ✗ Sub-agent for idea ${ideaId} failed`);
    return false;
  }

  console.log(`  ✓ Sub-agent for idea ${ideaId} complete`);
  return true;
}

function updateOrchestratorStatus(ideaId: string, status: string) {
  const statusFile = "output/orchestrator_status.md";
  if (!existsSync(statusFile)) return;

  let content = readFileSync(statusFile, "utf-8");
  // Update the status for this idea in the table
  const pattern = new RegExp(`(\\| ${ideaId} \\|[^|]+\\|)\\s*\\w+\\s*(\\|)`);
  content = content.replace(pattern, `$1 ${status} $2`);
  writeFileSync(statusFile, content);
}

async function runPhase3Research(): Promise<boolean> {
  console.log("\n═══════════════════════════════════════");
  console.log("  Phase 3: Dispatching Research Agents");
  console.log("═══════════════════════════════════════\n");

  await $`mkdir -p output/results`.quiet();

  const ideaFiles = readdirSync("output/ideas")
    .filter((f) => f.startsWith("idea_") && f.endsWith(".md"))
    .sort();

  if (ideaFiles.length === 0) {
    console.error("No idea files found in output/ideas/");
    return false;
  }

  console.log(`Found ${ideaFiles.length} research ideas to investigate\n`);

  // Extract idea IDs (e.g., "001" from "idea_001.md")
  const ideaIds = ideaFiles.map((f) => f.replace("idea_", "").replace(".md", ""));

  // Process in batches of MAX_PARALLEL_AGENTS
  let allSucceeded = true;
  for (let i = 0; i < ideaIds.length; i += MAX_PARALLEL_AGENTS) {
    const batch = ideaIds.slice(i, i + MAX_PARALLEL_AGENTS);
    console.log(
      `\nBatch ${Math.floor(i / MAX_PARALLEL_AGENTS) + 1}: ideas ${batch.join(", ")}`
    );

    // Update status to "in_progress"
    for (const id of batch) {
      updateOrchestratorStatus(id, "in_progress");
    }

    // Run batch in parallel
    const results = await Promise.all(batch.map((id) => runSubAgent(id)));

    // Update statuses
    for (let j = 0; j < batch.length; j++) {
      updateOrchestratorStatus(
        batch[j],
        results[j] ? "complete" : "failed"
      );
      if (!results[j]) allSucceeded = false;
    }
  }

  return allSucceeded;
}

// ── Final Report ──────────────────────────────────────────────────

async function generateFinalReport(): Promise<boolean> {
  console.log("\n═══════════════════════════════════════");
  console.log("  Generating Final Report");
  console.log("═══════════════════════════════════════\n");

  const proc = Bun.spawn(
    [
      "claude",
      "--dangerously-skip-permissions",
      "-p",
      FINAL_REPORT_PROMPT,
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
    console.error("\nFinal report generation failed");
    return false;
  }

  console.log("\n✓ Final report: output/final_report.md");
  return true;
}

// ── Main Pipeline ─────────────────────────────────────────────────

async function main() {
  console.log("╔═══════════════════════════════════════╗");
  console.log("║       Git Researcher Pipeline         ║");
  console.log("╚═══════════════════════════════════════╝\n");

  // Phase 1
  if (!skipPhase1) {
    const phase1Ok = await runPhase1();
    if (!phase1Ok) {
      console.error("\nPipeline aborted at Phase 1");
      process.exit(1);
    }
  } else {
    console.log("Skipping Phase 1 (--skip-phase1 flag)\n");
    if (!existsSync("output/confirmed_statement.md")) {
      console.error("Error: output/confirmed_statement.md not found. Run phase1 first.");
      process.exit(1);
    }
  }

  // Phase 2: Generate ideas
  const phase2Ok = await runPhase2Ideas();
  if (!phase2Ok) {
    console.error("\nPipeline aborted at Phase 2 (idea generation)");
    process.exit(1);
  }

  // Phase 3: Dispatch sub-agents
  const phase3Ok = await runPhase3Research();
  if (!phase3Ok) {
    console.warn("\nSome sub-agents failed, but continuing to final report...");
  }

  // Final report
  const reportOk = await generateFinalReport();
  if (!reportOk) {
    console.error("\nFinal report generation failed");
    process.exit(1);
  }

  console.log("\n╔═══════════════════════════════════════╗");
  console.log("║     Research Pipeline Complete! ✓      ║");
  console.log("╚═══════════════════════════════════════╝");
  console.log("\nResults:");
  console.log("  → output/confirmed_statement.md");
  console.log("  → output/ideas/");
  console.log("  → output/results/");
  console.log("  → output/final_report.md");
}

main();
