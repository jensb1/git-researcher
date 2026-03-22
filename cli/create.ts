#!/usr/bin/env bun
/**
 * create-git-researcher CLI
 *
 * Scaffolds a new AI-driven research project.
 *
 * Usage:
 *   npx create-git-researcher my-research
 *   npx create-git-researcher my-research --name "My Research Topic"
 */

import { mkdirSync, writeFileSync, existsSync } from "fs";
import { join, resolve } from "path";

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const GREEN = "\x1b[32m";
const CYAN = "\x1b[36m";
const DIM = "\x1b[2m";

function log(msg: string) {
  console.log(msg);
}

function success(msg: string) {
  console.log(`${GREEN}✓${RESET} ${msg}`);
}

function header(msg: string) {
  console.log(`\n${BOLD}${CYAN}${msg}${RESET}`);
}

// ─── Templates ────────────────────────────────────────────────────

const CLAUDE_MD = `# Git Researcher — Agent Instructions

This is an AI-driven research system. Claude agents operate in three phases.

## CRITICAL SAFETY RULES

1. **NEVER execute training code.** Do not run Python scripts, install ML frameworks (PyTorch, JAX, etc.), download models, or execute any GPU/CPU training workloads. No evaluation scripts either.
2. **NEVER clone repositories** or install packages. You can web search and read papers — that's your research tool.
3. **60-second bash command limit.** Any individual bash command you run MUST complete within 60 seconds. Prefix anything potentially slow with \`timeout 60\`. If a command hangs, kill it and move on. The ONLY bash commands you should need are: \`mkdir -p\` for creating directories, and file read/write operations. No training, no evaluation, no pip, no python.
4. **Take your time thinking.** There is NO timeout on your overall research session. Think deeply, search thoroughly, design carefully. The timeout only applies to bash commands so you don't get stuck waiting on a runaway process.
5. **Markdown only.** Your outputs are markdown files with analysis, proposed methods, pseudocode, and mathematical formulations. Not running code.
6. **Generalizable methods only.** Every proposed method must work across model sizes, datasets, and domains. No narrow/brittle solutions.

## Project Structure
- \`research/problem.md\` — the user's research problem (input)
- \`research/supporting/\` — supporting documents (input)
- \`output/confirmed_statement.md\` — verified research statement (phase 1 output)
- \`output/ideas/idea_NNN.md\` — individual research ideas (phase 2 output)
- \`output/orchestrator_status.md\` — live progress tracker (phase 2 output)
- \`output/results/result_NNN.md\` — sub-agent results (phase 3 output)
- \`output/final_report.md\` — aggregated final report (final output)

## Agent Roles

### Phase 1: Researcher Agent
- Read \`research/problem.md\` and any files in \`research/supporting/\`
- Verify understanding of the problem
- Produce \`output/confirmed_statement.md\` containing:
  - Restated problem in your own words
  - Key metrics about what is known vs unknown
  - Identified knowledge gaps
  - Suggested research angles

### Phase 2: Orchestrator Agent (idea generation)
- Read \`output/confirmed_statement.md\`
- Generate 3-8 distinct research ideas/angles
- Write each to \`output/ideas/idea_NNN.md\`
- Each idea file must include: title, description, approach, expected metrics, generalizability argument

### Phase 3: Sub-Agent (research execution)
- You will receive a specific research idea and the original problem
- Investigate thoroughly using web search and analysis
- Design a concrete, generalizable method with:
  - Mathematical formulation and update rules
  - Pseudocode
  - Convergence argument
  - Generalizability analysis
- Write results to your assigned \`output/results/result_NNN.md\`

## Conventions
- All output files are markdown
- Use clear headers and structured formatting
- Include metrics as YAML frontmatter where applicable
- Never modify files in \`research/\` — that's user input
- Proposed methods must include pseudocode, not just prose
`;

const PROBLEM_TEMPLATE = `# Research Problem

## Title
<!-- A clear, concise title for your research question -->

## Problem Statement
<!-- Describe the research problem in detail. What are you trying to understand or solve? -->

## Context
<!-- What background information is relevant? What domain does this belong to? -->

## Known Information
<!-- What do you already know about this topic? List existing knowledge, prior research, constraints. -->

## Success Criteria
<!-- How will you know when the research is successful? What would a good answer look like? -->

## Scope

### In Scope
<!-- What should be investigated? -->

### Out of Scope
<!-- What should NOT be investigated? -->

## Supporting Files
<!-- List any supporting files you've added to research/supporting/ -->
`;

const GITIGNORE = `node_modules/
bun.lock
.DS_Store
output/
!output/.gitkeep
data/
`;

function getPackageJson(name: string) {
  return JSON.stringify(
    {
      name,
      version: "0.1.0",
      description: "AI-driven research project powered by git-researcher",
      scripts: {
        start: "bun run src/orchestrator.ts",
        phase1: "bun run src/phase1-researcher.ts",
        phase2: "bun run src/orchestrator.ts --skip-phase1",
      },
      type: "module",
    },
    null,
    2
  );
}

function getTsconfig() {
  return JSON.stringify(
    {
      compilerOptions: {
        target: "ESNext",
        module: "ESNext",
        moduleResolution: "bundler",
        types: ["bun-types"],
        strict: true,
        esModuleInterop: true,
        skipLibCheck: true,
        outDir: "dist",
      },
      include: ["src/**/*.ts"],
    },
    null,
    2
  );
}

// Prompts file (imported from main project pattern)
const PROMPTS_TS = `const SAFETY_PREAMBLE = \`
CRITICAL CONSTRAINTS — READ CAREFULLY:
- NEVER execute training code, install ML frameworks, download models, or run GPU/CPU workloads.
- NEVER run pip install, clone repos, or execute Python/training scripts.
- If you run ANY bash command, it MUST complete within 60 seconds. Prefix long-running commands with "timeout 60".
- You are a THEORETICAL RESEARCHER. Your output is markdown analysis, not running code.
- All proposed methods MUST be generalizable across model sizes and datasets.
- Use web search for literature review. Use file read/write for your markdown outputs. That's it.
\`;

export const PHASE1_PROMPT = \`You are a research analyst. Your job is to thoroughly understand a research problem and produce a verified research statement.
\${SAFETY_PREAMBLE}
INSTRUCTIONS:
1. Read the file research/problem.md carefully
2. Read any files in research/supporting/ for additional context
3. Analyze the problem and identify:
   - Core research question
   - What is already known
   - What is unknown / needs investigation
   - Key metrics that would indicate research success
   - Potential research angles
4. Write your analysis to output/confirmed_statement.md

Make sure output/ directory exists before writing (mkdir -p output).
Do NOT run any training code. Your only output is the markdown file.
\`;

export const ORCHESTRATOR_IDEA_PROMPT = \`You are a research orchestrator. Generate distinct research ideas.
\${SAFETY_PREAMBLE}
INSTRUCTIONS:
1. Read output/confirmed_statement.md
2. Create 3-8 focused research ideas
3. Create output/ideas/ directory (mkdir -p output/ideas)
4. Write each idea to output/ideas/idea_NNN.md with: title, description, approach, expected outcomes, generalizability
5. Create output/orchestrator_status.md tracking all ideas

Do NOT run any training code. Your only output is the markdown files.
\`;

export function getSubAgentPrompt(ideaId: string): string {
  return \`You are a research sub-agent. Investigate one specific research angle.
\${SAFETY_PREAMBLE}
INSTRUCTIONS:
1. Read research/problem.md, output/confirmed_statement.md, and output/ideas/idea_\${ideaId}.md
2. Use web search to find relevant papers and prior art
3. Design your proposed method with mathematical rigor
4. Write results to output/results/result_\${ideaId}.md

Create output/results/ directory if needed (mkdir -p output/results).
Include: summary, proposed method with pseudocode, literature support, generalizability analysis, matching metrics, key takeaways, limitations.

REMEMBER: Design methods, don't run them. Write your output file and stop.
\`;
}

export const FINAL_REPORT_PROMPT = \`You are a research synthesizer. Compile all sub-agent results into a final report.
\${SAFETY_PREAMBLE}
INSTRUCTIONS:
1. Read research/problem.md, output/confirmed_statement.md, all output/results/, and output/orchestrator_status.md
2. Synthesize into output/final_report.md with: research question, executive summary, methodology, proposed methods summary, comparative analysis, synthesis, top recommendations, conclusions, open questions, next steps
3. Update output/orchestrator_status.md to mark Phase 3 as complete.

Do NOT run any training code. Your only output is the markdown files.
\`;
`;

const ORCHESTRATOR_TS = `import { $ } from "bun";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "fs";
import { runPhase1 } from "./phase1-researcher";
import {
  ORCHESTRATOR_IDEA_PROMPT,
  FINAL_REPORT_PROMPT,
  getSubAgentPrompt,
} from "./prompts";

const MAX_PARALLEL_AGENTS = 4;
const skipPhase1 = process.argv.includes("--skip-phase1");

async function spawnClaude(prompt: string, label: string, inherit = true): Promise<number> {
  const proc = Bun.spawn(
    ["claude", "--dangerously-skip-permissions", "-p", prompt, "--output-format", "text"],
    { cwd: process.cwd(), stdout: inherit ? "inherit" : "pipe", stderr: inherit ? "inherit" : "pipe" }
  );
  return await proc.exited;
}

async function runPhase2Ideas(): Promise<boolean> {
  console.log("\\n═══ Phase 2: Generating Research Ideas ═══\\n");
  await $\`mkdir -p output/ideas\`.quiet();
  const exit = await spawnClaude(ORCHESTRATOR_IDEA_PROMPT, "phase2-ideas");
  if (exit !== 0) { console.error("Idea generation failed"); return false; }
  const ideas = readdirSync("output/ideas").filter(f => f.startsWith("idea_"));
  if (ideas.length === 0) { console.error("No ideas generated"); return false; }
  console.log(\`\\n✓ \${ideas.length} ideas generated\`);
  return true;
}

async function runSubAgent(ideaId: string): Promise<boolean> {
  console.log(\`  → Sub-agent for idea \${ideaId}\`);
  const exit = await spawnClaude(getSubAgentPrompt(ideaId), \`sub-agent-\${ideaId}\`, false);
  if (exit !== 0) { console.error(\`  ✗ Sub-agent \${ideaId} failed\`); return false; }
  console.log(\`  ✓ Sub-agent \${ideaId} complete\`);
  return true;
}

async function runPhase3(): Promise<boolean> {
  console.log("\\n═══ Phase 3: Research Sub-Agents ═══\\n");
  await $\`mkdir -p output/results\`.quiet();
  const ideas = readdirSync("output/ideas").filter(f => f.startsWith("idea_") && f.endsWith(".md")).sort();
  if (ideas.length === 0) { console.error("No ideas found"); return false; }
  const ids = ideas.map(f => f.replace("idea_", "").replace(".md", ""));

  for (let i = 0; i < ids.length; i += MAX_PARALLEL_AGENTS) {
    const batch = ids.slice(i, i + MAX_PARALLEL_AGENTS);
    console.log(\`\\nBatch: ideas \${batch.join(", ")}\`);
    await Promise.all(batch.map(id => runSubAgent(id)));
  }
  return true;
}

async function generateFinalReport(): Promise<boolean> {
  console.log("\\n═══ Generating Final Report ═══\\n");
  const exit = await spawnClaude(FINAL_REPORT_PROMPT, "final-report");
  if (exit !== 0) { console.error("Final report failed"); return false; }
  console.log("\\n✓ Final report: output/final_report.md");
  return true;
}

async function main() {
  console.log("╔═══════════════════════════════════╗");
  console.log("║     Git Researcher Pipeline       ║");
  console.log("╚═══════════════════════════════════╝\\n");

  if (!skipPhase1) {
    if (!(await runPhase1())) { console.error("Aborted at Phase 1"); process.exit(1); }
  } else {
    if (!existsSync("output/confirmed_statement.md")) {
      console.error("Run phase1 first"); process.exit(1);
    }
  }

  if (!(await runPhase2Ideas())) { console.error("Aborted at Phase 2"); process.exit(1); }
  const p3 = await runPhase3();
  if (!p3) console.warn("Some sub-agents failed, continuing...");
  if (!(await generateFinalReport())) { console.error("Final report failed"); process.exit(1); }

  console.log("\\n✓ Research complete! See output/final_report.md");
}

main();
`;

const PHASE1_TS = `import { $ } from "bun";
import { existsSync } from "fs";
import { PHASE1_PROMPT } from "./prompts";

export async function runPhase1(): Promise<boolean> {
  console.log("═══ Phase 1: Problem Verification ═══\\n");

  if (!existsSync("research/problem.md")) {
    console.error("Error: research/problem.md not found.");
    console.error("Edit it with your research problem first.");
    process.exit(1);
  }

  await $\`mkdir -p output\`.quiet();
  console.log("Spawning researcher agent...\\n");

  const proc = Bun.spawn(
    ["claude", "--dangerously-skip-permissions", "-p", PHASE1_PROMPT, "--output-format", "text"],
    { cwd: process.cwd(), stdout: "inherit", stderr: "inherit" }
  );

  const exitCode = await proc.exited;
  if (exitCode !== 0) { console.error("Phase 1 failed"); return false; }
  if (!existsSync("output/confirmed_statement.md")) {
    console.error("Phase 1 did not produce confirmed_statement.md");
    return false;
  }

  console.log("\\n✓ Phase 1 complete: output/confirmed_statement.md");
  return true;
}

if (import.meta.main) {
  const ok = await runPhase1();
  process.exit(ok ? 0 : 1);
}
`;

// ─── CLI Logic ────────────────────────────────────────────────────

function scaffold(dir: string, projectName: string) {
  const abs = resolve(dir);

  if (existsSync(abs)) {
    console.error(`Error: ${abs} already exists`);
    process.exit(1);
  }

  header(`Creating research project: ${projectName}`);

  // Directories
  const dirs = [
    "",
    "research",
    "research/supporting",
    "output",
    "src",
    "templates",
    "experiments",
  ];
  for (const d of dirs) {
    mkdirSync(join(abs, d), { recursive: true });
  }
  success("Created directory structure");

  // Files
  const files: [string, string][] = [
    ["CLAUDE.md", CLAUDE_MD],
    ["research/problem.md", PROBLEM_TEMPLATE],
    [".gitignore", GITIGNORE],
    ["package.json", getPackageJson(projectName)],
    ["tsconfig.json", getTsconfig()],
    ["src/prompts.ts", PROMPTS_TS],
    ["src/orchestrator.ts", ORCHESTRATOR_TS],
    ["src/phase1-researcher.ts", PHASE1_TS],
    ["output/.gitkeep", ""],
    ["research/supporting/.gitkeep", ""],
    ["templates/problem_template.md", PROBLEM_TEMPLATE],
  ];

  for (const [path, content] of files) {
    writeFileSync(join(abs, path), content);
  }
  success(`Created ${files.length} files`);

  // Summary
  header("Done! Next steps:");
  log("");
  log(`  ${DIM}cd ${dir}${RESET}`);
  log(`  ${DIM}bun install${RESET}              ${DIM}# install dependencies${RESET}`);
  log(`  ${DIM}# edit research/problem.md${RESET}  ${DIM}# define your research problem${RESET}`);
  log(`  ${DIM}bun run start${RESET}             ${DIM}# run the full research pipeline${RESET}`);
  log("");
  log(`  ${DIM}Or run phases individually:${RESET}`);
  log(`  ${DIM}bun run phase1${RESET}            ${DIM}# problem verification only${RESET}`);
  log(`  ${DIM}bun run phase2${RESET}            ${DIM}# idea generation + sub-agents${RESET}`);
  log("");
}

// ─── Main ─────────────────────────────────────────────────────────

const args = process.argv.slice(2);

if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
  log(`
${BOLD}create-git-researcher${RESET} — scaffold an AI-driven research project

${BOLD}Usage:${RESET}
  npx create-git-researcher <project-name>

${BOLD}Example:${RESET}
  npx create-git-researcher my-bitnet-research
  cd my-bitnet-research
  # edit research/problem.md
  bun run start

${BOLD}Requirements:${RESET}
  - Bun (https://bun.sh)
  - Claude Code CLI (authenticated)
`);
  process.exit(0);
}

const dirName = args[0];
const projectName = dirName.replace(/[^a-zA-Z0-9-_]/g, "-").toLowerCase();

scaffold(dirName, projectName);
