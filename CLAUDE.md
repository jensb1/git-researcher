# Git Researcher — Agent Instructions

This is an AI-driven research system. Claude agents operate in three phases.

## CRITICAL SAFETY RULES

1. **NEVER execute training code.** Do not run Python scripts, install ML frameworks (PyTorch, JAX, etc.), download models, or execute any GPU/CPU training workloads. No evaluation scripts either.
2. **NEVER clone repositories** or install packages. You can web search and read papers — that's your research tool.
3. **60-second bash command limit.** Any individual bash command you run MUST complete within 60 seconds. Prefix anything potentially slow with `timeout 60`. If a command hangs, kill it and move on. The ONLY bash commands you should need are: `mkdir -p` for creating directories, and file read/write operations. No training, no evaluation, no pip, no python.
4. **Take your time thinking.** There is NO timeout on your overall research session. Think deeply, search thoroughly, design carefully. The timeout only applies to bash commands so you don't get stuck waiting on a runaway process.
5. **Markdown only.** Your outputs are markdown files with analysis, proposed methods, pseudocode, and mathematical formulations. Not running code.
6. **Generalizable methods only.** Every proposed training method must work across model sizes (100M to 100B+), datasets, and domains. No narrow/brittle solutions.
7. **Memory constraint: ≤ 4 bytes extra per parameter.** All proposed methods must prove they fit within this budget. Show the math.

## Project Structure
- `research/problem.md` — the user's research problem (input)
- `research/supporting/` — supporting documents (input)
- `output/confirmed_statement.md` — verified research statement (phase 1 output)
- `output/ideas/idea_NNN.md` — individual research ideas (phase 2 output)
- `output/orchestrator_status.md` — live progress tracker (phase 2 output)
- `output/results/result_NNN.md` — sub-agent results (phase 3 output)
- `output/final_report.md` — aggregated final report (final output)

## Agent Roles

### Phase 1: Researcher Agent
- Read `research/problem.md` and any files in `research/supporting/`
- Verify understanding of the problem
- Produce `output/confirmed_statement.md` containing:
  - Restated problem in your own words
  - Key metrics about what is known vs unknown
  - Identified knowledge gaps
  - Suggested research angles

### Phase 2: Orchestrator Agent (idea generation)
- Read `output/confirmed_statement.md`
- Generate 3-8 distinct research ideas/angles
- Write each to `output/ideas/idea_NNN.md`
- Each idea file must include: title, description, approach, expected metrics, generalizability argument

### Phase 3: Sub-Agent (research execution)
- You will receive a specific research idea and the original problem
- Investigate thoroughly using web search and analysis
- Design a concrete, generalizable training method with:
  - Mathematical formulation and update rules
  - Pseudocode for the training loop
  - Memory budget proof (≤ 4 bytes extra/param)
  - Convergence argument
  - Generalizability analysis
- Write results to your assigned `output/results/result_NNN.md`

## Conventions
- All output files are markdown
- Use clear headers and structured formatting
- Include metrics as YAML frontmatter where applicable
- Never modify files in `research/` — that's user input
- Proposed methods must include pseudocode, not just prose
- Always show memory budget arithmetic explicitly
