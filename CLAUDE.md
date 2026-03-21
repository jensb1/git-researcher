# Git Researcher — Agent Instructions

This is an AI-driven research system. Claude agents operate in three phases.

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
- Each idea file must include: title, description, approach, expected metrics

### Phase 3: Sub-Agent (research execution)
- You will receive a specific research idea and the original problem
- Investigate thoroughly using web search and analysis
- Calculate matching metrics (relevance, confidence, completeness)
- Write results to your assigned `output/results/result_NNN.md`

## Conventions
- All output files are markdown
- Use clear headers and structured formatting
- Include metrics as YAML frontmatter where applicable
- Never modify files in `research/` — that's user input
