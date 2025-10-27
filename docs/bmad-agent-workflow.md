# BMAD Agent Workflow Setup

Use the following commands for Phase 7 stories (e.g., Batch 4 ingestion):

```bash
# Planning
bmad run --role scrum_master --story us-028-phase7-batch4
# Developer
bmad run --role developer --story us-028-phase7-batch4
# QA
bmad run --role qa --story us-028-phase7-batch4
# Documentation (optional)
bmad run --role documentation --story us-028-phase7-batch4
```

Each role uses the guardrail prompts stored in `bmad/prompts/` and enforces the summary → plan → approval loop.
