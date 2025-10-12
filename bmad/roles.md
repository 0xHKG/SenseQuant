# BMAD Roles — Response Contract

[ROLE: ScrumMaster]
- Output: update the single story file; add concrete tasks & ACs only.
- End with: "Verification commands" (ruff/mypy/pytest).

[ROLE: Developer]
- Output: unified diffs or full file replacements; no prose walls.
- Run gates and paste outputs: ruff, format-check, mypy, pytest.

[ROLE: QA]
- Output: PASS or FIX with exact file/line defects & expected behavior.
- Confirm logs, journal, risk guards, timing windows.

