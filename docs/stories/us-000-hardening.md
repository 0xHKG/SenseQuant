# US-000 — Project Hardening & Structure

## Goal
Refactor scaffold into production-grade structure with typing, linting, tests, and clean boundaries.

## Context
- Keep current features (dry-run, Breeze adapter, basic strategies), but improve quality + stability.
- Use layered architecture: adapters, domain (strategies), services (engine), app (main).
- Enforce tooling: ruff, mypy, pytest; add pre-commit optional.

## Tasks
1) Restructure folders:
   - src/config.py → src/app/config.py
   - src/breeze_client.py → src/adapters/breeze_client.py
   - src/strategies/* → src/domain/strategies/*
   - src/engine.py → src/services/engine.py
   - src/main.py → src/app/main.py
2) Add typing everywhere; enable mypy strict on src/.
3) Add ruff config (pep8/flake) and fix all lint.
4) Strengthen BreezeClient:
   - clear dataclasses/DTOs for requests/responses
   - robust error handling, retries, timeouts, rate-limit backoff
   - deterministic dry-run behavior, structured logs
5) Add pytest for adapter & strategies (mocks for Breeze)
6) Add Makefile/commands: `make lint`, `make test`, `make run`.
7) Update docs/architecture.md with final module map.

## Acceptance
- AC1: `ruff .` and `mypy src/` pass (no errors)
- AC2: `pytest -q` passes (≥ 80% coverage on adapters/strategies)
- AC3: Breeze adapter gracefully handles error scenarios (network, bad auth, bad symbol)
- AC4: Dry-run produces structured logs for all would-be orders
- AC5: docs/architecture.md matches folder layout

