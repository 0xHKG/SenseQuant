# SenseQuant — Product Requirements (v1)

## Purpose
AI assistant for Indian equities (NSE/BSE) supporting **intraday** and **swing** trading, using:
- Teacher–Student learning loop (progressive walk-forward refinement)
- Technical signals + **sentiment analysis**
- **ICICI Direct Breeze API** for data & order execution

## Core Requirements
- Intraday strategy: open/close positions within session; force square-off before 15:29 IST.
- Swing strategy: multi-day holds with risk controls (position sizing, SL/TP).
- Sentiment: ingest news/social, compute stock-level sentiment, gate/boost signals.
- Data: fetch historical + live via Breeze; resilient reconnection; rate-limit aware.
- Orders: market/limit support, error handling, audit logs, idempotent retries.
- Risk: global max exposure, per-trade SL, daily loss cap; circuit-breaker awareness.
- Ops: .env config, structured logs, dry-run mode, backtest hooks.

## Non-Functionals
- Runs on free/low-cost host; < 300MB RAM steady; graceful restarts.
- Modular code; unit tests for adapters (API, strategies); docstrings.

## Success (v1)
- End-to-end: receive live ticks/bars → produce signals → place orders (or dry-run)
- Backtest utility runs on downloaded historical bars

