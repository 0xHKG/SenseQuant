# SenseQuant — Product Requirements (v1)

## Purpose
AI assistant for Indian equities (NSE/BSE) supporting **intraday** and **swing** trading, using:
- Teacher–Student learning loop (progressive walk-forward refinement)
- Technical signals + **sentiment analysis**
- **ICICI Direct Breeze API** for data & order execution

## Core Requirements

### 1. Trading Strategies
**Intraday**
- AC: System opens positions after 9:15 IST and force-closes all by 15:29 IST
- AC: No intraday positions held overnight

**Swing**
- AC: Positions can be held 2–10 days with automatic SL/TP enforcement
- AC: Position sizing respects global exposure limits

### 2. Teacher–Student Learning Loop
- AC: "Teacher" (advanced model) generates labels on historical windows
- AC: "Student" (lightweight model) trains on teacher labels, runs real-time inference
- AC: Walk-forward: retrain student on expanding window every N days

### 3. Sentiment Analysis
- AC: Ingest news/social feeds for selected stocks
- AC: Compute sentiment score (−1 to +1) per stock
- AC: Signals are gated (blocked if sentiment < threshold) or boosted (scaled if sentiment > threshold)

### 4. Breeze API Integration
- AC: Fetch historical OHLCV (1min, 5min, daily) with retry on failure
- AC: Subscribe to live market data; reconnect on disconnect
- AC: Place market/limit orders; log all responses; retry once on transient errors
- AC: Respect rate limits (back-off on 429)

### 5. Risk Management
- AC: Global max exposure cap (₹/notional)
- AC: Per-trade stop-loss and take-profit enforced
- AC: Daily loss cap triggers circuit-breaker (stop new trades)
- AC: Aware of exchange circuit-breaker halts

### 6. Operations
- AC: All credentials/config in .env (not hardcoded)
- AC: Structured JSON logs (timestamp, level, component, message)
- AC: Dry-run mode simulates orders without API calls
- AC: Backtest mode runs strategy on historical data, outputs P&L

## Non-Functionals
- AC: Runs on ≤1 vCPU, <300MB RAM under steady load
- AC: Graceful shutdown (close positions, flush logs)
- AC: Modular code: separate modules for data, strategy, risk, execution
- AC: ≥70% unit test coverage for adapters and core logic
- AC: Docstrings for all public functions

## Success Criteria (v1)
- AC: Live mode receives ticks → generates signals → places orders (or dry-run logs)
- AC: Backtest utility loads historical bars → runs strategy → outputs metrics (Sharpe, max DD, win rate)
- AC: No unhandled exceptions in 24h operation under normal market conditions

