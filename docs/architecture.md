# SenseQuant — Architecture (v1)

## Modules
- **Config**: loads .env (Breeze keys, symbols, risk, mode=dryrun/live)
- **BreezeClient (Adapter)**: wraps breeze_connect (auth, LTP, quotes, historical, place_order)
- **Strategies**:
  - IntradayStrategy: minute bars, momentum/mean-reversion baseline + sentiment gating
  - SwingStrategy: daily bars, trend/structure baseline + sentiment gating
- **Sentiment**: pluggable provider; baseline uses simple NLP; later: news APIs/HF models
- **Engine**: scheduler + data loop; merges strategy outputs; risk checks; dispatches orders
- **Logging/Audit**: structured logs, trade journal csv
- **Backtest**: offline runner over historical bars

## Data Flow
Breeze (live/hist) → Feature calc (TA) + Sentiment → Strategies → Signal bus → Risk/Position sizing → Execution (Breeze) → Logs/Journal → Teacher review (EOD) → Param update.

## Teacher–Student (v1 stub)
- Teacher: EOD reviewer adjusts student params (thresholds, sizing) with simple rules
- Student: strategies consuming those params; later upgrade to learned models

## Files
- src/config.py, src/breeze_client.py, src/sentiment.py, src/strategies/{intraday,swing}.py
- src/engine.py (main loop), src/main.py (CLI entry), tests/*

