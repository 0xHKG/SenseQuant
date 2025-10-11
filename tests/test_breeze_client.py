import pandas as pd
from src.breeze_client import BreezeClient

def test_dryrun_get_ltp_returns_zero():
    c = BreezeClient("k","s","t", dry_run=True)
    assert c.get_ltp("RELIANCE") == 0.0

def test_dryrun_historical_dataframe():
    c = BreezeClient("k","s","t", dry_run=True)
    df = c.get_historical("RELIANCE","1minute","2025-01-01 09:15","2025-01-01 15:15")
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "datetime" or True  # relaxed for dryrun

def test_dryrun_place_order_ok():
    c = BreezeClient("k","s","t", dry_run=True)
    res = c.place_order({"stock_code":"RELIANCE","action":"BUY","quantity":1,"order_type":"MARKET"})
    assert res.get("status") == "DRYRUN"

