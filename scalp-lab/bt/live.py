"""Live klines с открытого API Binance (без ключа): data-api.binance.vision.

Спотовые свечи в реальном времени, задержка ~1 бар от текущей минуты.
При недоступности API бумажный трейдинг откатывается на дневные архивы (bt.data).
"""
import time

import pandas as pd
import requests

LIVE_URL = "https://data-api.binance.vision/api/v3/klines"


def klines_live(symbol, tf, days, max_bars=20000):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400000
    rows = []
    cur_end = end_ms
    while cur_end > start_ms and len(rows) < max_bars:
        r = requests.get(LIVE_URL, params={
            "symbol": symbol, "interval": tf, "limit": 1000,
            "endTime": cur_end}, timeout=10)
        r.raise_for_status()
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        rows = batch + rows
        cur_end = batch[0][0] - 1
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "iv"]
    df = pd.DataFrame(rows, columns=cols)
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "quote_vol"]:
        df[c] = pd.to_numeric(df[c])
    df = df[df["open_time"] >= start_ms].reset_index(drop=True)
    # последний бар ещё формируется - работаем только с закрытыми барами
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)
    return df
