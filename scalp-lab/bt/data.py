import os
import zipfile

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_vol", "trades", "taker_base", "taker_quote", "ignore"]
KEEP = ["dt", "open", "high", "low", "close", "volume", "quote_vol",
        "trades", "taker_base", "taker_quote"]

_MEMO = {}


def _fetch(url, path, timeout=60):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except requests.exceptions.RequestException:
        return False


def _read_zip(path):
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None, names=COLS)
    # newer Binance archives include a header row - drop non-numeric rows
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
    return df


def _klines_raw(symbol, tf, days, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days)
    frames = []
    for m in pd.period_range(start, end, freq="M"):
        fn = symbol + "-" + tf + "-" + str(m) + ".zip"
        p = os.path.join(cache_dir, fn)
        url = BASE + "/monthly/klines/" + symbol + "/" + tf + "/" + fn
        if not os.path.exists(p):
            if not _fetch(url, p):
                continue
        frames.append(_read_zip(p))
    cur_start = end.to_period("M").start_time.tz_localize("UTC")
    for d in pd.date_range(cur_start, end, freq="D"):
        fn = symbol + "-" + tf + "-" + d.strftime("%Y-%m-%d") + ".zip"
        p = os.path.join(cache_dir, fn)
        url = BASE + "/daily/klines/" + symbol + "/" + tf + "/" + fn
        if not os.path.exists(p):
            if not _fetch(url, p):
                continue
        frames.append(_read_zip(p))
    if not frames:
        raise RuntimeError("no data for " + symbol + " " + tf)
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df = df[df["dt"] >= start].reset_index(drop=True)
    df = df[KEEP]
    for c in KEEP[1:]:
        df[c] = df[c].astype(float)
    return df


def klines(symbol, tf, days, cache_dir=".cache"):
    key = (symbol, tf, days)
    if key in _MEMO:
        return _MEMO[key]
    try:
        df = _klines_raw(symbol, tf, days, cache_dir)
    except RuntimeError:
        # some futures symbols are quoted as 1000<SYM>USDT (e.g. PEPE)
        if not symbol.endswith("USDT") or symbol.startswith("1000"):
            raise
        alt = "1000" + symbol
        df = _klines_raw(alt, tf, days, cache_dir)
    _MEMO[key] = df
    return df


# ===== деривативные данные: funding rate, открытый интерес, long/short =====

FUND_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
METRICS_COLS = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
                "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
_DERIV_MEMO = {}


def _read_zip_named(path, cols):
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=cols, dtype=str)
    tcol = df[cols[0]]
    ts = pd.to_numeric(tcol, errors="coerce")
    if ts.isna().all():
        # некоторые архивы хранят время строкой - парсим в ms
        ts = pd.to_datetime(tcol, utc=True, errors="coerce").astype("int64") // 1000000
    df[cols[0]] = ts
    df = df[df[cols[0]] > 0]
    return df


def _deriv_raw(kind, symbol, days, cache_dir, timeout=20, max_req=40):
    """fundingRate / metrics: monthly + daily архивы data.binance.vision (futures um).
    Жёсткие таймауты и лимит запросов: медленный/битый источник не может
    повесить воркер или бумажную торговлю."""
    cols = FUND_COLS if kind == "fundingRate" else METRICS_COLS
    os.makedirs(cache_dir, exist_ok=True)
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days)
    frames = []
    reqs = [0]

    def load(fn, sub):
        p = os.path.join(cache_dir, fn)
        if not os.path.exists(p):
            reqs[0] += 1
            if reqs[0] > max_req:
                raise RuntimeError("deriv request cap " + symbol)
            if not _fetch(BASE + "/" + sub + "/" + symbol + "/" + fn, p, timeout):
                return None
        try:
            return _read_zip_named(p, cols)
        except Exception:
            return None

    for m in pd.period_range(start, end, freq="M"):
        df = load(symbol + "-" + kind + "-" + str(m) + ".zip", "monthly/" + kind)
        if df is not None:
            frames.append(df)
    cur_start = end.to_period("M").start_time.tz_localize("UTC")
    for day in pd.date_range(cur_start, end, freq="D"):
        df = load(symbol + "-" + kind + "-" + day.strftime("%Y-%m-%d") + ".zip", "daily/" + kind)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError("no " + kind + " data for " + symbol)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(cols[0]).sort_values(cols[0]).reset_index(drop=True)
    return df


def funding(symbol, days, cache_dir=".cache", timeout=20):
    """Исторический funding rate: DataFrame [dt, funding]."""
    key = ("fundingRate", symbol, days)
    if key in _DERIV_MEMO:
        return _DERIV_MEMO[key]
    try:
        raw = _deriv_raw("fundingRate", symbol, days, cache_dir, timeout)
    except RuntimeError:
        if not symbol.endswith("USDT") or symbol.startswith("1000"):
            raise
        raw = _deriv_raw("fundingRate", "1000" + symbol, days, cache_dir, timeout)
    fr = pd.DataFrame({
        "dt": pd.to_datetime(raw["calc_time"].astype("int64"), unit="ms", utc=True),
        "funding": pd.to_numeric(raw["last_funding_rate"], errors="coerce")})
    fr = fr.dropna().sort_values("dt").reset_index(drop=True)
    _DERIV_MEMO[key] = fr
    return fr


def metrics(symbol, days, cache_dir=".cache", timeout=20):
    """Метрики позиционирования: DataFrame [dt, oi, ls_ratio] (daily metrics)."""
    key = ("metrics", symbol, days)
    if key in _DERIV_MEMO:
        return _DERIV_MEMO[key]
    try:
        raw = _deriv_raw("metrics", symbol, days, cache_dir, timeout)
    except RuntimeError:
        if not symbol.endswith("USDT") or symbol.startswith("1000"):
            raise
        raw = _deriv_raw("metrics", "1000" + symbol, days, cache_dir, timeout)
    mt = pd.DataFrame({
        "dt": pd.to_datetime(raw["create_time"].astype("int64"), unit="ms", utc=True),
        "oi": pd.to_numeric(raw["sum_open_interest"], errors="coerce"),
        "ls_ratio": pd.to_numeric(raw["sum_taker_long_short_vol_ratio"], errors="coerce")})
    mt = mt.dropna().sort_values("dt").reset_index(drop=True)
    _DERIV_MEMO[key] = mt
    return mt


def add_deriv(df, symbol, days, cache_dir=".cache", timeout=20):
    """Присоединяет funding/OI/long-short к свечам (asof, без утечки будущего).
    Отсутствующая сторона данных не роняет задачу: колонки заполняются нулями."""
    df = df.copy().reset_index(drop=True)
    d_ts = df["dt"].astype("int64").values
    sides = []
    for mk in (funding, metrics):
        try:
            sides.append(mk(symbol, days, cache_dir, timeout))
        except Exception as e:
            print("deriv enrich skip:", mk.__name__, str(e)[:80], flush=True)
    for side in sides:
        s_ts = side["dt"].astype("int64").values
        pos = np.searchsorted(s_ts, d_ts, side="right") - 1
        for c in side.columns:
            if c == "dt":
                continue
            v = side[c].values
            df[c] = np.where(pos >= 0, v[np.clip(pos, 0, len(v) - 1)], 0.0)
    for c in ("funding", "oi", "ls_ratio"):
        if c not in df.columns:
            df[c] = 0.0
    return df
