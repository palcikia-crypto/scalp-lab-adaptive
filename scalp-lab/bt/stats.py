import math

import numpy as np


def metrics(r, bars_per_year):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    out = {"n": int(n), "sharpe": 0.0, "sortino": 0.0, "maxdd": 0.0,
           "winrate": 0.0, "pf": 0.0, "t": 0.0, "total": 0.0}
    if n < 100:
        return out
    sd = float(r.std())
    if sd == 0:
        return out
    mu = float(r.mean())
    out["sharpe"] = float(mu / sd * math.sqrt(bars_per_year))
    neg = r[r < 0]
    dsd = float(neg.std()) if len(neg) > 5 else sd
    if dsd > 0:
        out["sortino"] = float(mu / dsd * math.sqrt(bars_per_year))
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    out["maxdd"] = float((eq - peak).min())
    out["winrate"] = float((r > 0).mean())
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    out["pf"] = float(gains / losses) if losses > 0 else 99.0
    out["t"] = float(mu / (sd / math.sqrt(n)))
    out["total"] = float(r.sum())
    return out
