import math

import numpy as np

HORIZONS = [1, 3, 5, 15, 60]


def run_scan(df, symbol, tf):
    ret = df["close"].pct_change()
    o = df["open"]; c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]
    vm = v.rolling(200).mean(); vs = v.rolling(200).std()
    vol_z = (v - vm) / vs
    atr = (h - l).rolling(14).mean()
    rng = (h - l) / atr
    up = (c > o).astype(int)
    dn = (c < o).astype(int)
    body_hi = np.maximum(o, c); body_lo = np.minimum(o, c)
    events = {
        "vol_z>3": vol_z > 3,
        "vol_z>5": vol_z > 5,
        "range>2atr": rng > 2,
        "up3": up.rolling(3).sum() == 3,
        "down3": dn.rolling(3).sum() == 3,
        "big_upper_wick": (h - body_hi) > (h - l) * 0.6,
        "big_lower_wick": (body_lo - l) > (h - l) * 0.6,
    }
    out = []
    for name, ev in events.items():
        mask = ev.fillna(False).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) < 50:
            continue
        for hz in HORIZONS:
            fwd = ret.rolling(hz).sum().shift(-hz).to_numpy()
            x = fwd[idx]
            x = x[~np.isnan(x)]
            if len(x) < 50:
                continue
            n = len(x)
            mu = float(np.mean(x))
            sd = float(np.std(x))
            t = float(mu / (sd / math.sqrt(n))) if sd > 0 else 0.0
            out.append({"event": name, "horizon": int(hz), "n": int(n),
                        "mean_fwd": mu, "std": sd, "t": t,
                        "pos_share": float((x > 0).mean())})
    return {"symbol": symbol, "tf": tf, "findings": out}
