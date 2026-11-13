"""BRAIN: статистический мозг рынка (детерминированный, без API LLM). v2

Что делает:
- по живым свечам определяет режим рынка для каждого (symbol, tf):
  trend / range / spike / chop / calm + уровень волатильности;
- каждой стратегии задаёт семейство ЯВНО (по сути, не по имени):
  meanrev (возврат к среднему) или trend (импульс/пробой);
  неизвестным - эвристический fallback по имени;
- по семейству и режиму выдаёт множитель веса [0.4 .. 1.3];
- самообучение: история режимов (results/brain_hist.json), семейные
  множители подкручиваются по реальному PnL paper-сделок. Порог входа
  3 сделки, уверенность набирается плавно (полная - к 12 сделкам).

Результат: results/brain.json (для панели и отчёта) + dict fit для paper.py.
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd

STATE_HIST = "results/brain_hist.json"
STATE_BRAIN = "results/brain.json"

# === ЯВНЫЕ семейства: meanrev = возврат к среднему, trend = импульс/пробой ===
FAMILY = {
    # возвратные: торговля против выноса, к среднему
    "bollinger_reversion": "meanrev", "vwap_reversion": "meanrev",
    "streak_reversion": "meanrev", "wick_reversion": "meanrev",
    "bb_maker": "meanrev", "streak_vol": "meanrev", "bb_vol": "meanrev",
    "wick_vol": "meanrev", "spike_revert": "meanrev", "crash_revert": "meanrev",
    "spike_bb": "meanrev", "vol_burst_revert": "meanrev", "bb_regime": "meanrev",
    "wick_regime": "meanrev", "bb_fast": "meanrev", "ema_snap": "meanrev",
    "flow_fade": "meanrev", "qr_revert": "meanrev", "level_fade": "meanrev",
    "streak_bb": "meanrev", "funding_revert": "meanrev", "oi_exhaust": "meanrev",
    "ls_contrarian": "meanrev",
    # импульсные: торговля в сторону выноса, на пробоях
    "ema_cross": "trend", "donchian": "trend", "donchian_trend": "trend",
    "volume_surge": "trend", "vol_drift": "trend", "spike_follow": "trend",
    "squeeze_pop": "trend", "vol_pop": "trend",
}

# эвристический fallback для новых/неизвестных имён
MEAN_REV_HINTS = ("bb", "wick", "revert", "regime", "maker", "vwap", "range", "fade")
TREND_HINTS = ("streak", "burst", "wave", "momentum", "break", "trend", "momo", "cross", "surge", "pop")


def family(name):
    n = (name or "").lower()
    if n in FAMILY:
        return FAMILY[n]
    if any(s in n for s in MEAN_REV_HINTS):
        return "meanrev"
    if any(s in n for s in TREND_HINTS):
        return "trend"
    return "other"


def classify(df):
    """Режим по свечам: трендовость, волатильность, шипы.

    Быстрая реакция: трендовость считается на коротком окне (40 баров)
    и смешивается с длинным (120) как 0.6/0.4 - смена режима ловится
    за 1-2 бара, но шум не дёргает веса. Шипы - окно 30 баров.
    """
    c = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(c) < 60:
        return {"regime": "unknown", "vol": 0.5, "trend_score": 0.0, "spike_score": 0.0}
    r = np.log(c).diff().dropna().values

    def autocorr(arr):
        if len(arr) < 10:
            return 0.0
        a = arr[:-1] - arr[:-1].mean()
        b = arr[1:] - arr[1:].mean()
        denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
        return float((a * b).sum() / denom) if denom > 0 else 0.0

    ac_short = autocorr(r[-40:])
    ac_long = autocorr(r[-120:]) if len(r) >= 120 else ac_short
    ac = 0.6 * ac_short + 0.4 * ac_long
    # волатильность: стд за последние 15 баров против всей истории
    vol = float(np.std(r[-15:]) / (np.std(r) + 1e-12))
    # шипы: доля баров с |r| > 3 стд за последние 30 баров
    sd = np.std(r) + 1e-12
    spike = float(np.mean(np.abs(r[-30:]) > 3.0 * sd))
    trend_score = round(max(-1.0, min(1.0, ac * 4.0)), 3)
    if spike > 0.08:
        regime = "spike"
    elif trend_score > 0.25:
        regime = "trend"
    elif trend_score < -0.15:
        regime = "range"
    elif vol > 1.4:
        regime = "chop"
    else:
        regime = "calm"
    return {"regime": regime, "vol": round(vol, 2),
            "trend_score": trend_score, "spike_score": round(spike, 3)}


def load_hist():
    if os.path.exists(STATE_HIST):
        try:
            return json.load(open(STATE_HIST))
        except Exception:
            pass
    return {"family_regime": {}, "snapshots": []}


def learned_mult(hist, fam, regime):
    """Множитель из собственного опыта: PnL семейства в этом режиме.

    Порог входа 3 сделки. Уверенность растёт плавно: при 3 сделках опыт
    влияет на 25%, при 12+ - полностью (масштаб min(1.0, n/12)).
    """
    rec = hist.get("family_regime", {}).get(fam + ":" + regime)
    if not rec:
        return 1.0
    n = rec.get("n", 0)
    pnl = rec.get("pnl_sum", 0.0)
    if n < 3:
        return 1.0
    avg = pnl / n
    conf = min(1.0, n / 12.0)
    m = 1.0 + max(-0.3, min(0.2, avg * 8.0)) * conf
    return round(m, 3)


BASE_FIT = {
    # (семейство, режим): множитель по логике стратегии
    ("meanrev", "range"): 1.25, ("meanrev", "calm"): 1.1,
    ("meanrev", "trend"): 0.6, ("meanrev", "spike"): 0.8,
    ("meanrev", "chop"): 0.9, ("meanrev", "unknown"): 1.0,
    ("trend", "trend"): 1.3, ("trend", "spike"): 1.0,
    ("trend", "range"): 0.5, ("trend", "calm"): 0.9,
    ("trend", "chop"): 0.8, ("trend", "unknown"): 1.0,
    ("other", "range"): 0.95, ("other", "calm"): 1.05,
    ("other", "trend"): 0.95, ("other", "spike"): 0.85,
    ("other", "chop"): 0.9, ("other", "unknown"): 1.0,
}


def regime_fit(tops, klines_fn, paper_state=None):
    """Главная функция: dict key -> множитель веса + results/brain.json."""
    hist = load_hist()
    regimes = {}
    fit = {}
    cache = {}
    for t in tops:
        sk = t["symbol"] + "|" + t["tf"]
        if sk not in cache:
            try:
                df = klines_fn(t["symbol"], t["tf"])
                cache[sk] = classify(df)
            except Exception:
                cache[sk] = {"regime": "unknown", "vol": 0.5,
                             "trend_score": 0.0, "spike_score": 0.0}
        reg = cache[sk]
        fam = family(t["strategy"])
        base = BASE_FIT.get((fam, reg["regime"]), 1.0)
        learn = learned_mult(hist, fam, reg["regime"])
        m = round(max(0.4, min(1.3, base * learn)), 3)
        fit[t["key"]] = m
        regimes.setdefault(reg["regime"], []).append(t["key"])
    # самообучение: обновляем опыт по закрытым paper-сделкам
    if paper_state is not None:
        fr = hist.setdefault("family_regime", {})
        # режим на момент сделки неизвестен точно - используем текущий режим
        # пары (symbol|tf) из кэша; это честная аппроксимация для недавних сделок
        for x in (paper_state.get("closed") or [])[-60:]:
            sk = x.get("symbol", "") + "|" + x.get("tf", "")
            reg = cache.get(sk)
            if not reg:
                continue
            k = family(x.get("strategy", "")) + ":" + reg["regime"]
            rec = fr.setdefault(k, {"n": 0, "pnl_sum": 0.0})
            # экспоненциальное затухание: свежий опыт важнее
            rec["n"] = rec.get("n", 0) * 0.95 + 1.0
            rec["pnl_sum"] = rec.get("pnl_sum", 0.0) * 0.95 + float(x.get("pnl_pct", 0.0))
        snap = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "regimes": {k: len(v) for k, v in regimes.items()},
                "equity": (paper_state or {}).get("equity")}
        hist["snapshots"] = (hist.get("snapshots", []) + [snap])[-200:]
        os.makedirs("results", exist_ok=True)
        try:
            json.dump(hist, open(STATE_HIST, "w"), ensure_ascii=False, indent=1)
        except Exception:
            pass
    out = {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "regimes": {k: v for k, v in regimes.items()},
           "regime_detail": {sk.split("|")[0] + "-" + sk.split("|")[1]: v
                             for sk, v in cache.items()},
           "fit": fit,
           "learned": {k: round(v.get("pnl_sum", 0) / max(1, v.get("n", 1)), 3)
                       for k, v in hist.get("family_regime", {}).items()}}
    os.makedirs("results", exist_ok=True)
    try:
        json.dump(out, open(STATE_BRAIN, "w"), ensure_ascii=False, indent=1)
    except Exception:
        pass
    return fit


if __name__ == "__main__":
    # автономный прогон: читает paper.json, пишет brain.json
    st = {}
    try:
        st = json.load(open("results/paper.json"))
    except Exception:
        pass
    tops = st.get("tops", [])

    def kl(sym, tf):
        import config
        from bt import data as _d
        try:
            from bt import live as _l
            return _l.klines_live(sym, tf, 12)
        except Exception:
            return _d.klines(sym, tf, 12)

    fit = regime_fit(tops, kl, paper_state=st if st else None)
    print("brain: fit for %d strategies" % len(fit), flush=True)
