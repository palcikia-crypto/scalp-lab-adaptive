"""Paper trading ADAPTIVE: реальные сигналы топ-стратегий на живых данных
Binance + адаптивное управление размером позиций + мозг рынка (brain.py).

Каждый запуск (воркер каждые 30 минут):
- берет топ-стратегии из результатов бэктестов (прошедшие фильтры);
- адаптивный аллокатор (allocator.py) назначает каждой стратегии вес
  по её реальным сделкам: минусовые серии -> меньше размер или shadow
  (сигналы пишутся, капитал не тратится), стабильный плюс -> вес до 1.5;
- мозг рынка (brain.py) определяет режим (trend/range/spike/chop) по живым
  свечам и даёт каждой стратегии множитель под текущий режим;
- на свежих данных генерирует сигнал последнего закрытого бара;
- открывает/закрывает виртуальные позиции, капитал $1000, сложный процент,
  вычитает funding по открытым позициям;
- пишет results/paper.json для панели (включая alloc с весами).
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd

import allocator
import config
from bt import data, strategies
try:
    from bt import live
except Exception:
    live = None
try:
    import brain
except Exception:
    brain = None

STATE_PATH = "results/paper.json"
CAPITAL_START = 1000.0
MAX_TRACK = 8
# Бог-уровень отбора: композитный скор; золотое подтверждение даёт бонус.
MIN_T = 1.5      # t-статистика OOS: ниже - эдж статистически не значим
MIN_PF = 1.01    # минимальный profit factor после издержек
STAB_REF = 0.35  # эталон стабильности параметров (ниже - штраф к скору)
LOOKBACK_DAYS = {"1m": 2, "5m": 7, "15m": 12, "1h": 40}
PAPER_BUDGET_S = 120
COST = {"taker": 2 * (config.FEE + config.SLIP),
        "maker": 2 * (config.FEE_MAKER + config.SLIP_MAKER)}


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                st = json.load(f)
            if "positions" in st and "closed" in st and "equity" in st:
                return st
        except Exception:
            pass
    return {"capital_start": CAPITAL_START, "equity": CAPITAL_START,
            "positions": {}, "closed": [], "updated": "", "runs": 0}


def save_state(st):
    os.makedirs("results", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)


def gold_keys():
    """Ключи (strategy-symbol-tf) с золотым подтверждением эджа."""
    out = set()
    for f in glob.glob("results/gold/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("gold"):
            out.add("-".join([r.get("strategy", ""), r.get("symbol", ""), r.get("tf", "")]))
    return out


def top_strategies():
    best = {}
    for f in glob.glob("results/backtest/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if not (isinstance(r, dict) and "oos" in r):
            continue
        k = (r.get("strategy"), r.get("symbol"), r.get("tf"))
        try:
            mt = os.path.getmtime(f)
        except Exception:
            mt = 0
        if k not in best or mt >= best[k][0]:
            best[k] = (mt, r)
    all_res = [v[1] for v in best.values()]
    bts = []
    cand = []
    stats = {"total": len(all_res), "low_trades": 0, "low_folds": 0,
             "low_pf": 0, "low_t": 0, "no_params": 0, "admitted": 0}
    gold = gold_keys()
    for r in all_res:
        o = r["oos"]
        reason = None
        if r.get("trades", 0) < 200:
            reason = "low_trades"
        elif r.get("folds", 0) < 8:
            reason = "low_folds"
        elif o.get("pf", 0) <= MIN_PF:
            reason = "low_pf"
        elif o.get("t", 0) <= MIN_T:
            reason = "low_t"
        if reason:
            stats[reason] += 1
            if o.get("sharpe", 0) > 0.5:
                cand.append({"strategy": r.get("strategy"),
                             "symbol": r.get("symbol"), "tf": r.get("tf"),
                             "sharpe": round(o.get("sharpe", 0), 2),
                             "t": round(o.get("t", 0), 2),
                             "pf": round(o.get("pf", 0), 3),
                             "stability": round(r.get("param_stability", 0), 2),
                             "reason": reason})
            continue
        # композитный скор: OOS Sharpe x штраф за нестабильность параметров,
        # золотое подтверждение добавляет бонус x1.25
        score = o.get("sharpe", 0) * min(1.0, r.get("param_stability", 0) / STAB_REF)
        if "-".join([r["strategy"], r["symbol"], r["tf"]]) in gold:
            score *= 1.25
        bts.append((score, r))
    bts.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen = set()
    for score, r in bts:
        key = (r["strategy"], r["symbol"], r["tf"])
        if key in seen:
            continue
        seen.add(key)
        try:
            params = json.loads(r.get("top_param", "{}"))
        except Exception:
            stats["no_params"] += 1
            continue
        if not params:
            stats["no_params"] += 1
            continue
        fee_mode = "maker" if r["strategy"] in strategies.MAKER_SET else "taker"
        out.append({"key": "-".join(key), "strategy": r["strategy"],
                    "symbol": r["symbol"], "tf": r["tf"],
                    "params": params, "fee_mode": fee_mode,
                    "gold": "-".join(key) in gold,
                    "oos_sharpe": round(r["oos"]["sharpe"], 2),
                    "score": round(score, 2)})
        if len(out) >= MAX_TRACK:
            break
    stats["admitted"] = len(out)
    cand.sort(key=lambda c: -c["sharpe"])
    return out, stats, cand[:12]


def close_pos(st, key, px, ts, reason):
    pos = st["positions"].pop(key)
    d = pos["signal"]
    fund = pos.get("fund", 0.0)
    w = pos.get("weight", 1.0)
    pnl = d * (px / pos["entry"] - 1.0) - COST[pos["fee_mode"]] - fund
    # адаптивный размер: PnL масштабируется весом аллокатора и мозгом на момент входа
    pnl_w = pnl * w
    if not pos.get("shadow"):
        st["equity"] = st["equity"] * (1.0 + pnl_w)
    st["closed"].append({
        "key": key, "strategy": pos["strategy"], "symbol": pos["symbol"],
        "tf": pos["tf"], "side": "long" if d > 0 else "short",
        "entry": pos["entry"], "exit": px,
        "pnl_pct": round(pnl_w * 100, 4),
        "pnl_raw_pct": round(pnl * 100, 4),
        "weight": w, "shadow": bool(pos.get("shadow")),
        "fund_pct": round(fund * 100, 4),
        "opened": pos["opened"], "closed_ts": ts,
        "fee_mode": pos["fee_mode"], "reason": reason})
    st["closed"] = st["closed"][-300:]


def get_klines(symbol, tf):
    """Живые свечи (лаг ~1 минута); при сбое - дневные архивы (лаг ~1 день)."""
    if live is not None:
        try:
            return live.klines_live(symbol, tf, LOOKBACK_DAYS.get(tf, 10))
        except Exception as e:
            print("live data failed, fallback to archives:", str(e)[:80], flush=True)
    return data.klines(symbol, tf, LOOKBACK_DAYS.get(tf, 10))


def main():
    st = load_state()
    tops, fstats, cands = top_strategies()
    # === адаптивный аллокатор: веса по реальным сделкам ===
    alloc = allocator.weights(st.get("closed", []), tops)
    st["alloc"] = alloc
    # === мозг рынка: режим и множители под него ===
    fit = {}
    if brain is not None:
        try:
            fit = brain.regime_fit(tops, get_klines, paper_state=st)
        except Exception as e:
            print("brain failed:", str(e)[:80], flush=True)
    st["brain_fit"] = fit
    top_keys = set(t["key"] for t in tops)
    # закрыть позиции стратегий, выпавших из топа
    for key in list(st["positions"].keys()):
        if key in top_keys:
            continue
        t = st["positions"][key]
        try:
            df = get_klines(t["symbol"], t["tf"])
            close_pos(st, key, float(df["close"].iloc[-1]),
                      str(df["dt"].iloc[-1]), "rotated out")
        except Exception:
            pass
    seen = st.setdefault("seen", {})
    live_keys = set(t["key"] for t in tops)
    for k in list(seen.keys()):
        if k not in live_keys:
            del seen[k]
    deadline = time.time() + PAPER_BUDGET_S
    for t in tops:
        if time.time() > deadline:
            print("paper: time budget reached, rest strategies next run", flush=True)
            break
        a = alloc.get(t["key"], {})
        shadow = a.get("mode") == "shadow"
        # вес = аллокатор x множитель мозга за режим рынка
        w = float(a.get("weight", 1.0)) * float(fit.get(t["key"], 1.0))
        if shadow:
            print("paper: shadow mode for", t["key"], "-", a.get("reason", ""), flush=True)
        try:
            df = get_klines(t["symbol"], t["tf"])
            if t["strategy"] in strategies.DERIV_SET:
                df = data.add_deriv(df, t["symbol"], min(LOOKBACK_DAYS.get(t["tf"], 10), 30), timeout=8)
            fn = strategies.STRATEGIES[t["strategy"]][0]
            sig = fn(df, **t["params"]).fillna(0.0)
        except Exception as e:
            print("paper skip", t["key"], str(e)[:80], flush=True)
            continue
        tf_min = int(t["tf"][:-1])
        fpb = None
        if "funding" in df.columns:
            fr = pd.to_numeric(df["funding"], errors="coerce").fillna(0.0).values
            fpb = fr * (tf_min / 480.0)
        dts = df["dt"].astype(str).values
        closes = df["close"].values
        vals = np.asarray(sig.values, dtype=float)
        if len(dts) < 2:
            continue
        t["signal"] = float(vals[-1])
        t["weight"] = round(w, 3)
        t["fit"] = float(fit.get(t["key"], 1.0))
        t["mode"] = a.get("mode", "live")
        nz = np.nonzero(vals)[0]
        t["sig_last"] = str(dts[nz[-1]]) if len(nz) else ""
        t["sig_pct"] = round(100.0 * len(nz) / len(vals), 2)
        last_seen = seen.get(t["key"], "")
        if last_seen:
            idx = np.nonzero(dts > last_seen)[0]
            start = int(idx[0]) if len(idx) else int(len(dts) - 1)
        else:
            start = int(len(dts) - 1)
        for i in range(start, len(dts)):
            s = float(vals[i])
            px = float(closes[i])
            ts = str(dts[i])
            pos = st["positions"].get(t["key"])
            if pos is not None and fpb is not None:
                # funding за бар удержания: лонг платит при положительном
                pos["fund"] = pos.get("fund", 0.0) + pos["signal"] * fpb[i]
            if s != 0.0:
                if pos is None or pos["signal"] != s:
                    if pos is not None:
                        close_pos(st, t["key"], px, ts, "signal flip")
                    st["positions"][t["key"]] = {
                        "strategy": t["strategy"], "symbol": t["symbol"],
                        "tf": t["tf"], "signal": s, "entry": px,
                        "opened": ts, "fee_mode": t["fee_mode"],
                        "oos_sharpe": t["oos_sharpe"], "last_ts": ts,
                        "last_px": px, "fund": 0.0,
                        "weight": w, "shadow": shadow}
                else:
                    pos["last_ts"] = ts
                    pos["last_px"] = px
            elif pos is not None:
                close_pos(st, t["key"], px, ts, "flat signal")
        seen[t["key"]] = str(dts[-1])
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    st["runs"] = int(st.get("runs", 0)) + 1
    st["tops"] = tops
    trades = [x for x in st["closed"] if not x.get("shadow")]
    wins = sum(1 for x in trades if x["pnl_pct"] > 0)
    shadow_n = sum(1 for x in st["closed"] if x.get("shadow"))
    st["stats"] = {
        "n_trades": len(trades),
        "wins": wins,
        "winrate": round(100.0 * wins / len(trades), 1) if trades else 0.0,
        "equity": round(st["equity"], 2),
        "pnl_pct": round((st["equity"] / st["capital_start"] - 1.0) * 100, 2),
        "open": len(st["positions"]),
        "shadow_trades": shadow_n,
        "shadow_open": sum(1 for p in st["positions"].values() if p.get("shadow"))}
    st["filter_stats"] = fstats
    st["candidates"] = cands
    save_state(st)
    print("paper: equity %.2f, trades %d, open %d, shadow %d" % (
        st["equity"], len(trades), len(st["positions"]), shadow_n), flush=True)


if __name__ == "__main__":
    main()
