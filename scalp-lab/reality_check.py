"""REALITY CHECK: страж соответствия paper <-> реальный рынок.

Задача (по опыту: на бумаге плюс, на реале минус):
1. Риск-контроль: следит за эквити, при просадке от пика автоматически
   режет глобальный размер (risk-off), при глубокой - сильнее.
2. Дрейф стратегии: если последние сделки стратегии стабильно хуже
   её бэктест-ожидания - флаг дрейфа (weight x0.5 рекомендация).
3. Аудит исполнения: pnl-арифметика, зависшие позиции >24ч.
4. Отчёт: results/reality.json + reports/REALITY.md.

paper.py читает results/reality.json и умножает веса на global_mult.
"""
import glob
import json
import os
import time

STATE = "results/reality.json"
HIST = "results/reality_hist.json"
REPORT = "reports/REALITY.md"

DD_SOFT = 0.06    # просадка 6% от пика -> глобальный размер x0.6
DD_HARD = 0.12    # 12% -> x0.3
DRIFT_TRADES = 8  # минимум сделок для суждения о дрейфе
DRIFT_GAP = 0.25  # paper хуже бэктеста на 0.25%/сделку -> дрейф


def load(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return default


def backtest_expect(key):
    """Ожидание по бэктесту: средний pnl на сделку (%), из OOS-результата."""
    parts = key.split("-")
    if len(parts) < 3:
        return None
    strat, sym = parts[0], parts[1]
    tf = parts[-1]
    best = None
    for f in glob.glob("results/backtest/ev-%s-%s-%s*.json" % (strat, sym, tf)):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if not isinstance(r, dict) or "oos" not in r:
            continue
        if best is None or os.path.getmtime(f) > best[0]:
            best = (os.path.getmtime(f), r)
    if not best:
        return None
    o = best[1].get("oos", {})
    n = best[1].get("trades", 0)
    for fname in ("avg_pnl", "avg", "pnl_pct"):
        v = o.get(fname)
        if isinstance(v, (int, float)):
            return float(v)
    tot = o.get("total_pnl")
    if isinstance(tot, (int, float)) and n:
        return float(tot) / n * 100.0
    tr = o.get("total_return_pct")
    if isinstance(tr, (int, float)) and n:
        return float(tr) / n
    return None


def main():
    st = load("results/paper.json", {})
    hist = load(HIST, {"peak": 1000.0, "snapshots": []})
    alerts = []
    actions = []

    equity = float(st.get("equity", 1000.0))
    peak = max(float(hist.get("peak", 1000.0)), equity)
    hist["peak"] = peak
    dd = (peak - equity) / peak if peak > 0 else 0.0

    # === 1. глобальный risk-off по просадке ===
    global_mult = 1.0
    if dd >= DD_HARD:
        global_mult = 0.3
        alerts.append("ПРОСАДКА %.1f%% от пика: глобальный размер x0.3" % (dd * 100))
        actions.append("risk_off_hard")
    elif dd >= DD_SOFT:
        global_mult = 0.6
        alerts.append("Просадка %.1f%% от пика: глобальный размер x0.6" % (dd * 100))
        actions.append("risk_off_soft")

    # === 2. дрейф стратегий: paper хуже бэктеста ===
    per_strategy = {}
    closed = st.get("closed", [])
    by_key = {}
    for x in closed:
        by_key.setdefault(x.get("key", ""), []).append(x)
    for key, trades in by_key.items():
        recent = trades[-DRIFT_TRADES:]
        if len(recent) < DRIFT_TRADES:
            continue
        avg_paper = sum(float(x.get("pnl_raw_pct", 0)) for x in recent) / len(recent)
        exp = backtest_expect(key)
        rec = {"n_paper": len(trades), "avg_paper_pct": round(avg_paper, 3),
               "bt_expect_pct": round(exp, 3) if exp is not None else None}
        if exp is not None and avg_paper < exp - DRIFT_GAP:
            rec["drift"] = True
            rec["advice"] = "weight x0.5"
            alerts.append("ДРЕЙФ %s: paper %.2f%%/сделку vs бэктест %.2f%%/сделку" % (
                key, avg_paper, exp))
            actions.append("drift:" + key)
        elif exp is not None:
            rec["drift"] = False
        per_strategy[key] = rec

    # === 3. аудит исполнения ===
    bad_math = 0
    for x in closed:
        raw = float(x.get("pnl_raw_pct", 0))
        w = float(x.get("weight", 1.0))
        if abs(raw * w - float(x.get("pnl_pct", 0))) > 0.05:
            bad_math += 1
    if bad_math:
        alerts.append("АРИФМЕТИКА: %d сделок с несходящимся pnl (возможен баг)" % bad_math)
    now = time.time()
    stale = 0
    for p in (st.get("positions") or {}).values():
        try:
            t0 = time.mktime(time.strptime(p.get("opened", ""), "%Y-%m-%d %H:%M:%S"))
            if now - t0 > 86400:
                stale += 1
        except Exception:
            pass
    if stale:
        alerts.append("ЗАВИСШИЕ: %d позиций открыты >24ч (проверить сигнал/ротацию)" % stale)

    out = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "equity": round(equity, 2), "peak": round(peak, 2),
        "drawdown_pct": round(dd * 100, 2),
        "global_mult": global_mult,
        "alerts": alerts, "actions": actions,
        "per_strategy": per_strategy,
    }
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    json.dump(out, open(STATE, "w"), ensure_ascii=False, indent=1)
    hist["snapshots"] = (hist.get("snapshots", []) + [
        {"ts": out["updated"], "equity": round(equity, 2), "dd": round(dd * 100, 2),
         "mult": global_mult}])[-300:]
    json.dump(hist, open(HIST, "w"), ensure_ascii=False, indent=1)

    lines = ["# REALITY CHECK %s" % out["updated"], "",
             "- equity: %.2f (peak %.2f, просадка %.2f%%)" % (equity, peak, dd * 100),
             "- глобальный множитель размера: %.2f" % global_mult,
             "- алертов: %d" % len(alerts), ""]
    lines += (["- " + a for a in alerts] or ["- всё чисто"])
    open(REPORT, "w").write("\n".join(lines) + "\n")
    print("reality: mult %.2f, dd %.2f%%, alerts %d" % (global_mult, dd * 100, len(alerts)), flush=True)


if __name__ == "__main__":
    main()
