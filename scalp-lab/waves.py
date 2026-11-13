"""Золотое подтверждение и реабилитация стратегий (worker-волны).

gold_wave: лидеры портфеля прогоняются через bt.robust.gold_check
(соседние параметры + подпериоды + Monte Carlo) -> results/gold/*.json.
Только стратегии с золотом получают бонус в портфеле.

rescue: стратегии с высоким Sharpe, но не прошедшие фильтр из-за малого
числа сделок, прогоняются на глубокой истории (дни x4) — если эдж
подтверждается на большом объёме данных, они попадут в портфель.
"""
import glob
import json

import config


def _existing_ids(queues):
    ids = set()
    for _, q in queues:
        for t in q:
            ids.add(t["id"])
    return ids


def _n_queues():
    return len(glob.glob("results/queue*.json"))


def gold_wave(queues, max_tasks=6):
    """Золотое подтверждение топ-лидеров (по OOS Sharpe)."""
    bts = []
    for f in glob.glob("results/backtest/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if isinstance(r, dict) and "oos" in r and r.get("trades", 0) >= 200:
            bts.append(r)
    bts.sort(key=lambda r: r["oos"].get("sharpe", 0), reverse=True)
    ids = _existing_ids(queues)
    tasks = []
    for r in bts:
        if len(tasks) >= max_tasks:
            break
        try:
            params = json.loads(r.get("top_param", "{}"))
        except Exception:
            continue
        if not params:
            continue
        tid = "gold-" + r["strategy"] + "-" + r["symbol"] + "-" + r["tf"]
        if tid in ids:
            continue
        ids.add(tid)
        cfg = config.TFS.get(r["tf"])
        if cfg is None:
            continue
        tasks.append({"id": tid, "kind": "gold",
                      "symbol": r["symbol"], "tf": r["tf"],
                      "strategy": r["strategy"], "params": params,
                      "days": cfg["days"], "train_days": cfg["train_days"],
                      "test_days": cfg["test_days"],
                      "wave": "gold", "status": "pending"})
    if not tasks:
        return 0
    path = "results/queue%d.json" % (_n_queues() + 1)
    with open(path, "w") as f:
        json.dump(tasks, f, ensure_ascii=False)
    print("gold: %d leaders -> robustness check" % len(tasks), flush=True)
    return len(tasks)


def rescue(queues, max_tasks=10):
    """Реабилитация: высокий Sharpe + мало сделок -> глубокий прогон."""
    bts = []
    for f in glob.glob("results/backtest/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if not (isinstance(r, dict) and "oos" in r):
            continue
        o = r["oos"]
        if r.get("trades", 0) < 200 and o.get("sharpe", 0) > 1.2 and o.get("t", 0) > 1.5:
            bts.append(r)
    bts.sort(key=lambda r: r["oos"].get("sharpe", 0), reverse=True)
    ids = _existing_ids(queues)
    tasks = []
    for r in bts:
        if len(tasks) >= max_tasks:
            break
        cfg = config.TFS.get(r["tf"])
        if cfg is None:
            continue
        tid = "rsc-" + r["strategy"] + "-" + r["symbol"] + "-" + r["tf"]
        if tid in ids:
            continue
        ids.add(tid)
        tasks.append({"id": tid, "kind": "backtest",
                      "symbol": r["symbol"], "tf": r["tf"],
                      "strategy": r["strategy"],
                      "days": min(cfg["days"] * 4, 1460),
                      "train_days": cfg["train_days"] * 4,
                      "test_days": cfg["test_days"] * 4,
                      "wave": "rescue", "status": "pending"})
    if not tasks:
        return 0
    path = "results/queue%d.json" % (_n_queues() + 1)
    with open(path, "w") as f:
        json.dump(tasks, f, ensure_ascii=False)
    print("rescue: %d candidates -> deep re-run" % len(tasks), flush=True)
    return len(tasks)
