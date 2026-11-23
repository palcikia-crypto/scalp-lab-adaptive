# -*- coding: utf-8 -*-
# Архив ВСЕХ стратегий, которые когда-либо разрабатывала система.
# Источники: результаты бэктестов всех волн (results/backtest/*.json),
# сканы (results/scan/*.json), все задачи всех очередей (results/queue*.json),
# лидерборд (reports/leaderboard.csv) и бумажный портфель (results/paper.json).
# Архив ведётся инкрементально: ничего не теряется между прогонами.
# Выходы:
#   results/archive/strategies.jsonl — полная база, по строке на комбинацию
#   reports/strategy_archive.csv     — плоская сводка (по sharpe, по убыванию)
#   results/archive/stats.json       — счётчики по волнам
import csv
import glob
import json
import os
import time


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def lb_keys():
    keys = set()
    try:
        with open("reports/leaderboard.csv") as fh:
            fh.readline()
            for line in fh:
                p = line.strip().split(",")
                if len(p) >= 3:
                    keys.add(p[0] + "-" + p[1] + "-" + p[2])
    except Exception:
        pass
    return keys


def live_keys():
    keys = set()
    try:
        with open("results/paper.json") as fh:
            p = json.load(fh)
        for t in (p.get("tops") or []):
            if (t.get("mode") or "live") == "live":
                keys.add(t.get("key"))
    except Exception:
        pass
    return keys


def sharpe_of(e):
    s = e.get("sharpe")
    return s if isinstance(s, (int, float)) else -999.0


def main():
    os.makedirs("results/archive", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    lbk = lb_keys()
    live = live_keys()
    arch = {}
    try:
        with open("results/archive/strategies.jsonl") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("id"):
                        arch[e["id"]] = e
                except Exception:
                    continue
    except IOError:
        pass
    n_bt = 0
    for f in sorted(glob.glob("results/backtest/*.json")):
        try:
            with open(f) as fh:
                r = json.load(fh)
        except Exception:
            continue
        if not isinstance(r, dict) or not r.get("strategy"):
            continue
        n_bt += 1
        oos = r.get("oos") or {}
        e = {
            "id": os.path.basename(f)[:-5],
            "kind": "backtest",
            "wave": r.get("wave") or "base",
            "strategy": r.get("strategy"),
            "symbol": r.get("symbol"),
            "tf": r.get("tf"),
            "days": r.get("days"),
            "train_days": r.get("train_days"),
            "test_days": r.get("test_days"),
            "trades": r.get("trades"),
            "folds": r.get("folds"),
            "sharpe": oos.get("sharpe"),
            "pf": oos.get("pf"),
            "t": oos.get("t"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(f))),
            "key": "%s-%s-%s" % (r.get("strategy"), r.get("symbol"), r.get("tf")),
        }
        e["admitted"] = e["key"] in lbk
        e["live"] = e["key"] in live
        prev = arch.get(e["id"])
        if prev is None or str(prev.get("updated", "")) < e["updated"]:
            arch[e["id"]] = e
    # задачи из очередей: покрывают даже те, у кого ещё нет файла результата
    for qf in sorted(glob.glob("results/queue*.json")):
        try:
            with open(qf) as fh:
                q = json.load(fh)
        except Exception:
            continue
        for t in q:
            if not isinstance(t, dict) or not t.get("id"):
                continue
            prev = arch.setdefault(t["id"], {
                "id": t["id"], "kind": t.get("kind"),
                "wave": t.get("wave") or "base", "sharpe": None})
            prev["wave"] = t.get("wave") or prev.get("wave") or "base"
            if prev.get("strategy") is None:
                prev["strategy"] = t.get("strategy")
                prev["symbol"] = t.get("symbol")
                prev["tf"] = t.get("tf")
    waves = {}
    admitted = 0
    for e in arch.values():
        w = e.get("wave") or "base"
        waves[w] = waves.get(w, 0) + 1
        if e.get("admitted"):
            admitted += 1
    rows = sorted(arch.values(), key=lambda e: (-sharpe_of(e), e["id"]))
    with open("results/archive/strategies.jsonl", "w") as fh:
        for e in rows:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open("reports/strategy_archive.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "symbol", "tf", "wave", "sharpe", "pf", "t", "trades",
                    "folds", "days", "admitted", "live", "id"])
        for e in rows:
            w.writerow([e.get("strategy"), e.get("symbol"), e.get("tf"), e.get("wave"),
                        e.get("sharpe"), e.get("pf"), e.get("t"), e.get("trades"),
                        e.get("folds"), e.get("days"),
                        1 if e.get("admitted") else 0, 1 if e.get("live") else 0, e.get("id")])
    with open("results/archive/stats.json", "w") as fh:
        json.dump({"total": len(arch), "backtest_results": n_bt, "admitted": admitted,
                   "waves": waves, "updated": now_iso()}, fh, indent=1)
    print("archive: %d стратегий в архиве (%d бэктест-результатов), волны: %s" % (len(arch), n_bt, waves), flush=True)


if __name__ == "__main__":
    main()
