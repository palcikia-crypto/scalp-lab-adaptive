import argparse
import glob
import json
import os
import time

import config
import waves
from runner import run_task

MAX_ATTEMPTS = 3


def queue_files():
    files = sorted(glob.glob("results/queue*.json"))
    return files if files else ["results/queue.json"]


def save_progress(all_tasks, start, cycle):
    done = sum(1 for t in all_tasks if t.get("status") == "done")
    err = sum(1 for t in all_tasks if t.get("status") == "error")
    pend = sum(1 for t in all_tasks if t.get("status") == "pending")
    prog = {"total": len(all_tasks), "done": done, "errors": err,
            "pending": pend, "cycle": cycle,
            "elapsed_min": round((time.time() - start) / 60, 1),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs("results", exist_ok=True)
    with open("results/progress.json", "w") as f:
        json.dump(prog, f)
    hist = []
    try:
        with open("results/history.json") as f:
            hist = json.load(f)
    except Exception:
        hist = []
    if not hist or hist[-1].get("done") != done:
        hist.append({"updated": prog["updated"], "done": done,
                     "total": len(all_tasks), "errors": err})
        with open("results/history.json", "w") as f:
            json.dump(hist[-500:], f)


def load_queues():
    os.makedirs("results", exist_ok=True)
    if not os.path.exists("results/queue.json"):
        from queue_builder import build
        with open("results/queue.json", "w") as f:
            json.dump(build(), f, ensure_ascii=False)
    queues = []
    for p in queue_files():
        if not os.path.exists(p):
            continue
        with open(p) as f:
            q = json.load(f)
        for t in q:
            # tasks stuck in running (cancelled run) get retried
            if t.get("status") == "running":
                t["status"] = "pending"
        queues.append((p, q))
    return queues


def evolve(queues):
    """All tasks finished: build a new deeper research wave from the
    best out-of-sample survivors (longer history, stronger validation)."""
    bts = []
    for f in glob.glob("results/backtest/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if isinstance(r, dict) and "oos" in r:
            bts.append(r)
    passed = [r for r in bts
              if r.get("trades", 0) >= 200
              and r.get("folds", 0) >= 8
              and r["oos"].get("pf", 0) > 1.0
              and r["oos"].get("t", 0) > 1.0]
    passed.sort(key=lambda r: r["oos"].get("sharpe", 0), reverse=True)
    ids = set()
    for _, q in queues:
        for t in q:
            ids.add(t["id"])
    tasks = []
    seen = set()
    for r in passed:
        key = (r["strategy"], r["symbol"], r["tf"])
        if key in seen:
            continue
        seen.add(key)
        cfg = config.TFS.get(r["tf"])
        if cfg is None:
            continue
        base = "ev-" + r["strategy"] + "-" + r["symbol"] + "-" + r["tf"]
        for mult in [2, 4]:
            tid = base + "-v" + str(mult)
            if tid in ids:
                continue
            ids.add(tid)
            tasks.append({"id": tid, "kind": "backtest",
                          "symbol": r["symbol"], "tf": r["tf"],
                          "strategy": r["strategy"],
                          "days": min(cfg["days"] * mult, 1460),
                          "train_days": cfg["train_days"] * mult,
                          "test_days": cfg["test_days"] * mult,
                          "wave": "deep", "status": "pending"})
        if len(tasks) >= 30:
            break
    if not tasks:
        return 0
    n = len(queue_files())
    path = "results/queue%d.json" % (n + 1)
    with open(path, "w") as f:
        json.dump(tasks, f, ensure_ascii=False)
    print("evolve: new deep wave with %d tasks -> %s" % (len(tasks), path), flush=True)
    return len(tasks)


def frontier(queues):
    """Бог-уровень: фронт-волна исследований. Пока существуют
    непротестированные комбинации (стратегия x пара x TF), система сама
    ставит их в очередь. Приоритет - стратегиям, уже доказавшим эдж
    на других парах (число строк в лидерборде)."""
    from bt.strategies import STRATEGIES
    tested = set()
    for f in glob.glob("results/backtest/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if isinstance(r, dict) and r.get("strategy"):
            tested.add((r["strategy"], r.get("symbol"), r.get("tf")))
    strength = {}
    try:
        with open("reports/leaderboard.csv") as fh:
            fh.readline()
            for line in fh:
                p = line.strip().split(",")
                if len(p) >= 3:
                    strength[p[0]] = strength.get(p[0], 0) + 1
    except Exception:
        pass
    ids = set()
    for _, q in queues:
        for t in q:
            ids.add(t["id"])
    tasks = []
    for st_name in sorted(STRATEGIES, key=lambda s: -strength.get(s, 0)):
        for tf, cfg in config.TFS.items():
            for sym in config.PAIRS:
                if (st_name, sym, tf) in tested:
                    continue
                tid = "fr-" + st_name + "-" + sym + "-" + tf
                if tid in ids:
                    continue
                ids.add(tid)
                tasks.append({"id": tid, "kind": "backtest",
                              "symbol": sym, "tf": tf,
                              "strategy": st_name,
                              "days": cfg["days"],
                              "train_days": cfg["train_days"],
                              "test_days": cfg["test_days"],
                              "wave": "frontier", "status": "pending"})
    tasks = tasks[:40]
    if not tasks:
        return 0
    n = len(queue_files())
    path = "results/queue%d.json" % (n + 1)
    with open(path, "w") as f:
        json.dump(tasks, f, ensure_ascii=False)
    print("frontier: %d untested combos -> %s" % (len(tasks), path), flush=True)
    return len(tasks)


ROTATE_DAYS = 7


def rotate_leaders(queues):
    """Еженедельная ротация: топ-стратегии лидерборда перепроверяются на
    свежих данных; устаревшие результаты заменяются (dedup по mtime)."""
    try:
        with open("results/rotation.json") as fh:
            rot = json.load(fh)
    except Exception:
        rot = {}
    last = rot.get("last_rot", "")
    if last:
        try:
            age = time.time() - time.mktime(time.strptime(last, "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            age = 1e18
        if age < ROTATE_DAYS * 86400:
            return 0
    if not os.path.exists("reports/leaderboard.csv"):
        return 0
    rows = []
    with open("reports/leaderboard.csv") as fh:
        fh.readline()
        for line in fh:
            p = line.strip().split(",")
            if len(p) >= 3:
                rows.append(p)
    ids = set()
    for _, q in queues:
        for t in q:
            ids.add(t["id"])
    datekey = time.strftime("%Y%m%d", time.gmtime())
    tasks = []
    seen = set()
    for p in rows:
        st, sym, tf = p[0], p[1], p[2]
        if (st, sym, tf) in seen:
            continue
        seen.add((st, sym, tf))
        tid = "rot-" + datekey + "-" + st + "-" + sym + "-" + tf
        if tid in ids:
            continue
        cfg = config.TFS.get(tf)
        if cfg is None:
            continue
        tasks.append({"id": tid, "kind": "backtest",
                      "symbol": sym, "tf": tf, "strategy": st,
                      "days": cfg["days"], "train_days": cfg["train_days"],
                      "test_days": cfg["test_days"], "wave": "rotation",
                      "status": "pending"})
        if len(tasks) >= 10:
            break
    if not tasks:
        return 0
    n = len(queue_files())
    path = "results/queue%d.json" % (n + 1)
    with open(path, "w") as fh:
        json.dump(tasks, fh, ensure_ascii=False)
    with open("results/rotation.json", "w") as fh:
        json.dump({"last_rot": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "tasks": len(tasks)}, fh)
    print("rotation: %d re-verify tasks -> %s" % (len(tasks), path), flush=True)
    return len(tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=40)
    args = ap.parse_args()
    start = time.time()
    os.makedirs("reports", exist_ok=True)
    queues = load_queues()
    done = 0
    while time.time() - start < args.budget * 60 - 300:
        task = None
        for path, q in queues:
            for t in q:
                if t.get("status") == "pending":
                    task = (path, q, t)
                    break
            if task:
                break
        if not task:
            break
        path, q, t = task
        t["status"] = "running"
        with open(path, "w") as f:
            json.dump(q, f, ensure_ascii=False)
        try:
            res = run_task(t)
            out = os.path.join("results", t["kind"], t["id"] + ".json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump(res, f)
            t["status"] = "done"
            t["result"] = out
            done += 1
            print("done", t["id"], flush=True)
        except Exception as e:
            att = int(t.get("attempts", 0)) + 1
            t["attempts"] = att
            t["error"] = str(e)[:300]
            if att < MAX_ATTEMPTS:
                t["status"] = "pending"  # auto-retry
            else:
                t["status"] = "error"
            print("error(%d/%d)" % (att, MAX_ATTEMPTS), t["id"], str(e)[:120], flush=True)
        with open(path, "w") as f:
            json.dump(q, f, ensure_ascii=False)
        all_tasks = [x for _, qq in queues for x in qq]
        save_progress(all_tasks, start, "running")
    all_tasks = [x for _, qq in queues for x in qq]
    pend = sum(1 for t in all_tasks if t.get("status") == "pending")
    if pend == 0:
        evolve(queues)
        frontier(queues)
        rotate_leaders(queues)
        # ADAPTIVE: золотое подтверждение лидеров + реабилитация кандидатов
        waves.gold_wave(queues)
        waves.rescue(queues)
    queues = load_queues()
    all_tasks = [x for _, qq in queues for x in qq]
    save_progress(all_tasks, start, "final")
    print("processed", done)


if __name__ == "__main__":
    main()
