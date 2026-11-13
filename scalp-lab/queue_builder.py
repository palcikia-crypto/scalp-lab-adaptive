import json
import os

import config
from bt.strategies import STRATEGIES


def build():
    tasks = []
    for sym in config.PAIRS:
        for tf, cfg in config.TFS.items():
            tasks.append({"id": "scan-" + sym + "-" + tf, "kind": "scan",
                          "symbol": sym, "tf": tf, "days": cfg["days"],
                          "status": "pending"})
            for st in STRATEGIES:
                tasks.append({"id": "bt-" + st + "-" + sym + "-" + tf,
                              "kind": "backtest", "symbol": sym, "tf": tf,
                              "strategy": st, "days": cfg["days"],
                              "train_days": cfg["train_days"],
                              "test_days": cfg["test_days"],
                              "status": "pending"})
    return tasks


def main():
    os.makedirs("results", exist_ok=True)
    tasks = build()
    with open(os.path.join("results", "queue.json"), "w") as f:
        json.dump(tasks, f, ensure_ascii=False)
    print(len(tasks), "tasks")
