"""STATUS: сводка состояния всей системы + журнал событий.

Собирает в results/system.json:
- стадию конвейера и прогресс в процентах (очередь/волны/золото);
- золото: сколько проверено, сколько прошло, список прошедших;
- мозг: режимы, выученные множители, уровень уверенности;
- страж реальности: множитель, просадка, алерты;
- paper: эквити, сделки, портфель с весами;
- последние ошибки (для аудита).

Пишет журнал results/system_log.jsonl (последние 500 записей)
и отчёт reports/SYSTEM.md. Панель читает results/system.json.
"""
import glob
import json
import os
import time

SYS = "results/system.json"
LOG = "results/system_log.jsonl"
MD = "reports/SYSTEM.md"


def load(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default


def jload_text(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prog = load("results/progress.json", {})
    total = int(prog.get("total", 0))
    done = int(prog.get("done", 0))
    pending = int(prog.get("pending", 0))
    errors = int(prog.get("errors", 0))
    pct = round(100.0 * done / total, 1) if total else 0.0

    # золото
    gold_files = glob.glob("results/gold/*.json")
    gold_passed = []
    gold_checked = 0
    for f in gold_files:
        r = jload_text(f)
        if not isinstance(r, dict):
            continue
        gold_checked += 1
        if r.get("gold"):
            gold_passed.append({"key": "%s-%s-%s" % (r.get("strategy"), r.get("symbol"), r.get("tf")),
                                "sharpe": r.get("base_sharpe")})
    # очередь: последние ошибки
    err_tasks = []
    for qf in sorted(glob.glob("results/queue*.json")):
        q = jload_text(qf) or []
        for t in q:
            if t.get("status") == "error":
                err_tasks.append({"id": t.get("id"), "error": (t.get("error") or "")[:120]})
    err_tasks = err_tasks[-10:]

    brain = load("results/brain.json", {})
    reality = load("results/reality.json", {})
    paper = load("results/paper.json", {})
    stats = paper.get("stats", {})

    # стадия конвейера
    if pending > 0:
        stage = "разгон очереди"
    elif gold_checked == 0:
        stage = "первое исследование"
    else:
        stage = "исследование + золото"

    # уровень обучения мозга: полная уверенность при 12 сделках в режиме
    learned = brain.get("learned", {})
    hist = load("results/brain_hist.json", {})
    fr = hist.get("family_regime", {})
    conf = {}
    for k, rec in fr.items():
        n = rec.get("n", 0)
        conf[k] = round(min(1.0, n / 12.0) * 100)

    sys = {
        "updated": now,
        "stage": stage,
        "queue": {"total": total, "done": done, "pending": pending,
                  "errors": errors, "progress_pct": pct},
        "gold": {"checked": gold_checked, "passed": len(gold_passed),
                 "passed_list": gold_passed},
        "brain": {"regimes": {k: len(v) for k, v in (brain.get("regimes") or {}).items()},
                  "learned": learned,
                  "confidence_pct": conf},
        "reality": {"mult": reality.get("global_mult", 1.0),
                    "drawdown_pct": reality.get("drawdown_pct", 0.0),
                    "alerts": reality.get("alerts", [])},
        "paper": {"equity": stats.get("equity"), "pnl_pct": stats.get("pnl_pct"),
                  "winrate": stats.get("winrate"), "n_trades": stats.get("n_trades"),
                  "open": stats.get("open"),
                  "portfolio": [{"key": t.get("key"), "weight": t.get("weight"),
                                 "fit": t.get("fit"), "gold": t.get("gold"),
                                 "signal": t.get("signal")} for t in (paper.get("tops") or [])]},
        "recent_errors": err_tasks,
    }
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    json.dump(sys, open(SYS, "w"), ensure_ascii=False, indent=1)

    # журнал: одна строка = один прогон
    entry = {"ts": now, "stage": stage, "pct": pct,
             "equity": stats.get("equity"), "pnl": stats.get("pnl_pct"),
             "gold_passed": len(gold_passed), "gold_checked": gold_checked,
             "reality_mult": sys["reality"]["mult"],
             "errors_new": errors}
    log = []
    if os.path.exists(LOG):
        try:
            log = [json.loads(l) for l in open(LOG) if l.strip()]
        except Exception:
            log = []
    log.append(entry)
    with open(LOG, "w") as f:
        for l in log[-500:]:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    # отчёт
    lines = ["# СИСТЕМА: сводка %s" % now, "",
             "- Стадия: %s, прогресс очереди: %.1f%%" % (stage, pct),
             "- Золото: %d/%d прошло проверку" % (len(gold_passed), gold_checked),
             "- Страж реальности: множитель %.2f, просадка %.2f%%" % (
                 sys["reality"]["mult"], sys["reality"]["drawdown_pct"]),
             "- Paper: equity %s, PnL %s%%, сделок %s, винрейт %s%%" % (
                 stats.get("equity"), stats.get("pnl_pct"), stats.get("n_trades"), stats.get("winrate")),
             "", "## Портфель", ""]
    for t in sys["paper"]["portfolio"]:
        g = " [GOLD]" if t.get("gold") else ""
        lines.append("- %s%s: вес %s, режим-множитель %s, сигнал %s" % (
            t["key"], g, t.get("weight"), t.get("fit"), t.get("signal")))
    lines.append("")
    lines.append("## Мозг (выученное)")
    for k, v in learned.items():
        lines.append("- %s: средний PnL %s%%/сделку, уверенность %s%%" % (k, v, conf.get(k, 0)))
    if err_tasks:
        lines.append("")
        lines.append("## Последние ошибки очереди")
        for e in err_tasks:
            lines.append("- %s: %s" % (e["id"], e["error"]))
    open(MD, "w").write("\n".join(lines) + "\n")
    print("status: stage=%s pct=%.1f gold=%d/%d" % (stage, pct, len(gold_passed), gold_checked), flush=True)


if __name__ == "__main__":
    main()
