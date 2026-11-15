import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

MIN_TRADES = 200
MIN_FOLDS = 8


def p_from_t(t):
    return math.erfc(abs(t) / math.sqrt(2.0))


def bh_keep(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    ok = np.where(ranked <= thresh)[0]
    if len(ok) == 0:
        return []
    kmax = int(ok.max())
    return [int(i) for i in order[:kmax + 1]]


def fmt(x):
    return ("%.4f" % x) if abs(x) < 100 else ("%.1f" % x)


def fee_mode(r):
    return r.get("fee_mode", "taker")


def maker_sharpe(r):
    m = r.get("oos_maker")
    return m["sharpe"] if isinstance(m, dict) else ""


def maker_pf(r):
    m = r.get("oos_maker")
    return m["pf"] if isinstance(m, dict) else ""


def economics(r):
    """Экономика стратегии: сложный процент в день и в месяц,
    сделок в день, прибыль на капитал $1000."""
    o = r["oos"]
    days = max(int(r.get("days", 1)), 1)
    total = o.get("total", 0.0)
    if total <= -1.0:
        daily = total / days
    else:
        daily = (1.0 + total) ** (1.0 / days) - 1.0
    monthly = (1.0 + daily) ** 30 - 1.0
    tpd = r["trades"] / float(days)
    return daily, monthly, tpd


def main():
    os.makedirs("reports", exist_ok=True)
    best = {}
    for f in glob.glob("results/backtest/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if isinstance(r, dict) and "oos" in r:
            k = (r.get("strategy"), r.get("symbol"), r.get("tf"))
            try:
                mt = os.path.getmtime(f)
            except Exception:
                mt = 0
            if k not in best or mt >= best[k][0]:
                best[k] = (mt, r)
    bts = [v[1] for v in best.values()]
    scans = []
    for f in glob.glob("results/scan/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        for x in r.get("findings", []):
            y = dict(x)
            y["symbol"] = r.get("symbol")
            y["tf"] = r.get("tf")
            scans.append(y)

    lines = []
    lines.append("# Crypto Scalp Lab - отчет")
    lines.append("")
    lines.append("Все результаты - только out-of-sample (walk-forward), с учетом комиссий и проскальзывания.")
    lines.append("maker = вход лимитными ордерами (0.02%), taker = по рынку (0.05%).")
    lines.append("Экономика: сложный процент; капитал $1000, размер позиции 100%.")
    lines.append("")

    lines.append("## Прогресс")
    try:
        prog = json.load(open("results/progress.json"))
        lines.append("- готово задач: %d из %d (ошибок: %d), обновлено %s" % (
            prog.get("done", 0), prog.get("total", 0), prog.get("errors", 0), prog.get("updated", "")))
    except Exception:
        lines.append("- пока нет данных")
    lines.append("")

    passed = [r for r in bts
              if r["trades"] >= MIN_TRADES
              and r["folds"] >= MIN_FOLDS
              and r["oos"]["pf"] > 1.0
              and r["oos"]["t"] > 1.0]
    passed.sort(key=lambda r: r["oos"]["sharpe"], reverse=True)

    lines.append("## Лидерборд стратегий (OOS) с экономикой")
    lines.append("")
    lines.append("| # | стратегия | пара | TF | Sharpe | PF | t | сделок | сд/день | в день | в месяц | $/мес на $1000 | стаб. | режим |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(passed[:30]):
        o = r["oos"]
        d, m, tpd = economics(r)
        robust = "*" if (r["param_stability"] >= 0.5 and o["t"] > 2.0) else ""
        lines.append("| %d | %s%s | %s | %s | %.2f | %.2f | %.1f | %d | %.1f | %+.3f%% | %+.2f%% | $%+.0f | %.0f%% | %s |" % (
            i + 1, r["strategy"], robust, r["symbol"], r["tf"], o["sharpe"], o["pf"],
            o["t"], r["trades"], tpd, d * 100, m * 100, m * 1000,
            r["param_stability"] * 100, fee_mode(r)))
    lines.append("")
    lines.append("Робастные отмечены звездочкой: стабильные параметры между фолдами + t>2.")
    lines.append("Экономика - оценка на историческом OOS периоде, не прогноз будущей прибыли.")
    lines.append("")

    lines.append("## Устойчивость по монетам (агрегация)")
    lines.append("")
    lines.append("Закономерность живет на многих парах; подгонка - обычно на одной.")
    lines.append("")
    lines.append("| стратегия | TF | пар | медиана Sharpe | медиана PF | пар с Sharpe>0 | режим |")
    lines.append("|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in passed:
        groups[(r["strategy"], r["tf"])].append(r)
    for (st, tf), xs in sorted(groups.items()):
        sh = [x["oos"]["sharpe"] for x in xs]
        pf = [x["oos"]["pf"] for x in xs]
        lines.append("| %s | %s | %d | %.2f | %.2f | %d | %s |" % (
            st, tf, len(xs), float(np.median(sh)), float(np.median(pf)),
            sum(1 for s in sh if s > 0), fee_mode(xs[0])))
    lines.append("")

    if scans:
        pvals = [p_from_t(x["t"]) for x in scans]
        keep = set(bh_keep(pvals))
        surv = [x for i, x in enumerate(scans) if i in keep]
        groups2 = defaultdict(list)
        for x in surv:
            groups2[(x["event"], x["horizon"])].append(x)
        lines.append("## Паттерны, выжившие поправку Бенджамини-Хохберга (FDR 5%%)")
        lines.append("")
        if groups2:
            lines.append("| событие | горизонт(баров) | пар | средн. форвард | медиана t |")
            lines.append("|---|---|---|---|---|")
            for (ev, hz), xs in sorted(groups2.items(), key=lambda kv: -abs(np.mean([a["mean_fwd"] for a in kv[1]]))):
                mf = np.mean([a["mean_fwd"] for a in xs])
                mt = np.median([a["t"] for a in xs])
                lines.append("| %s | %d | %d | %s | %.1f |" % (ev, hz, len(xs), fmt(mf), mt))
        else:
            lines.append("Пока ни один паттерн не прошел статзначимую проверку (это нормальный результат - зато без подгонок).")
        lines.append("")

    with open("reports/REPORT.md", "w") as f:
        f.write(chr(10).join(lines))

    with open("reports/leaderboard.csv", "w") as f:
        f.write("strategy,symbol,tf,oos_sharpe,oos_pf,oos_t,trades,param_stability,is_sharpe_avg,folds,fee_mode,maker_sharpe,maker_pf,daily_pct,monthly_pct,trades_per_day,usd_per_month_1000" + chr(10))
        for r in passed[:200]:
            o = r["oos"]
            d, m, tpd = economics(r)
            f.write(",".join(str(x) for x in [r["strategy"], r["symbol"], r["tf"], round(o["sharpe"], 3), round(o["pf"], 3), round(o["t"], 2), r["trades"], round(r["param_stability"], 2), round(r["is_sharpe_avg"], 2), r["folds"], fee_mode(r), maker_sharpe(r), maker_pf(r), round(d, 6), round(m, 6), round(tpd, 2), round(m * 1000, 2)]) + chr(10))

    passed_ids = set(id(r) for r in passed)
    bts.sort(key=lambda r: r["oos"]["sharpe"], reverse=True)
    with open("reports/archive.csv", "w") as f:
        f.write("strategy,symbol,tf,oos_sharpe,oos_pf,oos_t,trades,folds,param_stability,days,wave,verdict" + chr(10))
        for r in bts:
            o = r["oos"]
            verdict = "passed" if id(r) in passed_ids else "filtered"
            f.write(",".join(str(x) for x in [r["strategy"], r["symbol"], r["tf"], round(o["sharpe"], 3), round(o["pf"], 3), round(o["t"], 2), r["trades"], r["folds"], round(r["param_stability"], 2), r.get("days", ""), r.get("wave", ""), verdict]) + chr(10))

    print("backtest results:", len(bts), "passed filters:", len(passed), "scan findings:", len(scans))


if __name__ == "__main__":
    main()
