"""Адаптивный аллокатор ScalpLab: размер позиции по стратегии зависит
от её РЕАЛЬНЫХ результатов в бумажной торговле.

Правила (саморегулировка):
- новый ключ: вес 1.0, режим live;
- суммарный минус или 3+ убытка подряд: вес режется вдвое (меньше риск);
- 5+ убытков подряд или просадка серии: стратегия уходит в shadow —
  сигналы пишутся в журнал, но капитал не тратится. Вернётся в live,
  когда серия сделок снова станет плюсовой;
- прибыльные стратегии со стабильным winrate получают повышенный вес;
- вес ограничен [0, 1.5].

Веса пересчитываются на каждом запуске paper.py и пишутся в
paper.json -> alloc (панель и бот читают их оттуда).
"""

RECENT = 10          # сколько последних сделок смотрим
LOSS_STREAK_HALF = 3 # убытков подряд -> вес /2
LOSS_STREAK_SHADOW = 5
MAX_W = 1.5
MIN_LIVE_W = 0.25


def _series(closed, key):
    out = []
    for x in closed:
        if x.get("key") == key and not x.get("shadow"):
            out.append(x)
    return out[-RECENT:]


def weights(closed, tops):
    alloc = {}
    for t in tops:
        key = t["key"]
        tr = _series(closed, key)
        if not tr:
            alloc[key] = {"weight": 1.0, "mode": "live",
                          "reason": "новая, сделок ещё нет"}
            continue
        pnls = [x["pnl_pct"] for x in tr]
        s = sum(pnls)
        wr = 100.0 * sum(1 for p in pnls if p > 0) / len(pnls)
        # серия убытков в конце
        streak = 0
        for p in reversed(pnls):
            if p <= 0:
                streak += 1
            else:
                break
        w = 1.0
        mode = "live"
        reason = "норма"
        if streak >= LOSS_STREAK_SHADOW or (s < -5.0 and streak >= 3):
            mode = "shadow"
            w = 0.0
            reason = "%d убытков подряд / сумма %.1f%% — пауза, сигналы пишутся без риска" % (streak, s)
        elif s < 0 or streak >= LOSS_STREAK_HALF:
            w = 0.5
            reason = "минусовая серия (%.1f%%, %d убытков подряд) — размер снижен" % (s, streak)
        elif wr >= 60.0 and s > 0:
            w = min(MAX_W, 1.0 + s / 20.0)
            reason = "стабильный плюс (wr %.0f%%, +%.1f%%) — размер повышен" % (wr, s)
        w = max(0.0, min(MAX_W, w))
        if mode == "live" and w < MIN_LIVE_W:
            mode = "shadow"
            reason += " — вес ниже порога, переведена в shadow"
        alloc[key] = {"weight": round(w, 2), "mode": mode,
                      "reason": reason, "recent_pnl": round(s, 2),
                      "recent_wr": round(wr, 1), "loss_streak": streak,
                      "n_recent": len(tr)}
    return alloc
