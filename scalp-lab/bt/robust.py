"""Золотое подтверждение эджа (robustness): стратегия получает "золото"
только если её edge выдерживает три независимые проверки:

1. Соседние параметры: +-20% по каждому числовому параметру — медиана
   OOS Sharpe соседей должна быть положительной (эдж не узкий пик подгонки).
2. Подпериоды: OOS-доходность делится пополам — обе половины должны быть
   в плюсе по Sharpe (эдж стабилен во времени).
3. Monte Carlo: 200 бутстрап-перемешиваний фолдов OOS —
   5-перцентиль суммарной доходности не должен быть глубже -10%
   (результат не везение одной серии сделок).

Результат пишется в results/gold/*.json, панель и воркер используют
флаг gold при отборе в портфель.
"""
import numpy as np

from bt import engine


def _num_keys(params):
    return [k for k, v in params.items() if isinstance(v, (int, float))
            and not isinstance(v, bool)]


def _neighbors(params):
    out = []
    keys = _num_keys(params)
    for k in keys:
        v = params[k]
        for mult in (0.8, 1.25):
            nv = v * mult
            if isinstance(v, int):
                nv = max(1, int(round(nv)))
            if nv == v:
                continue
            np_ = dict(params)
            np_[k] = nv
            out.append(np_)
    return out


def gold_check(df, fn, params, fee, slip, tf_min, train_days, test_days,
               mc_iters=200, seed=7):
    tf_min = int(tf_min)
    bars_per_year = 365.0 * 24 * 60 / tf_min
    train_bars = int(train_days * 24 * 60 / tf_min)
    test_bars = int(test_days * 24 * 60 / tf_min)
    if len(df) < train_bars + 2 * test_bars:
        return {"gold": False, "reason": "not enough data"}

    sigs = {"base": fn(df, **params)}
    wf = engine.walk_forward(df, sigs, train_bars, test_bars, fee, slip, tf_min)
    if "error" in wf:
        return {"gold": False, "reason": wf["error"]}
    base_sharpe = wf["oos"].get("sharpe", 0.0)
    base_pf = wf["oos"].get("pf", 0.0)

    # 1) соседние параметры
    nb_sharpes = []
    for np_ in _neighbors(params)[:8]:
        try:
            s2 = {"n": fn(df, **np_)}
            wf2 = engine.walk_forward(df, s2, train_bars, test_bars, fee, slip, tf_min)
            if "error" not in wf2:
                nb_sharpes.append(wf2["oos"].get("sharpe", 0.0))
        except Exception:
            continue
    nb_median = float(np.median(nb_sharpes)) if nb_sharpes else 0.0

    # 2) подпериоды: пересчитываем OOS по половинам
    a, b = train_bars, train_bars + test_bars
    parts = []
    while a + test_bars <= len(df):
        net, _ = engine.position_returns(df.iloc[a:b], sigs["base"].iloc[a:b], fee, slip)
        parts.append(net)
        a += test_bars
        b += test_bars
    if len(parts) < 2:
        return {"gold": False, "reason": "few folds"}
    half = len(parts) // 2
    h1 = engine.metrics(np.concatenate(parts[:half]), bars_per_year)
    h2 = engine.metrics(np.concatenate(parts[half:]), bars_per_year)

    # 3) Monte Carlo бутстрап по фолдам
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(mc_iters):
        pick = rng.integers(0, len(parts), len(parts))
        totals.append(float(np.concatenate([parts[i] for i in pick]).sum()))
    p5_total = float(np.percentile(totals, 5))

    gold = (base_sharpe > 1.0 and base_pf > 1.05
            and nb_median > 0.5
            and h1.get("sharpe", 0) > 0 and h2.get("sharpe", 0) > 0
            and p5_total > -0.10)
    return {
        "gold": bool(gold),
        "base_sharpe": round(base_sharpe, 2),
        "base_pf": round(base_pf, 3),
        "neighbor_median_sharpe": round(nb_median, 2),
        "n_neighbors": len(nb_sharpes),
        "sharpe_half1": round(h1.get("sharpe", 0), 2),
        "sharpe_half2": round(h2.get("sharpe", 0), 2),
        "mc_p5_total": round(p5_total, 4),
        "folds": len(parts),
        "params": params,
    }
