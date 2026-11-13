import numpy as np
import pandas as pd
from collections import Counter

from bt.stats import metrics


def _funding_per_bar(df, tf_min):
    """Funding-платёж на бар: ставка за интервал (обычно 8 часов)
    масштабируется в долю бара. Лонг платит при положительном funding,
    шорт получает (и наоборот при отрицательном)."""
    if "funding" not in df.columns:
        return None
    fr = pd.to_numeric(df["funding"], errors="coerce").fillna(0.0).values
    return fr * (tf_min / 480.0)


def position_returns(df, pos, fee, slip, fpb=None, maker=None):
    """net-доходность позиции.

    maker = (fee_maker, slip_maker): консервативная модель исполнения.
    Лимитный ордер (вход или выход) засчитывается по maker-комиссии только
    если на баре исполнения цена реально коснулась уровня (close предыдущего
    бара): лонг - low <= ref, шорт - high >= ref. Если касания не было,
    переход считается исполненным тейкером (fee + slip). Это убирает
    завышение maker-результатов из-за неисполнившихся лимитников."""
    ret = df["close"].pct_change().fillna(0.0)
    p = pos.shift(1).fillna(0.0)
    gross = p * ret
    d = p.diff().fillna(0.0)
    cost_rate = np.full(len(df), fee + slip)
    if maker is not None:
        fee_m, slip_m = maker
        close = df["close"].values.astype(float)
        low = df["low"].values.astype(float) if "low" in df.columns else close
        high = df["high"].values.astype(float) if "high" in df.columns else close
        ref = np.roll(close, 1)
        ref[0] = close[0]
        dv = d.values
        touched = np.where(dv > 0, low <= ref,
                  np.where(dv < 0, high >= ref, True))
        cost_rate = np.where(touched, fee_m + slip_m, fee + slip)
    costs = d.abs().values * cost_rate
    if fpb is not None:
        costs = costs + p.values * fpb
    net = gross.values - costs
    n_trades = int((d.abs() > 1e-12).sum())
    return net, n_trades


def walk_forward(df, sigs, train_bars, test_bars, fee, slip, tf_min,
                 fee2=None, slip2=None):
    bars_per_year = 365.0 * 24 * 60 / tf_min
    keys = list(sigs.keys())
    if not keys:
        return {"error": "no valid params"}
    fpb = _funding_per_bar(df, tf_min)
    oos_parts = []
    oos2_parts = []
    chosen = []
    is_sharpes = []
    total_trades = 0
    t0 = train_bars
    while t0 + test_bars <= len(df):
        best_key, best_is = None, -1e9
        for k in keys:
            a, b = t0 - train_bars, t0
            net, _ = position_returns(df.iloc[a:b], sigs[k].iloc[a:b], fee, slip, fpb=fpb)
            m = metrics(net, bars_per_year)
            if m["sharpe"] > best_is:
                best_is, best_key = m["sharpe"], k
        a, b = t0, t0 + test_bars
        net, trades = position_returns(df.iloc[a:b], sigs[best_key].iloc[a:b], fee, slip, fpb=fpb)
        oos_parts.append(net)
        if fee2 is not None:
            net2, _ = position_returns(df.iloc[a:b], sigs[best_key].iloc[a:b],
                                       fee, slip, fpb=fpb, maker=(fee2, slip2))
            oos2_parts.append(net2)
        chosen.append(best_key)
        is_sharpes.append(best_is)
        total_trades += trades
        t0 += test_bars
    if not oos_parts:
        return {"error": "not enough bars"}
    oos_ret = np.concatenate(oos_parts)
    mo = metrics(oos_ret, bars_per_year)
    top = Counter(chosen).most_common(1)[0]
    res = {
        "oos": mo,
        "trades": int(total_trades),
        "is_sharpe_avg": float(np.mean(is_sharpes)),
        "folds": int(len(chosen)),
        "param_stability": float(top[1] / len(chosen)),
        "top_param": top[0],
        "funding_cost": bool(fpb is not None and np.any(fpb != 0.0)),
    }
    if oos2_parts:
        mo2 = metrics(np.concatenate(oos2_parts), bars_per_year)
        res["oos_maker"] = mo2
    return res
