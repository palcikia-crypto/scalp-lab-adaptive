import numpy as np
import pandas as pd


def ema_cross(df, fast, slow):
    f = df["close"].ewm(span=fast, adjust=False).mean()
    s = df["close"].ewm(span=slow, adjust=False).mean()
    return (f > s).astype(float) * 2 - 1


def bollinger_reversion(df, n, k):
    m = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    sig = pd.Series(np.nan, index=df.index)
    sig[df["close"] < m - k * sd] = 1.0
    sig[df["close"] > m + k * sd] = -1.0
    sig[(df["close"] - m).abs() < 0.1 * sd] = 0.0
    return sig.ffill().fillna(0.0)


def donchian(df, n):
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    mid = (hi + lo) / 2
    sig = pd.Series(np.nan, index=df.index)
    sig[df["close"] < mid] = 0.0
    sig[df["close"] > mid] = 0.0
    sig[df["close"] > hi] = 1.0
    sig[df["close"] < lo] = -1.0
    return sig.ffill().fillna(0.0)


def volume_surge(df, z, h):
    v = df["volume"]
    m = v.rolling(200).mean()
    s = v.rolling(200).std()
    ev = v > m + z * s
    dirn = np.sign(df["close"] - df["open"])
    sig = pd.Series(np.nan, index=df.index)
    sig = dirn.where(ev & (dirn != 0))
    return sig.ffill(limit=h).fillna(0.0)


def vwap_reversion(df, n, k):
    pv = df["quote_vol"].rolling(n).sum()
    v = df["volume"].rolling(n).sum()
    vwap = pv / v
    dev = df["close"] - vwap
    zscore = dev / dev.rolling(200).std()
    sig = pd.Series(np.nan, index=df.index)
    sig[zscore > k] = -1.0
    sig[zscore < -k] = 1.0
    sig[zscore.abs() < 0.1] = 0.0
    return sig.ffill().fillna(0.0)


# ===== fee-aware стратегии: редкие сделки, длинное удержание =====


def streak_reversion(df, n_streak, h):
    # паттерн down3/up3: серия одноцветных свечей -> возврат к среднему
    up = (df["close"] - df["open"]) > 0
    dn = ~up
    up_run = up.groupby((~up).cumsum()).cumsum()
    dn_run = dn.groupby((~dn).cumsum()).cumsum()
    sig = pd.Series(np.nan, index=df.index)
    sig[dn_run >= n_streak] = 1.0
    sig[up_run >= n_streak] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def vol_drift(df, z, h):
    # паттерн vol_z>3: после всплеска объёма дрейф вверх до 60 баров
    v = df["volume"]
    m = v.rolling(200).mean()
    s = v.rolling(200).std()
    ev = v > m + z * s
    sig = pd.Series(np.nan, index=df.index)
    sig[ev] = 1.0
    return sig.ffill(limit=h).fillna(0.0)


def wick_reversion(df, w, h):
    # большие тени -> откат; вход лимиткой, maker fee
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    up_wick = df["high"] - body_hi
    lo_wick = body_lo - df["low"]
    sig = pd.Series(np.nan, index=df.index)
    sig[up_wick > w * rng] = -1.0
    sig[lo_wick > w * rng] = 1.0
    return sig.ffill(limit=h).fillna(0.0)


def donchian_trend(df, n):
    # тренд с большим окном: в разы меньше сделок -> комиссии меньше грузят
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    mid = (hi + lo) / 2
    sig = pd.Series(np.nan, index=df.index)
    sig[df["close"] < mid] = 0.0
    sig[df["close"] > mid] = 0.0
    sig[df["close"] > hi] = 1.0
    sig[df["close"] < lo] = -1.0
    return sig.ffill().fillna(0.0)


def bb_maker(df, n, k):
    # bollinger с длинным окном: меньше сигналов, каждый крупнее комиссий;
    # исполнение лимитками у границ канала -> maker fee
    return bollinger_reversion(df, n, k)


# ===== комбинированные: сигнал + подтверждение объёмом (vol_z, FDR-выживший) =====

def _vol_z(df):
    v = df["volume"]
    m = v.rolling(200).mean()
    s = v.rolling(200).std().replace(0.0, np.nan)
    return (v - m) / s


def streak_vol(df, n_streak, vz, h):
    # серия одноцветных свечей + всплеск объёма -> реверс (объединение 2 FDR-паттернов)
    up = (df["close"] - df["open"]) > 0
    dn = ~up
    up_run = up.groupby((~up).cumsum()).cumsum()
    dn_run = dn.groupby((~dn).cumsum()).cumsum()
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[(dn_run >= n_streak) & (z > vz)] = 1.0
    sig[(up_run >= n_streak) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def bb_vol(df, n, k, vz, h):
    # выход за границу Bollinger, подтверждённый объёмом: меньше ложных входов
    m = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[(df["close"] < m - k * sd) & (z > vz)] = 1.0
    sig[(df["close"] > m + k * sd) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def wick_vol(df, w, vz, h):
    # большая тень + всплеск объёма -> откат
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[((body_lo - df["low"]) > w * rng) & (z > vz)] = 1.0
    sig[((df["high"] - body_hi) > w * rng) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def spike_revert(df, pct, vz, h):
    # памп/дамп: большое движение свечи + всплеск объёма -> вход ПРОТИВ движения
    # опирается на FDR-закономерность: vol_z-всплески откатываются
    r = df["close"] / df["open"] - 1.0
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[(r < -pct) & (z > vz)] = 1.0
    sig[(r > pct) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def spike_follow(df, pct, vz, h):
    # инерция пампа/дампа: большое движение + объём -> вход ПО движению
    r = df["close"] / df["open"] - 1.0
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[(r > pct) & (z > vz)] = 1.0
    sig[(r < -pct) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def crash_revert(df, k, thr, vz, h):
    # каскадный памп/дамп: суммарное движение за k баров + объём -> реверс
    r = df["close"] / df["open"] - 1.0
    cum = r.rolling(k).sum()
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[(cum < -thr) & (z > vz)] = 1.0
    sig[(cum > thr) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def spike_bb(df, pct, k, vz, h):
    # спайк + выход за Bollinger(50) одновременно -> усиленный реверс
    r = df["close"] / df["open"] - 1.0
    m = df["close"].rolling(50).mean()
    sd = df["close"].rolling(50).std()
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[(r < -pct) & (z > vz) & (df["close"] < m - k * sd)] = 1.0
    sig[(r > pct) & (z > vz) & (df["close"] > m + k * sd)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def vol_burst_revert(df, vz, h):
    # главный FDR-паттерн (vol_z>5, t=3.0): взрыв объёма -> откат против свечи
    z = _vol_z(df)
    up = (df["close"] - df["open"]) > 0
    sig = pd.Series(np.nan, index=df.index)
    sig[(z > vz) & (~up)] = 1.0
    sig[(z > vz) & up] = -1.0
    return sig.ffill(limit=h).fillna(0.0)

# ===== деривативные данные: funding rate, открытый интерес, long/short =====


def funding_revert(df, z, h):
    # экстремальный funding: толпа платит за позицию -> контрарианский вход
    f = df["funding"]
    fz = (f - f.rolling(90).mean()) / f.rolling(90).std().replace(0.0, np.nan)
    sig = pd.Series(np.nan, index=df.index)
    sig[fz > z] = -1.0
    sig[fz < -z] = 1.0
    return sig.ffill(limit=h).fillna(0.0)


def oi_exhaust(df, k, thr, oi_thr, h):
    # памп/дамп без поддержки открытого интереса -> движение выдыхается, откат
    r = df["close"].pct_change(k)
    oic = df["oi"].pct_change(k)
    sig = pd.Series(np.nan, index=df.index)
    sig[(r > thr) & (oic < -oi_thr)] = -1.0
    sig[(r < -thr) & (oic < -oi_thr)] = 1.0
    return sig.ffill(limit=h).fillna(0.0)


def ls_contrarian(df, z, h):
    # taker long/short ratio: толпа агрегированно в лонгах -> контрарианский шорт
    ls = df["ls_ratio"]
    lz = (ls - ls.rolling(200).mean()) / ls.rolling(200).std().replace(0.0, np.nan)
    sig = pd.Series(np.nan, index=df.index)
    sig[lz > z] = -1.0
    sig[lz < -z] = 1.0
    return sig.ffill(limit=h).fillna(0.0)


# ===== мета-стратегии: сигнал + фильтр режима рынка =====


def bb_regime(df, n, k, vz_max):
    # Bollinger-реверс, но только когда рынок спокоен (vol_z < vz_max)
    m = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    z = _vol_z(df)
    calm = z < vz_max
    sig = pd.Series(np.nan, index=df.index)
    sig[(df["close"] < m - k * sd) & calm] = 1.0
    sig[(df["close"] > m + k * sd) & calm] = -1.0
    sig[(df["close"] - m).abs() < 0.1 * sd] = 0.0
    return sig.ffill().fillna(0.0)


def wick_regime(df, w, vz_max, h):
    # теневой реверс только в спокойном рынке
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    z = _vol_z(df)
    calm = z < vz_max
    sig = pd.Series(np.nan, index=df.index)
    sig[((body_lo - df["low"]) > w * rng) & calm] = 1.0
    sig[((df["high"] - body_hi) > w * rng) & calm] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


# ===== волна 4: быстрые скальп-стратегии (частые сделки, короткое удержание) =====


def bb_fast(df, n, k):
    # короткий Bollinger: возврат к среднему на малом окне -> много сигналов
    return bollinger_reversion(df, n, k)


def ema_snap(df, n, k):
    # резкое отклонение цены от EMA -> отскок назад к EMA
    e = df["close"].ewm(span=n, adjust=False).mean()
    dev = df["close"] - e
    sd = dev.rolling(200).std().replace(0.0, np.nan)
    z = dev / sd
    sig = pd.Series(np.nan, index=df.index)
    sig[z > k] = -1.0
    sig[z < -k] = 1.0
    sig[z.abs() < 0.3] = 0.0
    return sig.ffill().fillna(0.0)


def flow_fade(df, k, h):
    # order-flow: перекос агрессивных покупок/продаж (taker flow) -> контрарианский откат
    q = df["quote_vol"].replace(0.0, np.nan)
    imb = df["taker_quote"] / q
    m = imb.rolling(200).mean()
    s = imb.rolling(200).std().replace(0.0, np.nan)
    z = (imb - m) / s
    sig = pd.Series(np.nan, index=df.index)
    sig[z > k] = -1.0
    sig[z < -k] = 1.0
    return sig.ffill(limit=h).fillna(0.0)


def squeeze_pop(df, n, m, vz, h):
    # сжатие диапазона -> прорыв с объёмом -> короткий моментум по движению
    rng = (df["high"] - df["low"]).rolling(n).mean()
    avg = rng.rolling(200).mean().replace(0.0, np.nan)
    sq = (rng < m * avg).fillna(False)
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[sq.shift(1).fillna(False) & (df["close"] > hi) & (z > vz)] = 1.0
    sig[sq.shift(1).fillna(False) & (df["close"] < lo) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def qr_revert(df, n, q):
    # цена в крайнем квантиле окна -> возврат к медиане
    qh = df["close"].rolling(n).quantile(1.0 - q)
    ql = df["close"].rolling(n).quantile(q)
    med = df["close"].rolling(n).median()
    sd = df["close"].rolling(n).std().replace(0.0, np.nan)
    sig = pd.Series(np.nan, index=df.index)
    sig[df["close"] > qh] = -1.0
    sig[df["close"] < ql] = 1.0
    sig[(df["close"] - med).abs() < 0.1 * sd] = 0.0
    return sig.ffill().fillna(0.0)


def level_fade(df, n, w, h):
    # касание границы диапазона без пробоя -> отбой от уровня
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    sig = pd.Series(np.nan, index=df.index)
    sig[(df["high"] >= hi * (1.0 - w)) & (df["close"] < hi)] = -1.0
    sig[(df["low"] <= lo * (1.0 + w)) & (df["close"] > lo)] = 1.0
    return sig.ffill(limit=h).fillna(0.0)


def streak_bb(df, n_streak, k, h):
    # серия одноцветных свечей + выход за короткий Bollinger -> усиленный реверс
    up = (df["close"] - df["open"]) > 0
    dn = ~up
    up_run = up.groupby((~up).cumsum()).cumsum()
    dn_run = dn.groupby((~dn).cumsum()).cumsum()
    m = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    sig = pd.Series(np.nan, index=df.index)
    sig[(dn_run >= n_streak) & (df["close"] < m - k * sd)] = 1.0
    sig[(up_run >= n_streak) & (df["close"] > m + k * sd)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def vol_pop(df, vz, h):
    # объёмный импульс после затишья -> короткий моментум по направлению свечи
    z = _vol_z(df)
    calm_prev = z.shift(1) < 1.0
    up = (df["close"] - df["open"]) > 0
    sig = pd.Series(np.nan, index=df.index)
    sig[calm_prev & (z > vz) & up] = 1.0
    sig[calm_prev & (z > vz) & (~up)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def grid(**kv):
    res = [{}]
    for key in list(kv):
        new = []
        for d in res:
            for v in kv[key]:
                nd = dict(d)
                nd[key] = v
                new.append(nd)
        res = new
    return res


STRATEGIES = {
    "ema_cross": (ema_cross, grid(fast=[5, 9, 12], slow=[21, 30, 50])),
    "bollinger_reversion": (bollinger_reversion, grid(n=[20, 50], k=[2.0, 2.5, 3.0])),
    "donchian": (donchian, grid(n=[20, 40, 60])),
    "volume_surge": (volume_surge, grid(z=[2.5, 3.0, 4.0], h=[3, 5, 10])),
    "vwap_reversion": (vwap_reversion, grid(n=[30, 60], k=[1.5, 2.0, 2.5])),
    # fee-aware волна
    "streak_reversion": (streak_reversion, grid(n_streak=[3, 4], h=[15, 30, 60])),
    "vol_drift": (vol_drift, grid(z=[3.0, 4.0], h=[30, 60, 120])),
    "wick_reversion": (wick_reversion, grid(w=[0.4, 0.5], h=[15, 30, 60])),
    "donchian_trend": (donchian_trend, grid(n=[96, 192, 384])),
    "bb_maker": (bb_maker, grid(n=[50, 100, 200], k=[2.0, 2.5])),
    # комбинированные: сигнал + объёмное подтверждение
    "streak_vol": (streak_vol, grid(n_streak=[3, 4], vz=[2.0, 3.0], h=[15, 30, 60])),
    "bb_vol": (bb_vol, grid(n=[50, 100], k=[2.0, 2.5], vz=[2.0, 3.0], h=[15, 30, 60])),
    "wick_vol": (wick_vol, grid(w=[0.4, 0.5], vz=[2.0, 3.0], h=[15, 30, 60])),
    # памп/дамп: резкий ход с объёмом
    "spike_revert": (spike_revert, grid(pct=[0.01, 0.02, 0.03, 0.05], vz=[2.0, 3.0], h=[30, 60])),
    "spike_follow": (spike_follow, grid(pct=[0.008, 0.015, 0.03], vz=[2.0, 3.0], h=[15, 30, 60])),
    # памп/дамп: волна 2 — каскады, спайк+канал, взрыв объёма
    "crash_revert": (crash_revert, grid(k=[3, 5], thr=[0.03, 0.05, 0.08], vz=[2.0, 3.0], h=[30, 60])),
    "spike_bb": (spike_bb, grid(pct=[0.01, 0.02], k=[2.0, 2.5], vz=[2.0, 3.0], h=[30, 60])),
    "vol_burst_revert": (vol_burst_revert, grid(vz=[4.0, 5.0, 6.0], h=[60, 120])),
    # волна 3: деривативные данные (funding / OI / long-short) + мета-фильтры режима
    "funding_revert": (funding_revert, grid(z=[1.5, 2.0, 2.5], h=[30, 60])),
    "oi_exhaust": (oi_exhaust, grid(k=[12, 24], thr=[0.02, 0.04], oi_thr=[0.02, 0.05], h=[30, 60])),
    "ls_contrarian": (ls_contrarian, grid(z=[1.5, 2.0, 2.5], h=[30, 60])),
    "bb_regime": (bb_regime, grid(n=[100, 200], k=[2.0, 2.5], vz_max=[1.0, 1.5])),
    "wick_regime": (wick_regime, grid(w=[0.4, 0.5], vz_max=[1.0, 1.5], h=[15, 30])),
    # волна 4: быстрые скальпы
    "bb_fast": (bb_fast, grid(n=[10, 15], k=[2.0, 2.5])),
    "ema_snap": (ema_snap, grid(n=[20, 50], k=[2.0, 2.5])),
    "flow_fade": (flow_fade, grid(k=[2.0, 2.5], h=[15, 30])),
    "squeeze_pop": (squeeze_pop, grid(n=[20, 60], m=[0.7], vz=[2.0], h=[30, 60])),
    "qr_revert": (qr_revert, grid(n=[100, 200], q=[0.05, 0.1])),
    "level_fade": (level_fade, grid(n=[60], w=[0.001, 0.002], h=[15, 30])),
    "streak_bb": (streak_bb, grid(n_streak=[3], k=[1.5, 2.0], h=[30, 60])),
    "vol_pop": (vol_pop, grid(vz=[3.0, 4.0], h=[15, 30])),
}

# режим исполнения: реверсивные входят лимитками (maker),
# трендовые - по рынку (taker)
MAKER_SET = {"streak_reversion", "wick_reversion", "bb_maker",
             "bollinger_reversion", "vwap_reversion",
             "streak_vol", "bb_vol", "wick_vol", "spike_revert", "crash_revert", "spike_bb", "vol_burst_revert",
             "funding_revert", "oi_exhaust", "ls_contrarian", "bb_regime", "wick_regime",
             "bb_fast", "ema_snap", "flow_fade", "qr_revert", "level_fade", "streak_bb"}

# стратегии, которым нужны funding/OI/long-short из архивов (runner обогащает df)
DERIV_SET = {"funding_revert", "oi_exhaust", "ls_contrarian"}
