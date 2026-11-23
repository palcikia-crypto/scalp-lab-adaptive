# -*- coding: utf-8 -*-
# Живая верификация сделок на реальных данных Binance (REST).
# Для каждой live/shadow стратегии бумажного портфеля и открытых позиций:
#   1) доступность REST API и задержка каждого запроса (мс);
#   2) свежесть последней ЗАКРЫТОЙ свечи по таймфрейму стратегии;
#   3) спред bid/ask по реальному стакану (bps);
#   4) проскальзывание рыночного ордера на NOTIONAL_USDT по стакану (bps);
#   5) дрейф цены открытой бумажной позиции против реального рынка (bps).
# Цель: до реальной торговли знать, что данные и исполнение идут без сюрпризов.
# Результат: results/verification.json
import json
import time

import requests

BINANCE = "https://api.binance.com"
DATA = "https://data-api.binance.vision"
NOTIONAL_USDT = 500.0
MAX_ROWS = 12
TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
WARN_SPREAD_BPS = 10.0
WARN_SLIP_BPS = 15.0
WARN_CANDLE_AGE_MULT = 3.0

BOOK_CACHE = {}


def _get(url, params=None, timeout=8):
    t0 = time.time()
    r = requests.get(url, params=params or {}, timeout=timeout)
    dt = round((time.time() - t0) * 1000.0)
    r.raise_for_status()
    return r.json(), dt


def candle_age_min(symbol, tf):
    k, lat = _get(DATA + "/api/v3/klines", {"symbol": symbol, "interval": tf, "limit": 3})
    close_time = int(k[-2][6]) / 1000.0  # последняя закрытая свеча
    age = (time.time() - close_time) / 60.0
    return round(max(age, 0.0), 1), lat


def book_stats(symbol):
    if symbol not in BOOK_CACHE:
        d, lat = _get(BINANCE + "/api/v3/depth", {"symbol": symbol, "limit": 50})
        bids = [(float(p), float(q)) for p, q in d["bids"]]
        asks = [(float(p), float(q)) for p, q in d["asks"]]
        bb, ba = bids[0][0], asks[0][0]
        mid = (bb + ba) / 2.0
        spread = round((ba - bb) / mid * 1e4, 2)
        # исполнение покупки на NOTIONAL_USDT по рынку: идём по стакану
        quote_spent = 0.0
        base_bought = 0.0
        for p, q in asks:
            take = min(NOTIONAL_USDT - quote_spent, p * q)
            if take <= 0:
                break
            base_bought += take / p
            quote_spent += take
            if quote_spent >= NOTIONAL_USDT:
                break
        slip = 0.0
        if base_bought > 0:
            slip = round((quote_spent / base_bought - ba) / ba * 1e4, 2)
        BOOK_CACHE[symbol] = {"spread_bps": spread, "slip_bps": slip, "mid": mid, "lat": lat}
    return BOOK_CACHE[symbol]


def main():
    rows, lats = [], []
    api_ok = True
    try:
        _get(BINANCE + "/api/v3/time")
    except Exception as e:
        api_ok = False
        rows.append({"key": "api", "strategy": "api", "symbol": "-", "tf": "-", "mode": "system",
                     "verdict": "fail", "note": "Binance API недоступен: " + str(e)[:160],
                     "latency_ms": None, "candle_age_min": None, "spread_bps": None,
                     "slip_bps": None, "drift_bps": None})
    if api_ok:
        try:
            with open("results/paper.json") as fh:
                p = json.load(fh)
        except Exception:
            p = {}
        targets = {}
        for t in (p.get("tops") or []):
            mode = t.get("mode") or "live"
            if mode not in ("live", "shadow"):
                continue
            key = t.get("key") or (str(t.get("strategy")) + "-" + str(t.get("symbol")) + "-" + str(t.get("tf")))
            targets[key] = {"strategy": t.get("strategy"), "symbol": t.get("symbol"),
                            "tf": t.get("tf"), "mode": mode}
        for k, q in (p.get("positions") or {}).items():
            targets[k] = {"strategy": q.get("strategy"), "symbol": q.get("symbol"),
                          "tf": q.get("tf"), "mode": "position", "last_px": q.get("last_px")}
        for key, t in list(targets.items())[:MAX_ROWS]:
            sym, tf = t.get("symbol"), t.get("tf")
            if not sym or not tf or tf not in TF_MIN:
                continue
            row = {"key": key, "strategy": t.get("strategy"), "symbol": sym, "tf": tf,
                   "mode": t.get("mode"), "latency_ms": None, "candle_age_min": None,
                   "spread_bps": None, "slip_bps": None, "drift_bps": None,
                   "verdict": "pass", "note": ""}
            try:
                age, lat = candle_age_min(sym, tf)
                row["candle_age_min"] = age
                b = book_stats(sym)
                row["latency_ms"] = max(lat, b["lat"])
                row["spread_bps"] = b["spread_bps"]
                row["slip_bps"] = b["slip_bps"]
                lats.append(row["latency_ms"])
                if t.get("mode") == "position" and t.get("last_px"):
                    row["drift_bps"] = round(abs(b["mid"] / float(t["last_px"]) - 1) * 1e4, 2)
                notes = []
                if row["candle_age_min"] > WARN_CANDLE_AGE_MULT * TF_MIN[tf]:
                    row["verdict"] = "warn"; notes.append("свеча старше %s TF" % TF_MIN[tf])
                if row["spread_bps"] > WARN_SPREAD_BPS:
                    row["verdict"] = "warn"; notes.append("широкий спред")
                if row["slip_bps"] > WARN_SLIP_BPS:
                    row["verdict"] = "warn"; notes.append("проскальзывание выше нормы")
                if row["drift_bps"] is not None and row["drift_bps"] > 500:
                    row["verdict"] = "warn"; notes.append("позиция далеко от рынка")
                row["note"] = "; ".join(notes)
            except Exception as e:
                row["verdict"] = "fail"
                row["note"] = str(e)[:160]
            rows.append(row)
    summ = {"pass": 0, "warn": 0, "fail": 0}
    for r in rows:
        summ[r["verdict"]] += 1
    spreads = [r["spread_bps"] for r in rows if r["spread_bps"] is not None]
    slips = [r["slip_bps"] for r in rows if r["slip_bps"] is not None]
    out = {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "api_ok": api_ok,
           "api_latency_ms": int(sorted(lats)[len(lats) // 2]) if lats else None,
           "worst_spread_bps": max(spreads) if spreads else None,
           "worst_slip_bps": max(slips) if slips else None,
           "summary": summ, "rows": rows}
    with open("results/verification.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("live_check: %d проверено, %s" % (len(rows), summ), flush=True)


if __name__ == "__main__":
    main()
