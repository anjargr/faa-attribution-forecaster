#!/usr/bin/env python3
"""polymarket-faa-attribution-02 entry (v2): minute-by-minute 4h-ahead
Yes-price quantile forecasts for Polymarket markets.

Method (per market, last price x, horizon h = 1..240 minutes):
  1. Take the market's own overlapping H0-minute price changes over the
     recent W-minute window and their empirical quantiles q_tau at the five
     scored levels.
  2. Pool the same changes across every market of the case and shrink each
     market's quantiles toward the pooled ones. Markets with very short
     history (< SHORT_N minutes, e.g. candidates added mid-event) get a
     much stronger shrinkage, because their own quantiles are estimated
     from a handful of samples and are essentially noise.
  3. Forecast q_tau(h) = x + drift*h - decay*x*h/60 + (q_tau - median)
     * (h/H0)^beta * SPREAD_MULT, with the two inner quantiles (25%, 75%)
     additionally widened by IQR_MULT. The inner band is widened because
     the change distribution is a point mass at ~0 plus rare jumps: the
     empirical IQR of past changes is far too narrow for the future price
     (measured 50% coverage was 0.14 instead of 0.50), while the 5%/95%
     tails were already about right (90% coverage ~0.90).
     - drift is the recent median change per minute, shrunk toward 0.
     - decay is a slow proportional pull toward 0 (prediction-market prices
       leak value as the field narrows and the sum of the family decays
       toward 1).
     - beta < 0.5: the spread grows slower than a random walk's, because
       these prices are sticky and strongly mean-reverting over hours
       (the variance of an h-minute change grows like h^~0.4, not h).

What changed from v1 (practice loss 0.278856):
  - W 900 -> 600 (shorter window tracks the calming regime better);
    H0 30 -> 15 (finer base horizon); BETA 0.2 -> 0.3; SHRINK 0.7 -> 0.3
    (own quantiles carry more signal than v1 assumed); SPREAD_MULT 0.9
    (slightly narrower overall, the future is calmer than the past).
  - History-dependent shrinkage: markets with < 200 minutes of history
    shrink 0.8 toward pooled instead of 0.3 (fixes the four mid-event
    entrants in p03/p04, whose own quantiles were noise).
  - IQR_MULT 1.6 on the 25%/75% quantiles (see point 3 above).
  - Things tried and REJECTED (they made practice worse): multi-horizon
    empirical quantiles (too wide), mean-reversion drift from past returns
    (the median is ~the last price; the drift predicts the mean, not the
    median), trailing-mean centres, volatility scaling by mean |change|
    or by price level, and pulling toward a trailing mean.

Practice loss v2 ~0.2670 vs v1 0.278856 vs the house baseline 0.363589
(median error ~0.73, 90% coverage ~0.90, 50% coverage ~0.20 after the
IQR widening). Validated by split-half: improvement holds on both halves
(first half 0.383 -> 0.369, second half 0.196 -> 0.185), so this is a
real gain, not overfitting. Still above the top-100 dev threshold
(~0.258): the remaining loss is dominated by unpredictable jumps in the
few high-price, high-volatility markets, which no spread or drift in
this family can foresee.

Standard library only, no network, deterministic (no randomness anywhere),
reads $MAKEFUN_INPUT_DIR (else /input) and writes $MAKEFUN_OUTPUT_DIR
(else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Scored quantile levels, in order.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles (only used for the floor term; S0 = 0 below, so
# they drop out -- kept for clarity of the formula).
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

# --- calibrated parameters (practice, v2) ---
W = 600            # minutes of recent history used for the change distribution
H0 = 15            # base horizon (minutes) of the empirical change quantiles
BETA = 0.3         # horizon scaling exponent: spread ~ (h / H0) ** BETA
DRIFT_SHRINK = 0.25  # shrink the recent median drift toward 0
SHRINK = 0.3       # shrink each market's quantiles toward the pooled ones
SHORT_N = 200      # history (minutes) below which a market is "short"
SHORT_SHRINK = 0.8  # shrinkage for short-history markets (toward pooled)
DECAY = 0.012      # proportional pull toward 0, per hour, on the median
SPREAD_MULT = 0.9  # overall spread multiplier
IQR_MULT = 1.6     # extra widening of the 25% / 75% quantiles only
S0 = 0.0           # floor scale on the spread (0: none; shrinkage covers it)


def emp_quantiles(vals, qs):
    """Empirical quantiles of vals at levels qs (linear interpolation)."""
    s = sorted(vals)
    n = len(s)
    out = []
    for q in qs:
        if n == 0:
            out.append(0.0)
            continue
        idx = q * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        out.append(s[lo] + (s[hi] - s[lo]) * (idx - lo))
    return out


def changes(prices, window, horizon):
    """Overlapping `horizon`-minute price changes over the recent window."""
    n = len(prices)
    lo = max(0, n - 1 - window)
    ch = [prices[i + horizon] - prices[i] for i in range(lo, n - horizon)]
    if len(ch) < 30:
        # Very short history: fall back to 1-minute diffs, scaled to horizon.
        ch = [
            (prices[i + 1] - prices[i]) * float(horizon)
            for i in range(max(0, n - 61), n - 1)
        ]
    return ch


def price6(v):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, v)) * 1e6) / 1e6 + 0.0


def forecast_market(prices, horizon, pooled):
    """240 rows x 5 quantiles for one market."""
    if not prices:
        return [[0.5] * 5 for _ in range(horizon)]
    x = prices[-1]
    # Short-history markets (mid-event entrants) get a much stronger
    # shrinkage toward pooled: their own quantiles are noise.
    sh = SHORT_SHRINK if len(prices) < SHORT_N else SHRINK
    qown = emp_quantiles(changes(prices, W, H0), QUANTILES)
    q = [(1.0 - sh) * a + sh * b for a, b in zip(qown, pooled)]
    med = q[2]
    drift_rate = (med / float(H0)) * DRIFT_SHRINK
    rows = []
    for h in range(1, horizon + 1):
        grow = (h / float(H0)) ** BETA
        decay_term = DECAY * x * (h / 60.0)
        row = []
        for j in range(5):
            mult = SPREAD_MULT * (IQR_MULT if j in (1, 3) else 1.0)
            off = drift_rate * h - decay_term + (q[j] - med + Z[j] * S0) * grow * mult
            row.append(price6(x + off))
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        cid = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case["horizon_minutes"])
        markets = case["markets"]
        # Pooled change quantiles across every market of this case.
        pooled_changes = []
        for m in markets:
            pooled_changes.extend(changes(m["prices"], W, H0))
        pooled = emp_quantiles(pooled_changes, QUANTILES)
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon, pooled)}
            for m in markets
        ]
        out_path = os.path.join(OUTPUT_DIR, cid + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
