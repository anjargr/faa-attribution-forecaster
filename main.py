#!/usr/bin/env python3
"""polymarket-faa-attribution-02 entry: minute-by-minute 4h-ahead Yes-price
quantile forecasts for Polymarket markets.

Method (per market, last price x, horizon h = 1..240 minutes):
  1. Take the market's own overlapping H0-minute price changes over the
     recent W-minute window and their empirical quantiles q_tau at the five
     scored levels.
  2. Pool the same changes across every market of the case and shrink each
     market's quantiles toward the pooled ones (shrinkage stabilises the
     noisy per-market tails, which is where most of the pinball loss lives).
  3. Forecast q_tau(h) = x + drift*h - decay*x*h/60 + (q_tau - median)
     * (h/H0)^beta, clamped to [0, 1] and rounded to 6 decimals.
     - drift is the recent median change per minute, shrunk toward 0.
     - decay is a slow proportional pull toward 0 (prediction-market prices
       leak value as the field narrows and the sum of the family decays
       toward 1).
     - beta < 0.5: the spread grows slower than a random walk's, because
       these prices are sticky and strongly mean-reverting over hours
       (the variance of an h-minute change grows like h^~0.4, not h).

Calibrated on the six practice cases (NVIDIA event) with a local scorer
built from Polymarket's public price history; the parameters below are
round, flat-optimum values, validated by split-half (tuned on p00-p02,
still clearly beats the house baseline on p03-p05, and vice versa).
Practice loss ~0.279 vs the house baseline's 0.363589 (median error ~0.73,
90% coverage ~0.94). Revised after grading v1 (0.284): a finer grid around
the optimum moved W 720 -> 900, drift shrink 0.5 -> 0.25, shrinkage
0.75 -> 0.7, decay 0.01 -> 0.012, and the gain holds on both halves of the
practice (first half 0.383 vs baseline 0.442, second half 0.197 vs 0.302).

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

# --- calibrated parameters (practice) ---
W = 900            # minutes of recent history used for the change distribution
H0 = 30            # base horizon (minutes) of the empirical change quantiles
BETA = 0.2         # horizon scaling exponent: spread ~ (h / H0) ** BETA
DRIFT_SHRINK = 0.25  # shrink the recent median drift toward 0
SHRINK = 0.7       # shrink each market's quantiles toward the pooled ones
DECAY = 0.012      # proportional pull toward 0, per hour, on the median
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
    qown = emp_quantiles(changes(prices, W, H0), QUANTILES)
    q = [(1.0 - SHRINK) * a + SHRINK * b for a, b in zip(qown, pooled)]
    med = q[2]
    drift_rate = (med / float(H0)) * DRIFT_SHRINK
    rows = []
    for h in range(1, horizon + 1):
        grow = (h / float(H0)) ** BETA
        decay_term = DECAY * x * (h / 60.0)
        row = []
        for j in range(5):
            off = drift_rate * h - decay_term + (q[j] - med + Z[j] * S0) * grow
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
