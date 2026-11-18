"""gta6-twitch-ccu-forecast-01 entry: forecast GTA VI's launch-day Twitch viewers, hour by hour.

Model (pure Python 3.13 standard library, deterministic, no network):
  The case file carries past big launches ("comparables"), each with its full
  release-day hourly curve. For every comparable we form its log-ratio curve
  L_c(h) = log v_c(h) - log v_c(1): the shape of a launch day relative to its
  own first hour. The forecast for the target launch is the distribution of
  log p1 + L_c(h) over the comparables, where p1 is the target's hour-1 print,
  with each comparable weighted by an exponential kernel on the distance
  between its log hour-1 level and the target's, exp(-|log v_c(1) - log p1|).
  Launches that opened at a similar scale share the target's unlock geometry
  (a high hour 1 means most regions are already unlocked and the day is flat;
  a low hour 1 means the big regional unlocks are still ahead and the day
  ramps), so scale neighbours carry the shape. A weighted linear regression
  of L_c(h) on log v_c(1) adds a small shift for targets whose scale sits
  outside the pool's centre. The five scored quantiles of that distribution
  are the forecast. If a case carries too few usable comparables, the program
  falls back to persistence with a random-walk spread (the house baseline).

  Backtest (leave-one-out over the 37 usable comparables of practice case
  p00, scored exactly as the contest scores, mean pinball loss on the log
  scale): this model 0.200, persistence baseline 0.465.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0),
standard library only. Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output. For every <id>.in in the
input directory it writes <id>.out: {"hours": [{"hour": h, "q": [q05, q25,
q50, q75, q95]} for h in 2..24]}, quantiles of that hour's viewer count.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (starter's pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Spread of an hour's move for the persistence fallback, log units per sqrt hour.
SIGMA = 0.15
# Width of the exponential kernel on log hour-1 distance (backtest-tuned).
KERNEL_TAU = 1.0
# The linear-shift correction is clamped to this many log units either way.
SHIFT_CLAMP = 0.7
# A comparable's day curve is usable with at least this many of its 24 hours present.
MIN_DAY_HOURS = 16
# Below this many usable comparables the program falls back to persistence.
MIN_POOL = 5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def day_curve(comp: dict) -> dict | None:
    """The comparable's release-day curve {hour: viewers}, gaps log-interpolated.

    Returns None when hour 1 is missing or non-positive, or too few of the
    day's hours are present to trust the shape.
    """
    start = parse_ts(comp["day_start"])
    pts: dict[int, list[float]] = {}
    for point in comp["hourly"]:
        t = parse_ts(point["from"])
        if t < start:
            continue
        hour = int((t - start) // 3600) + 1
        if 1 <= hour <= 24:
            pts.setdefault(hour, []).append(float(point["viewers"]))
    curve = {h: sum(v) / len(v) for h, v in pts.items()}
    if 1 not in curve or curve[1] <= 0 or len(curve) < MIN_DAY_HOURS:
        return None
    known = sorted(curve)
    filled: dict[int, float] = {}
    for h in range(1, 25):
        if h in curve:
            filled[h] = curve[h]
            continue
        lo = max((k for k in known if k < h), default=None)
        hi = min((k for k in known if k > h), default=None)
        if lo is None:
            filled[h] = curve[hi]
        elif hi is None:
            filled[h] = curve[lo]
        else:
            frac = (h - lo) / (hi - lo)
            filled[h] = math.exp(math.log(curve[lo]) * (1 - frac) + math.log(curve[hi]) * frac)
    return filled


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """The tau-quantile of values with weights: first value whose cumulative weight reaches tau."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, w in ordered:
        acc += w
        if acc / total >= tau:
            return value
    return ordered[-1][0]


def hour_log_quantiles(log_p1: float, pool: list[tuple[float, dict]], hour: int,
                       levels: list[float]) -> list[float]:
    """Forecast log-quantiles for one hour from the weighted comparable shapes."""
    xs = [log_h1 for log_h1, _ in pool]
    weights = [math.exp(-abs(x - log_p1) / KERNEL_TAU) for x in xs]
    ratios = [curve[hour] - curve[1] for _, curve in pool]  # curves already in log space here
    pairs = [(log_p1 + r, w) for r, w in zip(ratios, weights)]
    quantiles = [weighted_quantile(pairs, tau) for tau in levels]
    # Weighted linear shift: how the day's shape tilts with opening scale.
    sw = sum(weights)
    mx = sum(w * x for w, x in zip(weights, xs)) / sw
    my = sum(w * r for w, r in zip(weights, ratios)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(weights, xs))
    sxy = sum(w * (x - mx) * (r - my) for w, x, r in zip(weights, xs, ratios))
    slope = sxy / sxx if sxx > 0 else 0.0
    shift = max(-SHIFT_CLAMP, min(SHIFT_CLAMP, slope * (log_p1 - mx)))
    return [q + shift for q in quantiles]


def persistence_log_quantiles(log_p1: float, hour: int, levels: list[float]) -> list[float]:
    out = []
    for tau in levels:
        # Invert the standard normal CDF by bisection for arbitrary levels.
        lo, hi = -6.0, 6.0
        for _ in range(60):
            mid = (lo + hi) / 2
            cdf = 0.5 * (1 + math.erf(mid / math.sqrt(2)))
            if cdf < tau:
                lo = mid
            else:
                hi = mid
        out.append(log_p1 + (lo + hi) / 2 * SIGMA * math.sqrt(hour - 1))
    return out


def forecast_case(case: dict) -> dict:
    levels = case.get("levels") or LEVELS
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    pool: list[tuple[float, dict]] = []
    for comp in case.get("comparables") or []:
        curve = day_curve(comp)
        if curve is None:
            continue
        log_curve = {h: math.log(max(1.0, v)) for h, v in curve.items()}
        pool.append((log_curve[1], log_curve))

    rows = []
    # Sort by hour: the output lists hours 2..24 in order, whatever order the case names them in.
    for row in sorted(case["forecast"], key=lambda r: r["hour"]):
        hour = row["hour"]
        if len(pool) >= MIN_POOL:
            log_qs = hour_log_quantiles(log_p1, pool, hour, levels)
        else:
            log_qs = persistence_log_quantiles(log_p1, hour, levels)
        qs = []
        prev = 0.0
        for lq in log_qs:
            value = math.exp(max(math.log(1.0), min(math.log(1e9), lq)))
            value = max(value, prev)  # quantiles never decrease across levels
            prev = value
            qs.append(round2(value))
        rows.append({"hour": hour, "q": qs})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
