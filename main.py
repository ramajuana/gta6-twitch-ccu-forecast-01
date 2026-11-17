"""Forecast GTA VI launch-day Twitch viewers, hours 2..24, from hour 1's print.

Model (own design, tuned by leave-one-out scoring on the comparables):
  * Every past launch's release day is reduced to log-ratios
    log(viewers_h / viewers_1) for h = 2..24. Hour 1 is a bad normaliser
    for small categories (a 164-viewer hour 1 can grow 1000x by evening),
    but for big launches it is stable, so launches are weighted by how
    close their own hour-1 print is to this case's, with a Gaussian
    kernel of bandwidth 1.0 in log space. That lets the huge GTA VI
    print borrow mostly from the other huge launches (Elden Ring,
    Cyberpunk, Diablo IV, ...) and almost nothing from tiny categories.
  * Each hour's median log-ratio is the kernel-weighted median across
    comparables; the log-spread is the kernel-weighted interquartile
    range / 1.349, floored at 0.25 so the 90% band stays honest even
    when the neighbours agree. Both the median and the spread are
    smoothed over a 3-hour window, because the true day shape (dip
    overnight, step up at the US unlocks around hours 5-9, evening
    peak around hours 19-21) moves slowly, hour to hour.
  * Quantiles are p1 * exp(median_h + z_tau * sigma_h) with the usual
    normal z-scores. The median is clamped to at least 0.25 * p1 so a
    freak neighbour can never forecast a collapse to near zero.
  * If the case has no usable comparables, or hour 1 is missing, the
    forecast falls back to the flat persistence shape (median = p1,
    spread 0.15 * sqrt(h - 1)). Any exception while forecasting one
    case still writes that fallback instead of crashing.

Standard library only, no network, deterministic: comparables are
processed in name order and every aggregate is a deterministic
weighted quantile, so two runs write byte-identical files.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Kernel bandwidth (log-viewer units) for weighting comparables by
# closeness of their hour-1 print to this case's.
BANDWIDTH = 1.0
# Floor on the log-spread, so the 90% interval never collapses.
SIGMA_FLOOR = 0.25
# Median ratio floor: never forecast below a quarter of hour 1.
MEDIAN_RATIO_FLOOR = 0.25
# A comparable only counts if its release day is this complete.
MIN_DAY_HOURS = 20
# Baseline fallback spread, per square-root hour (house persistence).
BASE_SIGMA = 0.15


def _parse_time(text):
    """Parse an ISO-8601 timestamp (trailing Z allowed) to an aware datetime."""
    if not isinstance(text, str) or not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def _weighted_quantile(pairs, q):
    """Weighted quantile of [(value, weight), ...]; pairs must be sorted by value."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0] if pairs else 0.0
    target = q * total
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= target:
            return value
    return pairs[-1][0]


def _forecast_hours(case):
    """The hours this case asks for (2..24 per the rules), in order."""
    hours = []
    for row in case.get("forecast") or []:
        h = row.get("hour")
        if isinstance(h, int) and 2 <= h <= 24 and h not in hours:
            hours.append(h)
    return sorted(hours) if hours else list(range(2, 25))


def _smooth(values, hours):
    """3-hour moving average over the forecast hours (edges use what's available)."""
    out = {}
    for h in hours:
        near = [values[k] for k in (h - 1, h, h + 1) if k in values]
        out[h] = sum(near) / len(near)
    return out


def _release_day(comp):
    """Map a comparable to {hour: viewers} for its 24 release-day hours.

    Returns None when the comparable has no usable hour-1 print or too
    few release-day hours (e.g. the GTA VI category's own pre-release
    history, which has no release day yet).
    """
    day_start = _parse_time(comp.get("day_start"))
    if day_start is None:
        return None
    by_hour = {}
    for point in comp.get("hourly") or []:
        when = _parse_time(point.get("from"))
        viewers = point.get("viewers")
        if when is None or not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and hour not in by_hour:
            by_hour[hour] = float(viewers)
    if 1 not in by_hour or len(by_hour) < MIN_DAY_HOURS:
        return None
    return by_hour


def _baseline_rows(p1, hours=None):
    """Flat persistence forecast: every median is hour 1's print."""
    hours = hours or list(range(2, 25))
    rows = []
    for h in hours:
        sigma = BASE_SIGMA * math.sqrt(h - 1)
        qs = [_round2(p1 * math.exp(z * sigma)) for z in Z]
        rows.append({"hour": h, "q": _clamp_row(qs)})
    return {"hours": rows}


def _clamp_row(qs):
    """Force quantiles into (0, 1e9] and non-decreasing."""
    out = []
    prev = 0.0
    for q in qs:
        if not math.isfinite(q) or q <= 0:
            q = prev if prev > 0 else 1.0
        q = min(q, 1e9)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def _hour1_print(case):
    """Hour-1 print, with fallbacks from the hourly history, else 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    history = case.get("history_hourly") or []
    if history:
        last = history[-1]
        viewers = last.get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            return float(viewers)
    return 1.0


def forecast_case(case):
    """Forecast hours 2..24 for one case dict; never raises."""
    try:
        return _forecast_case(case)
    except Exception:
        try:
            p1 = _hour1_print(case)
            hours = _forecast_hours(case)
        except Exception:
            p1 = 1.0
            hours = list(range(2, 25))
        return _baseline_rows(max(1.0, p1), hours)


def _forecast_case(case):
    p1 = max(1.0, _hour1_print(case))
    hours = _forecast_hours(case)
    log_p1 = math.log(p1)

    # Collect each usable comparable's log-ratios, weighted by closeness
    # of its hour-1 print to this case's.
    curves = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: str(c.get("name"))):
        day = _release_day(comp)
        if day is None:
            continue
        weight = math.exp(-0.5 * ((math.log(day[1]) - log_p1) / BANDWIDTH) ** 2)
        if weight < 1e-6:
            continue
        curves.append((weight, day))

    if len(curves) < 5:
        return _baseline_rows(p1, hours)

    medians = {}
    sigmas = {}
    for h in hours:
        pairs = sorted(
            (math.log(day[h] / day[1]), weight)
            for weight, day in curves
            if h in day
        )
        if len(pairs) < 5:
            medians[h] = 0.0
            sigmas[h] = BASE_SIGMA * math.sqrt(h - 1)
            continue
        medians[h] = _weighted_quantile(pairs, 0.5)
        iqr = _weighted_quantile(pairs, 0.75) - _weighted_quantile(pairs, 0.25)
        sigmas[h] = max(SIGMA_FLOOR, iqr / 1.349)

    medians = _smooth(medians, hours)
    sigmas = _smooth(sigmas, hours)

    rows = []
    for h in hours:
        ratio = math.exp(medians[h])
        if ratio < MEDIAN_RATIO_FLOOR:
            ratio = MEDIAN_RATIO_FLOOR
        median = p1 * ratio
        qs = [_round2(median * math.exp(z * sigmas[h])) for z in Z]
        rows.append({"hour": h, "q": _clamp_row(qs)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            forecast = _baseline_rows(1.0)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
