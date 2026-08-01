"""Per-skill grades and an overall level estimate.

Volleyball has no universal player rating the way pickleball has DUPR, so the
output here is a per-skill level on the club ladder (B / BB / A / AA / Open)
plus a weighted overall. Phrase it as a resemblance, never a certification —
"your attacking resembles A level" is a defensible claim; "you are an A player"
is not.

`RUBRIC` is the single tuning point, exactly as `dupr.RUBRIC` is in dinkiq.
The anchor values in it are informed starting points from how the sport is
generally played, NOT fitted to data — nobody has yet run this against film of
players whose level is known. Until that happens the numbers are a consistent
yardstick for tracking your own progress, which is what they are used for, and
not a claim about where you sit against the field. Calibrating them is a matter
of replacing anchors here and nothing else.
"""

import numpy as np

LEVELS = {1.0: "B", 2.0: "BB", 3.0: "A", 4.0: "AA", 5.0: "Open"}

# anchors map a measured value to a level score; `invert` marks dimensions where
# lower is better, and `weight` sets how much the dimension moves the overall.
RUBRIC: dict[str, dict] = {
    "attacking": {
        "path": ("attacking", "hitting_pct"),
        "weight": 1.4,
        "anchors": [(-0.10, 1.0), (0.05, 2.0), (0.15, 3.0), (0.25, 4.0), (0.35, 5.0)],
        "label": "hitting percentage",
    },
    "passing": {
        "path": ("passing", "rating"),
        "weight": 1.3,
        "anchors": [(1.60, 1.0), (1.90, 2.0), (2.15, 3.0), (2.35, 4.0), (2.50, 5.0)],
        "label": "serve-receive rating",
    },
    "serving": {
        "path": ("serving", "ace_pct"),
        "weight": 0.9,
        "anchors": [(0.02, 1.0), (0.05, 2.0), (0.08, 3.0), (0.12, 4.0), (0.16, 5.0)],
        "label": "ace rate",
    },
    "serve_control": {
        "path": ("serving", "error_pct"),
        "weight": 0.8,
        "invert": True,
        "anchors": [(0.05, 5.0), (0.08, 4.0), (0.11, 3.0), (0.15, 2.0), (0.20, 1.0)],
        "label": "service error rate",
    },
    "blocking": {
        "path": ("blocking", "stuff_pct"),
        "weight": 0.9,
        "anchors": [(0.05, 1.0), (0.10, 2.0), (0.16, 3.0), (0.22, 4.0), (0.30, 5.0)],
        "label": "block-for-point rate",
    },
    "defense": {
        "path": ("defense", "conversion_pct"),
        "weight": 1.0,
        "anchors": [(0.30, 1.0), (0.42, 2.0), (0.52, 3.0), (0.62, 4.0), (0.72, 5.0)],
        "label": "dig-to-attack conversion",
    },
    "setting": {
        "path": ("setting", "assist_pct"),
        "weight": 0.7,
        "anchors": [(0.15, 1.0), (0.22, 2.0), (0.30, 3.0), (0.38, 4.0), (0.46, 5.0)],
        "label": "assist rate",
    },
    "athleticism": {
        "path": ("__jumps__", "best_m"),
        "weight": 0.8,
        "anchors": [(0.35, 1.0), (0.45, 2.0), (0.55, 3.0), (0.65, 4.0), (0.75, 5.0)],
        "label": "best jump",
    },
}


def interp_band(value: float, anchors: list[tuple[float, float]]) -> float:
    """Piecewise-linear level for a measured value, clamped at both ends."""
    xs = [a for a, _ in anchors]
    ys = [b for _, b in anchors]
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    return float(np.interp(value, xs, ys))


def next_anchor_target(value: float, anchors: list[tuple[float, float]],
                       invert: bool = False) -> tuple[float, float] | None:
    """The next (value, level) worth aiming at — what the feedback quotes as a
    concrete target rather than a vague 'improve this'."""
    ordered = sorted(anchors, key=lambda a: a[1])
    for v, level in ordered:
        better = v < value if invert else v > value
        if better:
            return (v, level)
    return None


def _dig(metrics: dict, jumps: dict, path: tuple[str, str]):
    group, key = path
    if group == "__jumps__":
        return jumps.get(key)
    return (metrics.get(group) or {}).get(key)


def _is_low_sample(metrics: dict, path: tuple[str, str]) -> bool:
    group, _ = path
    if group == "__jumps__":
        return (metrics.get("__jumps__") or {}).get("count", 0) < 5
    return bool((metrics.get(group) or {}).get("low_sample", True))


def extract_dimension_values(metrics: dict, jumps: dict) -> dict[str, dict]:
    """Measured value, level and sample status for every rubric dimension."""
    out: dict[str, dict] = {}
    for name, spec in RUBRIC.items():
        value = _dig(metrics, jumps, spec["path"])
        if value is None:
            continue
        low = (jumps.get("count", 0) < 5 if spec["path"][0] == "__jumps__"
               else _is_low_sample(metrics, spec["path"]))
        out[name] = {
            "value": round(float(value), 3),
            "level": round(interp_band(float(value), spec["anchors"]), 2),
            "label": spec["label"],
            "low_sample": bool(low),
            "weight": spec["weight"],
        }
    return out


def estimate(metrics: dict, jumps: dict | None = None) -> dict:
    """Overall level estimate with per-skill grades and a confidence.

    Low-sample dimensions still appear — hiding them would make a thin match
    look like a complete picture — but they carry half weight in the overall and
    they drag the confidence down.
    """
    jumps = jumps or {}
    dims = extract_dimension_values(metrics, jumps)
    if not dims:
        return {"level": None, "band": None, "confidence": 0.0, "dimensions": {},
                "note": "not enough was measurable to estimate a level"}

    weights = np.array([d["weight"] * (0.5 if d["low_sample"] else 1.0)
                        for d in dims.values()])
    levels = np.array([d["level"] for d in dims.values()])
    overall = float(np.average(levels, weights=weights))

    solid = sum(1 for d in dims.values() if not d["low_sample"])
    coverage = (metrics.get("coverage") or {})
    scored = coverage.get("rallies_with_winner", 0)
    confidence = _confidence(solid, len(dims), scored)

    return {
        "level": round(overall, 2),
        "band": band_for(overall),
        "confidence": round(confidence, 2),
        "dimensions": dims,
        "solid_dimensions": solid,
        "note": ("levels resemble the club ladder (B/BB/A/AA/Open); anchors are "
                 "uncalibrated, so treat these as a yardstick for your own "
                 "progress rather than a placement"),
    }


def _confidence(solid: int, total: int, scored_rallies: int) -> float:
    """Confidence in the estimate: how many skills had a real sample, how much
    of the rubric was measurable at all, and how many rallies could be scored."""
    if total == 0:
        return 0.0
    breadth = solid / len(RUBRIC)
    depth = min(1.0, scored_rallies / 60.0)
    measured = total / len(RUBRIC)
    return float(np.clip(0.5 * breadth + 0.3 * depth + 0.2 * measured, 0.0, 1.0))


def band_for(level: float) -> str:
    """Club-ladder name for a level score, rounded to the nearest whole band."""
    keys = sorted(LEVELS)
    nearest = min(keys, key=lambda k: abs(k - level))
    return LEVELS[nearest]


def strengths_and_weaknesses(rating: dict, n: int = 3) -> dict:
    """The dimensions furthest above and below this player's own average.

    Relative to himself on purpose: telling someone their weakest skill is the
    one they are worst at *compared to their own other skills* is the actionable
    version, and it does not depend on the anchors being calibrated.
    """
    dims = rating.get("dimensions") or {}
    if not dims:
        return {"strengths": [], "weaknesses": []}
    mean = float(np.mean([d["level"] for d in dims.values()]))
    ranked = sorted(dims.items(), key=lambda kv: kv[1]["level"])
    weak = [{"dimension": k, **v, "delta": round(v["level"] - mean, 2)}
            for k, v in ranked[:n]]
    strong = [{"dimension": k, **v, "delta": round(v["level"] - mean, 2)}
              for k, v in reversed(ranked[-n:])]
    return {"strengths": strong, "weaknesses": weak, "own_average": round(mean, 2)}
