"""Coaching notes keyed to the weakest rubric dimensions.

Same shape as dinkiq's feedback engine: a tip per dimension with a low and a
high variant, chosen by where the player sits, and filled with the player's own
measured number so the note is specific rather than generic advice.

Two volleyball-specific additions:

- Notes are emitted per position as well as overall, because a player who
  rotates through every position can be strong as an outside hitter and weak in
  the middle, and one blended note would describe neither.
- Anything resting on a thin sample says so in the note itself. A tip built on
  four swings that reads with the same confidence as one built on forty is
  worse than no tip.
"""

from rating import RUBRIC, next_anchor_target, strengths_and_weaknesses

TIPS: dict[str, dict] = {
    "attacking": {
        "low": "You hit {value:+.3f} ({kills} kills, {errors} errors on {attempts} "
               "swings). Errors are costing more than kills are winning — take "
               "pace off and hit high-and-deep to zone 6 until the ratio flips. "
               "A clean ball in play beats a big one in the antenna.",
        "high": "You hit {value:+.3f} on {attempts} swings — that is doing real "
                "damage. Next layer is shot selection: watch for the block "
                "closing and take the line or a roll shot when it does.",
    },
    "passing": {
        "low": "Serve-receive averaged {value:.2f} out of 3, with {shank_pct:.0%} "
               "of passes ending the rally outright. Get your platform out early "
               "and let the ball come to you instead of reaching. Passing is the "
               "highest-leverage thing on this list: no offence runs off a 1-ball.",
        "high": "Serve-receive averaged {value:.2f} — your setter can run the "
                "middle off that. Keep it there and start working on passing "
                "tighter to the target rather than just keeping it up.",
    },
    "serving": {
        "low": "You aced {value:.0%} of serves. Free points are available: add "
               "pace or float and pick a seam — between two passers, or at the "
               "passer who just shanked one.",
        "high": "You aced {value:.0%} of serves — that is a weapon. Keep "
                "attacking the seams, and note which passer you are beating.",
    },
    "serve_control": {
        "low": "You missed {value:.0%} of serves. That is points handed over "
               "before the rally starts. Take a step back off the ceiling of "
               "your range: a serve in play at 80 % is worth far more than a "
               "missed one at 100 %.",
        "high": "Only {value:.0%} of serves missed — you are giving away almost "
                "nothing. You have room to serve tougher.",
    },
    "blocking": {
        "low": "You turned {value:.0%} of blocks into points on {attempts} "
               "attempts. Press your hands over the net rather than up, and "
               "close to the outside blocker — a sealed block with no seam beats "
               "two players reaching.",
        "high": "You scored on {value:.0%} of your blocks. Strong. Start reading "
                "the setter's hands earlier to get to the pin sooner.",
    },
    "defense": {
        "low": "Only {value:.0%} of your digs turned into an attack. The dig is "
               "getting up but not to a hittable spot — angle the platform "
               "towards the setter's target and stop the ball travelling.",
        "high": "{value:.0%} of your digs turned straight into offence — you are "
                "not just keeping balls alive, you are starting attacks.",
    },
    "setting": {
        "low": "{value:.0%} of your sets led to a kill. Get square to your "
                "target before the ball arrives, and set consistent height so "
                "hitters can time an approach off it.",
        "high": "{value:.0%} of your sets led to a kill — you are putting hitters "
                "in good positions. Next step is distribution: check whether you "
                "are going to the same pin every time.",
    },
    "athleticism": {
        "low": "Best jump measured {value:.2f} m. Approach mechanics usually "
               "return more than strength work here — a full four-step approach "
               "with a hard double-arm swing typically adds more than the gym "
               "does in the same weeks.",
        "high": "Best jump measured {value:.2f} m — you are getting well above "
                "the net. Use it: hit over the block rather than around it.",
    },
}

LOW_LEVEL = 2.5   # below this a dimension gets the corrective note


def _fill(template: str, dim: dict, metrics: dict, name: str) -> str:
    group = RUBRIC[name]["path"][0]
    stats = (metrics.get(group) or {}) if group != "__jumps__" else {}
    fields = {
        "value": dim["value"],
        "attempts": stats.get("attempts", stats.get("digs", 0)),
        "kills": stats.get("kills", 0),
        "errors": stats.get("errors", 0),
        "shank_pct": stats.get("shank_pct") or 0.0,
    }
    try:
        return template.format(**fields)
    except (KeyError, ValueError):   # a stat the template wanted is missing
        return template.split(".")[0] + "."


def tips_for(rating: dict, metrics: dict, limit: int = 4) -> list[dict]:
    """Coaching notes, weakest dimension first."""
    dims = rating.get("dimensions") or {}
    out: list[dict] = []
    for name, dim in sorted(dims.items(), key=lambda kv: kv[1]["level"]):
        if name not in TIPS:
            continue
        low = dim["level"] < LOW_LEVEL
        note = _fill(TIPS[name]["low" if low else "high"], dim, metrics, name)
        target = next_anchor_target(dim["value"], RUBRIC[name]["anchors"],
                                    RUBRIC[name].get("invert", False))
        if dim["low_sample"]:
            note += (" (Small sample this match — treat as a hint, not a verdict.)")
        out.append({
            "dimension": name,
            "label": dim["label"],
            "level": dim["level"],
            "priority": "work on" if low else "keep",
            "note": note,
            "next_target": ({"value": target[0], "band": target[1]}
                            if target else None),
            "low_sample": dim["low_sample"],
        })
    work = [t for t in out if t["priority"] == "work on"][:limit]
    keep = [t for t in out if t["priority"] == "keep"][:2]
    return work + keep


from textutil import plural as _plural


def per_role_notes(by_role: dict, min_rallies: int = 6) -> list[dict]:
    """One line per position played, so the report answers 'where am I strong'
    for a player who rotates through all of them."""
    notes = []
    for role, stats in sorted(by_role.items(),
                              key=lambda kv: -kv[1].get("rallies", 0)):
        rallies = stats.get("rallies", 0)
        if rallies < min_rallies:
            notes.append({"role": role, "rallies": rallies,
                          "note": f"Only {_plural(rallies, 'rally', 'rallies')} "
                                  f"at {role} — not enough to say anything yet."})
            continue
        atk = stats.get("attacking", {})
        pas = stats.get("passing", {})
        parts = []
        if atk.get("hitting_pct") is not None:
            parts.append(f"hit {atk['hitting_pct']:+.3f} on {atk['attempts']} swings")
        if pas.get("rating") is not None:
            parts.append(f"passed {pas['rating']:.2f}")
        blk = stats.get("blocking", {})
        if blk.get("stuffs"):
            parts.append(_plural(blk["stuffs"], "stuff block", "stuff blocks"))
        summary = "; ".join(parts) if parts else "no scoring touches recorded"
        notes.append({
            "role": role, "rallies": rallies,
            "note": f"At {role} over {_plural(rallies, 'rally', 'rallies')}: "
                    f"{summary}."})
    return notes


MAX_EXAMPLES = 3


def example_rallies(dimension: str, rallies, plays_by_rally: dict,
                    subject_ids) -> list[int]:
    """Rallies that actually show the problem the tip is about.

    A tip saying "your errors cost more than your kills" is advice; the same
    tip attached to the three rallies where it happened is something you can
    watch. Only rallies where the subject did the thing in question qualify —
    a clip of somebody else's error teaches nothing.
    """
    from metrics import (attack_outcome, block_outcome, dig_converted,
                         ids_for, pass_rating, serve_outcome, set_is_assist)

    hits: list[tuple[float, int]] = []
    for rally in rallies:
        plays = plays_by_rally.get(rally.index, [])
        mine = ids_for(subject_ids, rally.index)
        for i, p in enumerate(plays):
            if p["track_id"] not in mine:
                continue
            action, rank = p["action"], None
            if dimension == "attacking" and action == "attack":
                if attack_outcome(plays, i, rally.winner) in ("error", "blocked"):
                    rank = 0.0
            elif dimension == "passing" and action == "pass":
                grade = pass_rating(plays, i)
                if grade <= 1:
                    rank = float(grade)
            elif dimension in ("serving", "serve_control") and action == "serve":
                if serve_outcome(plays, i, rally.winner) == "error":
                    rank = 0.0
            elif dimension == "blocking" and action == "block":
                if block_outcome(plays, i, rally.winner) != "stuff":
                    rank = 0.0
            elif dimension == "defense" and action == "dig":
                if not dig_converted(plays, i):
                    rank = 0.0
            elif dimension == "setting" and action == "set":
                if not set_is_assist(plays, i, rally.winner):
                    rank = 0.0
            if rank is not None:
                hits.append((rank, rally.index))
                break
    hits.sort()
    seen, out = set(), []
    for _, idx in hits:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
        if len(out) >= MAX_EXAMPLES:
            break
    return out


def build(rating: dict, metrics: dict, rallies=None, plays_by_rally=None,
          subject_ids=None) -> dict:
    """The whole feedback payload the dashboard renders."""
    sw = strengths_and_weaknesses(rating)
    overall = metrics.get("overall", metrics)
    tips = tips_for(rating, overall)
    if rallies is not None and plays_by_rally is not None:
        for tip in tips:
            if tip["priority"] == "work on":
                tip["examples"] = example_rallies(
                    tip["dimension"], rallies, plays_by_rally, subject_ids or {})
    return {
        "band": rating.get("band"),
        "level": rating.get("level"),
        "confidence": rating.get("confidence"),
        "strengths": sw["strengths"],
        "weaknesses": sw["weaknesses"],
        "tips": tips,
        "by_role": per_role_notes(metrics.get("by_role", {})),
        "caveat": rating.get("note"),
    }
