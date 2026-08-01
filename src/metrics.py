"""Volleyball statistics for the subject, overall and sliced by position.

Outcomes come from three things already established upstream: the decoded action
sequence (`grammar.py`), who won the rally (`rallies.py`, via the side-out rule),
and which touch was the last one. That combination is enough to separate a kill
from an attempt that stayed in play, without any ball tracking:

    attack is the rally's last contact  +  his side won   -> kill
    attack is the rally's last contact  +  his side lost  -> error
    attack is followed by more play                       -> attempt, in play

Everything here degrades honestly. A rally whose winner is unknown (the last of
a set, or a serve nobody could attribute) is counted as an attempt and excluded
from the kill/error split rather than guessed at, and every rate carries the
denominator it was computed from so a 100 % kill rate off two swings cannot be
mistaken for form.
"""

from collections import defaultdict

import numpy as np

MIN_SAMPLE = {          # below these, a rate is reported but marked low-sample
    "attack": 10,
    "serve": 8,
    "pass": 10,
    "set": 10,
    "block": 6,
    "dig": 8,
}


SubjectIds = "dict[int, set[int]] | set[int]"


def ids_for(subject_ids, rally_index: int) -> set[int]:
    """The subject's track ids during one rally.

    Accepts either a per-rally mapping — which is what the pipeline produces,
    because the tracker gives him different ids in different rallies — or a
    single set meaning "the same ids all match", which is the simple case and
    what most tests use.
    """
    if isinstance(subject_ids, dict):
        return subject_ids.get(rally_index, set())
    return subject_ids


def subject_plays(plays: list[dict], subject_ids, rally_index: int = 0) -> list[dict]:
    ids = ids_for(subject_ids, rally_index)
    return [p for p in plays if p["track_id"] in ids]


def _last_contact(plays: list[dict]) -> dict | None:
    return plays[-1] if plays else None


def attack_outcome(rally_plays: list[dict], idx: int, winner: str | None) -> str:
    """'kill', 'error', 'blocked', or 'in_play' for the attack at `idx`."""
    play = rally_plays[idx]
    last = _last_contact(rally_plays)
    if winner is None or last is None:
        return "unknown"
    is_last = idx == len(rally_plays) - 1
    if is_last:
        return "kill" if winner == play["side"] else "error"
    # an attack stopped dead by the opponent's block is charged to the attacker
    nxt = rally_plays[idx + 1]
    if (idx + 1 == len(rally_plays) - 1 and nxt["action"] == "block"
            and nxt["side"] != play["side"]):
        return "blocked" if winner != play["side"] else "in_play"
    return "in_play"


def serve_outcome(rally_plays: list[dict], idx: int, winner: str | None) -> str:
    """'ace', 'error', or 'in_play'. An ace is a serve nobody touched."""
    play = rally_plays[idx]
    if winner is None:
        return "unknown"
    if len(rally_plays) == 1:
        return "ace" if winner == play["side"] else "error"
    return "in_play"


def pass_rating(rally_plays: list[dict], idx: int) -> int:
    """A 0-3 passer rating inferred from what the pass made possible.

    This is the standard coaching scale expressed in terms we can actually
    observe: a 3-ball lets the setter run the offence, a 1-ball gets sent back
    over, a 0 ends the rally on the spot. It measures the *consequence* of the
    pass rather than its trajectory, which is the honest thing to do without
    reliable ball tracking — and is close to how a coach grades it anyway.
    """
    play = rally_plays[idx]
    rest = rally_plays[idx + 1:]
    if not rest:
        return 0                                     # nothing followed: shanked
    same = [p for p in rest if p["side"] == play["side"]]
    if not same:
        return 1                                     # went straight back over
    if rest[0]["side"] != play["side"]:
        return 1
    if rest[0]["action"] == "set":
        after_set = rest[1:2]
        if after_set and after_set[0]["action"] == "attack":
            return 3                                 # in system
        return 2
    if rest[0]["action"] == "attack":
        return 2                                     # out of system but attacked
    return 2


def set_is_assist(rally_plays: list[dict], idx: int, winner: str | None) -> bool:
    """A set counts as an assist when the attack it fed was a kill."""
    play = rally_plays[idx]
    nxt = rally_plays[idx + 1:idx + 2]
    if not nxt or nxt[0]["action"] != "attack" or nxt[0]["side"] != play["side"]:
        return False
    return attack_outcome(rally_plays, idx + 1, winner) == "kill"


def block_outcome(rally_plays: list[dict], idx: int, winner: str | None) -> str:
    """'stuff' when the block ended the rally in the blocker's favour."""
    play = rally_plays[idx]
    if winner is None:
        return "unknown"
    if idx == len(rally_plays) - 1:
        return "stuff" if winner == play["side"] else "error"
    return "touch"


def dig_converted(rally_plays: list[dict], idx: int) -> bool:
    """A dig is converted when his side got an attack out of it."""
    play = rally_plays[idx]
    for p in rally_plays[idx + 1:]:
        if p["side"] != play["side"]:
            return False
        if p["action"] == "attack":
            return True
    return False


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def skill_lines(rallies, plays_by_rally: dict[int, list[dict]],
                subject_ids, rally_indices: list[int] | None = None,
                roles_by_rally: dict | None = None) -> dict:
    """Aggregate every skill over a chosen set of rallies."""
    from rotation import back_row_attack

    wanted = set(rally_indices) if rally_indices is not None else None
    roles_by_rally = roles_by_rally or {}
    acc = defaultdict(int)
    pass_ratings: list[int] = []

    for rally in rallies:
        if wanted is not None and rally.index not in wanted:
            continue
        plays = plays_by_rally.get(rally.index, [])
        winner = rally.winner
        mine = ids_for(subject_ids, rally.index)
        role = roles_by_rally.get(rally.index)
        for i, p in enumerate(plays):
            if p["track_id"] not in mine:
                continue
            action = p["action"]
            acc[action] += 1
            if action == "attack":
                acc[f"attack_{attack_outcome(plays, i, winner)}"] += 1
                # a back-row player taking off in front of the 3 m line has
                # given the point away — a fault, not a statistic
                if role is not None and back_row_attack(role, p.get("y", 0.0)):
                    acc["attack_back_row_fault"] += 1
            elif action == "serve":
                acc[f"serve_{serve_outcome(plays, i, winner)}"] += 1
            elif action == "pass":
                r = pass_rating(plays, i)
                pass_ratings.append(r)
                acc[f"pass_{r}"] += 1
            elif action == "set":
                acc["set_assist"] += int(set_is_assist(plays, i, winner))
            elif action == "block":
                acc[f"block_{block_outcome(plays, i, winner)}"] += 1
            elif action == "dig":
                acc["dig_converted"] += int(dig_converted(plays, i))

    return _assemble(acc, pass_ratings)


def _assemble(acc: dict, pass_ratings: list[int]) -> dict:
    attacks = acc["attack"]
    kills = acc["attack_kill"]
    errors = acc["attack_error"] + acc["attack_blocked"]
    graded_attacks = kills + errors + acc["attack_in_play"]

    serves = acc["serve"]
    graded_serves = acc["serve_ace"] + acc["serve_error"] + acc["serve_in_play"]

    return {
        "attacking": {
            "attempts": attacks,
            "kills": kills,
            "errors": errors,
            "in_play": acc["attack_in_play"],
            "unscored": attacks - graded_attacks,
            "back_row_faults": acc["attack_back_row_fault"],
            # hitting percentage is (kills - errors) / attempts, the standard
            # volleyball rate — it can legitimately be negative
            "hitting_pct": _rate(kills - errors, graded_attacks),
            "kill_pct": _rate(kills, graded_attacks),
            "low_sample": graded_attacks < MIN_SAMPLE["attack"],
        },
        "serving": {
            "attempts": serves,
            "aces": acc["serve_ace"],
            "errors": acc["serve_error"],
            "in_play": acc["serve_in_play"],
            "unscored": serves - graded_serves,
            "ace_pct": _rate(acc["serve_ace"], graded_serves),
            "error_pct": _rate(acc["serve_error"], graded_serves),
            "low_sample": graded_serves < MIN_SAMPLE["serve"],
        },
        "passing": {
            "attempts": acc["pass"],
            "rating": round(float(np.mean(pass_ratings)), 2) if pass_ratings else None,
            "perfect_pct": _rate(acc["pass_3"], len(pass_ratings)),
            "shank_pct": _rate(acc["pass_0"], len(pass_ratings)),
            "distribution": {str(i): acc[f"pass_{i}"] for i in range(4)},
            "low_sample": len(pass_ratings) < MIN_SAMPLE["pass"],
        },
        "setting": {
            "attempts": acc["set"],
            "assists": acc["set_assist"],
            "assist_pct": _rate(acc["set_assist"], acc["set"]),
            "low_sample": acc["set"] < MIN_SAMPLE["set"],
        },
        "blocking": {
            "attempts": acc["block"],
            "stuffs": acc["block_stuff"],
            "touches": acc["block_touch"],
            "stuff_pct": _rate(acc["block_stuff"], acc["block"]),
            "low_sample": acc["block"] < MIN_SAMPLE["block"],
        },
        "defense": {
            "digs": acc["dig"],
            "converted": acc["dig_converted"],
            "conversion_pct": _rate(acc["dig_converted"], acc["dig"]),
            "low_sample": acc["dig"] < MIN_SAMPLE["dig"],
        },
    }


def movement(positions, rallies) -> dict:
    """Distance covered and work rate, measured only while the ball is in play.

    Restricting to rally windows is the point: a match is mostly dead time, and
    including it would turn "how hard did he work" into "how long was he on
    court".
    """
    if positions is None or len(positions) == 0 or not rallies:
        return {"distance_m": 0.0, "rallies": 0}
    total = 0.0
    counted = 0
    for r in rallies:
        seg = positions[(positions["t"] >= r.start) & (positions["t"] <= r.end)]
        if len(seg) < 2:
            continue
        d = np.hypot(np.diff(seg["x"].to_numpy()), np.diff(seg["y"].to_numpy()))
        # drop single-frame jumps larger than a sprint could produce: those are
        # tracker id switches, not the player teleporting
        total += float(d[d < 2.0].sum())
        counted += 1
    play_time = sum(r.duration for r in rallies)
    return {
        "distance_m": round(total, 1),
        "rallies": counted,
        "metres_per_rally": round(total / counted, 1) if counted else 0.0,
        "metres_per_minute_in_play": round(total / (play_time / 60), 1)
        if play_time else 0.0,
    }


def by_role(rallies, plays_by_rally: dict[int, list[dict]], subject_ids,
            role_groups: dict[str, list[int]], roles_by_rally=None) -> dict:
    """The same skill lines, computed separately for each position played."""
    return {
        role: skill_lines(rallies, plays_by_rally, subject_ids, indices,
                          roles_by_rally)
        | {"rallies": len(indices)}
        for role, indices in role_groups.items()
    }


def compute(rallies, plays_by_rally: dict[int, list[dict]], subject_ids,
            role_groups: dict[str, list[int]], positions=None,
            jump_summary: dict | None = None, roles_by_rally=None) -> dict:
    return {
        "overall": skill_lines(rallies, plays_by_rally, subject_ids,
                               roles_by_rally=roles_by_rally),
        "by_role": by_role(rallies, plays_by_rally, subject_ids, role_groups,
                           roles_by_rally),
        "movement": movement(positions, rallies),
        "jumps": jump_summary or {"count": 0},
        "coverage": {
            "rallies": len(rallies),
            "rallies_with_winner": sum(1 for r in rallies if r.winner),
            "rallies_with_subject": sum(
                1 for r in rallies if ids_for(subject_ids, r.index)),
            "subject_touches": sum(
                len(subject_plays(plays_by_rally.get(r.index, []),
                                  subject_ids, r.index))
                for r in rallies),
        },
    }
