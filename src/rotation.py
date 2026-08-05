"""Which position the subject was playing, rally by rally.

Eric plays every position, so a single blended stat line would average an
outside hitter's swings together with a libero's digs and describe nobody. Every
metric therefore carries the role he was actually in for that rally.

Two different zones matter and they are not the same thing:

- `serve_zone` is the rotational slot at the moment of the serve. This is the
  formal rotation position, and it is what determines whether he is legally
  front or back row for that rally.
- `play_zone` is where he actually spends the rally. Teams switch immediately
  after contact — a setter starting in zone 1 runs to the right front — so the
  specialist position he is really filling comes from here.

Reporting the specialist role off `serve_zone` would label every switching
player wrong, and reporting the front/back row off `play_zone` would call a
back-row attacker a front-row one. Both are kept.
"""

from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd

from court import ZONE_ROLES, ZONE_SPECIALIST, in_front_row, on_court, zone_for
from tracking import feet_px

SERVE_WINDOW_S = 0.8   # around the rally start, before players switch


@dataclass
class RallyRole:
    rally_index: int
    serve_zone: int | None
    play_zone: int | None
    row: str | None            # 'front' | 'back'
    role: str | None           # specialist position, from play_zone
    rotation_slot: str | None  # human name of serve_zone

    def as_dict(self) -> dict:
        return {
            "rally_index": self.rally_index,
            "serve_zone": self.serve_zone,
            "play_zone": self.play_zone,
            "row": self.row,
            "role": self.role,
            "rotation_slot": self.rotation_slot,
        }


def _zones(subject: pd.DataFrame, calib, f0: int, f1: int) -> list[int]:
    win = subject[(subject["frame"] >= f0) & (subject["frame"] <= f1)]
    if win.empty:
        return []
    pts = calib.to_court(feet_px(win), win["frame"])
    return [z for x, y in pts if on_court(x, y) and (z := zone_for(x, y)) is not None]


def rally_role(rally, subject: pd.DataFrame, calib, fps: float) -> RallyRole:
    """The subject's rotational slot and playing position for one rally."""
    f_start = int(rally.start * fps)
    serve_zones = _zones(subject, calib, f_start,
                         int((rally.start + SERVE_WINDOW_S) * fps))
    play_zones = _zones(subject, calib, f_start, int(rally.end * fps))

    serve_zone = Counter(serve_zones).most_common(1)[0][0] if serve_zones else None
    play_zone = Counter(play_zones).most_common(1)[0][0] if play_zones else None

    row = None
    if serve_zone is not None:
        # zones 2, 3, 4 are the front row by definition of the rotation
        row = "front" if serve_zone in (2, 3, 4) else "back"

    return RallyRole(
        rally_index=rally.index,
        serve_zone=serve_zone,
        play_zone=play_zone,
        row=row,
        role=ZONE_SPECIALIST.get(play_zone) if play_zone else None,
        rotation_slot=ZONE_ROLES.get(serve_zone) if serve_zone else None,
    )


def rally_roles(rallies, tracks: pd.DataFrame, ids_by_rally: dict[int, set[int]],
                calib, fps: float) -> list[RallyRole]:
    """One role per rally, each read from that rally's own subject track ids.

    The ids differ from rally to rally — the tracker does not keep one id for a
    player across a dead ball — so a single subject frame set would only
    describe whichever rally it came from.
    """
    out = []
    for r in rallies:
        ids = ids_by_rally.get(r.index, set())
        seg = tracks[tracks["track_id"].isin(ids)] if ids else tracks.iloc[0:0]
        out.append(rally_role(r, seg, calib, fps))
    return out


def back_row_attack(role: RallyRole, contact_y: float) -> bool:
    """True when a back-row player attacked from in front of the attack line.

    Worth surfacing because it is a fault, not a stat: a back-row player who
    takes off in front of the 3 m line has given the point away.
    """
    return role.row == "back" and in_front_row(contact_y)


def summarise(roles: list[RallyRole]) -> dict:
    """How the match was split across positions — the denominator for every
    per-role stat, and the honesty check on whether a role has enough sample."""
    known = [r for r in roles if r.role]
    counts = Counter(r.role for r in known)
    rows = Counter(r.row for r in roles if r.row)
    return {
        "rallies": len(roles),
        "rallies_with_role": len(known),
        "by_role": dict(counts.most_common()),
        "by_row": dict(rows),
        "primary_role": counts.most_common(1)[0][0] if counts else None,
        "positions_played": len(counts),
    }


def group_by_role(roles: list[RallyRole]) -> dict[str, list[int]]:
    """role -> rally indices, so metrics can be sliced without re-deriving."""
    out: dict[str, list[int]] = {}
    for r in roles:
        if r.role:
            out.setdefault(r.role, []).append(r.rally_index)
    return out


def rotation_coverage(roles: list[RallyRole]) -> dict:
    """How evenly the six rotational slots were seen.

    A match where he served from only two slots cannot support per-rotation
    claims, and this is what tells the report to say so.
    """
    slots = [r.serve_zone for r in roles if r.serve_zone]
    if not slots:
        return {"slots_seen": 0, "balanced": False}
    counts = Counter(slots)
    values = np.array(list(counts.values()), dtype=float)
    return {
        "slots_seen": len(counts),
        "per_slot": {ZONE_ROLES[z]: n for z, n in sorted(counts.items())},
        "balanced": bool(len(counts) >= 5 and values.min() >= 0.5 * values.mean()),
    }
