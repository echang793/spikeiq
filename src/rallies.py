"""Rally segmentation, serving side, and rally winners.

Two ideas carry this module:

1. **The referee brackets every rally.** A whistle starts the serve and another
   ends the point, so a rally is the interval between consecutive whistles that
   actually contains ball contacts. Intervals with no contacts are timeouts,
   substitutions and between-point dead air, and fall out for free. When a
   recording has no referee (open gym) we fall back to dinkiq's silence-gap
   segmentation.

2. **The side-out rule hands us the winner of every rally.** Whoever wins a
   rally serves the next one, so `winner(N) = serving_side(N + 1)` — no ball
   tracking, no scoreboard OCR, no manual tagging. The chain only holds inside
   a set: teams switch ends between sets, and the last rally of a set has no
   successor, so those are deliberately left unscored rather than guessed.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from audio import Contact, Whistle
from court import COURT_L, NET_Y, on_court
from tracking import feet_px

MIN_RALLY_S = 0.8
MAX_RALLY_S = 45.0
GAP_RALLY_S = 4.0          # fallback mode: silence longer than this ends a rally
GAP_MIN_CONTACTS = 2       # fallback mode has no whistle to vouch for a rally
SET_BREAK_S = 90.0         # dead time this long means a new set (and switched ends)
MIN_WHISTLES_FOR_CLOCK = 6  # below this the recording effectively has no referee

SERVE_SEARCH_S = 0.6       # window after the start whistle to look for the server
BEHIND_LINE_M = 0.3        # how far past their own endline the server must stand


@dataclass
class Rally:
    index: int
    start: float
    end: float
    set_index: int = 0
    contacts: list[float] = field(default_factory=list)
    serving_side: str | None = None   # 'far' | 'near'
    winner: str | None = None
    source: str = "whistle"           # 'whistle' | 'gap'

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def serve_t(self) -> float | None:
        """The serve is the first contact of the rally."""
        return self.contacts[0] if self.contacts else None

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "set_index": self.set_index,
            "contacts": [round(t, 3) for t in self.contacts],
            "serve_t": round(self.serve_t, 3) if self.serve_t is not None else None,
            "serving_side": self.serving_side,
            "winner": self.winner,
            "source": self.source,
        }


def segment_rallies(whistles: Sequence[Whistle], contacts: Sequence[Contact],
                    duration: float | None = None) -> list[Rally]:
    """Split the recording into rallies, whistle-bracketed where possible."""
    times = np.array([c.t for c in contacts], dtype=float)
    if len(whistles) >= MIN_WHISTLES_FOR_CLOCK:
        rallies = _segment_by_whistle(whistles, times)
        if rallies:
            return _mark_sets(rallies)
    return _mark_sets(_segment_by_gap(times))


def _segment_by_whistle(whistles: Sequence[Whistle],
                        times: np.ndarray) -> list[Rally]:
    out: list[Rally] = []
    for a, b in zip(whistles, whistles[1:]):
        start, end = a.end, b.start
        if not (MIN_RALLY_S <= end - start <= MAX_RALLY_S):
            continue
        inside = times[(times >= start) & (times <= end)]
        if len(inside) == 0:
            continue  # timeout, substitution, or a re-serve whistle pair
        out.append(Rally(index=len(out), start=start, end=end,
                         contacts=inside.tolist(), source="whistle"))
    return out


def _segment_by_gap(times: np.ndarray) -> list[Rally]:
    """dinkiq's silence-gap segmentation, for footage with no referee."""
    if len(times) == 0:
        return []
    groups: list[list[float]] = [[float(times[0])]]
    for t in times[1:]:
        if t - groups[-1][-1] > GAP_RALLY_S:
            groups.append([])
        groups[-1].append(float(t))
    out: list[Rally] = []
    for g in groups:
        if len(g) < GAP_MIN_CONTACTS or g[-1] - g[0] > MAX_RALLY_S:
            continue
        out.append(Rally(index=len(out), start=g[0], end=g[-1],
                         contacts=g, source="gap"))
    return out


def _mark_sets(rallies: list[Rally]) -> list[Rally]:
    """Number the sets, splitting on long dead periods.

    This matters for more than bookkeeping: teams switch ends between sets, so
    the side-out chain must not be carried across a break.
    """
    set_index = 0
    for i, r in enumerate(rallies):
        if i > 0 and r.start - rallies[i - 1].end > SET_BREAK_S:
            set_index += 1
        r.set_index = set_index
        r.index = i
    return rallies


def detect_serving_side(rally: Rally, tracks: pd.DataFrame, calib, fps: float,
                        search_s: float = SERVE_SEARCH_S) -> str | None:
    """Which side served, from who is standing behind their own endline.

    At the start whistle the server is the one player past their own endline;
    everyone else is inside the court. Returns None when nobody is clearly
    behind a line (server cropped out of frame, or tracking dropped them) —
    an honest None here is far better than a coin-flip, because
    `assign_winners` propagates this backwards into the previous rally's result.
    """
    f0 = int(rally.start * fps)
    f1 = int((rally.start + search_s) * fps)
    win = tracks[(tracks["frame"] >= f0) & (tracks["frame"] <= f1)]
    if win.empty:
        return None
    pts = calib.to_court(feet_px(win), win["frame"])
    best_side, best_depth = None, 0.0
    for x, y in pts:
        if not on_court(x, y):
            continue
        # depth past the player's own endline: near side is beyond y=18,
        # far side beyond y=0
        if y > COURT_L:
            side, depth = "near", y - COURT_L
        elif y < 0.0:
            side, depth = "far", -y
        else:
            continue
        if depth >= BEHIND_LINE_M and depth > best_depth:
            best_side, best_depth = side, depth
    return best_side


def assign_winners(rallies: list[Rally]) -> list[Rally]:
    """Fill in `winner` from the side-out rule: the next rally's server won.

    Left as None across a set boundary and for the final rally of each set,
    where there is no next serve to read the answer off.
    """
    for cur, nxt in zip(rallies, rallies[1:]):
        if nxt.set_index == cur.set_index:
            cur.winner = nxt.serving_side
    return rallies


def subject_side(rally: Rally, subject: pd.DataFrame, calib, fps: float) -> str | None:
    """Which half the subject is playing on during a rally.

    Computed per rally from the subject's own tracked position rather than
    assumed once for the match, which makes the between-set end switch a
    non-issue: whatever half he is standing on is his team's half.
    """
    f0, f1 = int(rally.start * fps), int(rally.end * fps)
    win = subject[(subject["frame"] >= f0) & (subject["frame"] <= f1)]
    if win.empty:
        return None
    pts = calib.to_court(feet_px(win), win["frame"])
    ys = np.array([y for x, y in pts if on_court(x, y)])
    if len(ys) == 0:
        return None
    return "far" if float(np.mean(ys < NET_Y)) >= 0.5 else "near"


def summarise(rallies: list[Rally]) -> dict:
    if not rallies:
        return {"rally_count": 0,
                "note": "no rallies detected — missing or very quiet audio?"}
    durations = [r.duration for r in rallies]
    scored = [r for r in rallies if r.winner]
    return {
        "rally_count": len(rallies),
        "set_count": rallies[-1].set_index + 1,
        "source": rallies[0].source,
        "avg_contacts": round(float(np.mean([len(r.contacts) for r in rallies])), 1),
        "avg_seconds": round(float(np.mean(durations)), 1),
        "max_seconds": round(float(np.max(durations)), 1),
        "scored_rallies": len(scored),
        "serving_side_known": sum(1 for r in rallies if r.serving_side),
    }
