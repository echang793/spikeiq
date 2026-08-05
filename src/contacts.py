"""Attribute each audio contact to a player and describe it with pose features.

The audio layer says *when* the ball was touched; this module says *who* touched
it and *what their body was doing*, which is everything `grammar.py` needs to
decide what kind of touch it was.

Every feature is scale-free — normalised by the player's own torso length or by
the court — because a player at the far endline is a third the size in pixels of
one at the near endline, and a raw-pixel threshold would quietly mean different
things at the two ends of the same court.
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from court import NET_Y, dist_from_net, side_of, zone_for
from tracking import feet_px

HAND_WINDOW_S = 0.20     # how far either side of the pop to look for the hand peak
MIN_TORSO_PX = 12.0      # below this the pose is too small to read reliably
JUMP_THRESHOLD = 0.25    # torso-lengths of ankle rise that counts as airborne


@dataclass
class ContactFeatures:
    t: float
    frame: int
    track_id: int
    side: str                 # 'far' | 'near'
    x: float                  # court metres
    y: float
    zone: int | None
    net_dist: float
    strength: float           # audio loudness, 0-1
    hand_height: float        # torso-lengths above the shoulders (negative = below)
    hand_spread: float        # torso-lengths between the two wrists
    airborne: float           # torso-lengths of ankle rise above the standing base
    hand_speed: float         # torso-lengths per second
    confidence: float         # 0-1, how much the pose supports any of this

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _mean_xy(df: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """Midpoint of a symmetric keypoint pair, with (0,0) meaning 'not detected'.

    Averaging blindly would drag the midpoint halfway to the origin whenever one
    side of the body is occluded — which, with twelve players, is most frames.
    """
    pa = df[[f"{a}x", f"{a}y"]].to_numpy(dtype=float)
    pb = df[[f"{b}x", f"{b}y"]].to_numpy(dtype=float)
    ok_a = (pa != 0).all(axis=1)
    ok_b = (pb != 0).all(axis=1)
    out = np.full((len(df), 2), np.nan)
    both = ok_a & ok_b
    out[both] = (pa[both] + pb[both]) / 2.0
    out[ok_a & ~ok_b] = pa[ok_a & ~ok_b]
    out[ok_b & ~ok_a] = pb[ok_b & ~ok_a]
    return out


def torso_length(df: pd.DataFrame) -> np.ndarray:
    """Shoulder-to-hip distance in pixels: the per-player scale reference."""
    sh = _mean_xy(df, "lsh", "rsh")
    hip = _mean_xy(df, "lhip", "rhip")
    return np.hypot(sh[:, 0] - hip[:, 0], sh[:, 1] - hip[:, 1])


def hand_metrics(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Per-frame hand height, spread and speed, all in torso-lengths.

    Image y grows downward, so a wrist ABOVE the shoulders has the smaller y —
    `hand_height` flips the sign so that positive means overhead, which is what
    separates a set or a block from a forearm pass.
    """
    df = df.sort_values("frame").reset_index(drop=True)
    torso = torso_length(df)
    torso = np.where(torso >= MIN_TORSO_PX, torso, np.nan)

    sh = _mean_xy(df, "lsh", "rsh")
    # .copy(): pandas 3 hands back read-only views under copy-on-write, and
    # these get masked in place
    lw = df[["lwx", "lwy"]].to_numpy(dtype=float).copy()
    rw = df[["rwx", "rwy"]].to_numpy(dtype=float).copy()
    lw[(lw == 0).all(axis=1)] = np.nan
    rw[(rw == 0).all(axis=1)] = np.nan

    highest_wrist_y = np.fmin(lw[:, 1], rw[:, 1])
    hand_height = (sh[:, 1] - highest_wrist_y) / torso
    hand_spread = np.hypot(lw[:, 0] - rw[:, 0], lw[:, 1] - rw[:, 1]) / torso

    t = df["frame"].to_numpy() / fps
    hands = np.where(np.isnan(lw), rw, lw)
    dt = np.diff(t)
    speed = np.full(len(df), np.nan)
    with np.errstate(invalid="ignore"):
        step = np.hypot(np.diff(hands[:, 0]), np.diff(hands[:, 1]))
        speed[1:] = step / np.clip(dt, 1e-3, None) / torso[1:]

    return pd.DataFrame({
        "frame": df["frame"].to_numpy(), "t": t,
        "hand_height": hand_height, "hand_spread": hand_spread,
        "hand_speed": speed, "torso": torso,
    })


def airborne_series(df: pd.DataFrame) -> np.ndarray:
    """Ankle rise above this track's own standing baseline, in torso-lengths.

    The baseline is the track's median ankle height, i.e. where this player
    stands when they are on the floor. That is per-track on purpose: a player at
    the far endline sits higher in the frame than one at the net, so a shared
    baseline would report every far-court player as permanently airborne.
    """
    ank = _mean_xy(df, "lank", "rank")
    torso = torso_length(df)
    torso = np.where(torso >= MIN_TORSO_PX, torso, np.nan)
    y = ank[:, 1]
    if np.isnan(y).all():
        return np.full(len(df), np.nan)
    base = np.nanmedian(y)
    return (base - y) / torso   # image y grows downward, so rising = smaller y


# A volleyball leaves a hard spike at roughly 25 m/s and slows in flight, so
# this is a generous ceiling on how far the ball can have travelled between two
# touches — generous on purpose, because the job is to rule out the impossible,
# not to model the trajectory.
BALL_MAX_SPEED = 25.0
REACH_SLACK_M = 2.0     # homography and tracking error, plus arm's length


class RallyTracks:
    """Per-rally track data with the pose derivatives computed once.

    Attribution used to recompute `hand_metrics` for every track on every
    contact, which is O(contacts x tracks) passes over the same rows.
    """

    def __init__(self, tracks: pd.DataFrame, rally, fps: float):
        f0, f1 = int(rally.start * fps), int(rally.end * fps)
        self.fps = fps
        self.seg = tracks[(tracks["frame"] >= f0) & (tracks["frame"] <= f1)]
        self.by_id = {int(tid): g.sort_values("frame").reset_index(drop=True)
                      for tid, g in self.seg.groupby("track_id")}
        self.hands = {tid: hand_metrics(g, fps) for tid, g in self.by_id.items()}

    def near(self, t: float, window_s: float = HAND_WINDOW_S) -> list[int]:
        return [tid for tid, g in self.by_id.items()
                if ((g["frame"] / self.fps - t).abs() <= window_s).sum() >= 2]

    def peak_hand_speed(self, tid: int, t: float,
                        window_s: float = HAND_WINDOW_S) -> float:
        hm = self.hands.get(tid)
        if hm is None or hm.empty:
            return float("nan")
        win = hm[(hm["t"] >= t - window_s) & (hm["t"] <= t + window_s)]
        if win.empty:
            return float("nan")
        speeds = win["hand_speed"].to_numpy()
        # a fully occluded player has no measurable hand speed at all; that is
        # an ordinary outcome here, not something to warn about
        if not np.isfinite(speeds).any():
            return float("nan")
        return float(np.nanmax(speeds))

    def court_at(self, tid: int, t: float, mapper):
        g = self.by_id.get(tid)
        if g is None or g.empty:
            return None
        i = (g["frame"] / self.fps - t).abs().to_numpy().argmin()
        return mapper.to_court(feet_px(g.iloc[[i]]), g["frame"].iloc[[i]])[0]


def attribute_contact(t: float, ctx: "RallyTracks", mapper,
                      prev_court=None, prev_t: float | None = None
                      ) -> tuple[int | None, float]:
    """Who made the touch heard at time `t`, and how sure we are.

    Hand speed alone is not enough. With twelve players someone is always
    swinging, and a player on the far side of the court could win the vote for a
    touch they could not physically have reached — which matters more than it
    sounds, because `grammar.decode` treats the attributed SIDE as observed fact
    when it works out possession. One bad attribution corrupts the whole rally.

    So candidates are first filtered by whether the ball could have got to them
    at all: at 25 m/s it can only cover so much ground in the time since the
    last touch. That needs no ball detection, only the previous contact.
    """
    candidates = ctx.near(t)
    if not candidates:
        return None, 0.0

    reach = None
    if prev_court is not None and prev_t is not None:
        reach = BALL_MAX_SPEED * max(t - prev_t, 0.0) + REACH_SLACK_M

    scored: list[tuple[float, int]] = []
    for tid in candidates:
        speed = ctx.peak_hand_speed(tid, t)
        if not np.isfinite(speed):
            continue
        score = speed
        if reach is not None:
            pos = ctx.court_at(tid, t, mapper)
            # NaN means the camera solver could not place this frame, so there is
            # no court position to reason about. Skipping is the honest move: a
            # NaN sails through `travelled > reach` as False and would otherwise
            # be silently treated as reachable.
            if pos is None or not np.isfinite(pos).all():
                continue
            travelled = float(np.hypot(*(pos - prev_court)))
            if travelled > reach:
                continue      # the ball could not have got there in time
            # among reachable players, prefer the one the ball had time to reach
            # comfortably rather than only just
            score *= 1.0 + 0.5 * (1.0 - travelled / reach)
        scored.append((score, tid))

    if not scored:
        return None, 0.0
    scored.sort(reverse=True)
    best_score, best_id = scored[0]
    if best_score <= 0:
        return None, 0.0
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    margin = 1.0 - (runner_up / best_score if best_score else 1.0)
    confidence = round(min(1.0, 0.35 + 0.65 * margin), 3)
    return best_id, confidence


def contact_features(t: float, strength: float, ctx: "RallyTracks", mapper,
                     prev_court=None, prev_t: float | None = None
                     ) -> ContactFeatures | None:
    """Full feature row for one contact, or None if nobody can be attributed."""
    tid, attribution = attribute_contact(t, ctx, mapper, prev_court, prev_t)
    if tid is None:
        return None
    g = ctx.by_id.get(tid)
    if g is None or g.empty:
        return None

    fps = ctx.fps
    frame = int(round(t * fps))
    near = g.iloc[(g["frame"] - frame).abs().to_numpy().argmin()]
    court = mapper.to_court(feet_px(pd.DataFrame([near])),
                            [int(near["frame"])])[0]
    # An unsolved camera frame gives NaN. Every court-derived feature below would
    # be meaningless, and `side_of(nan)` in particular returns "near" without
    # complaint — which would hand the rally decoder a fabricated possession.
    if not np.isfinite(court).all():
        return None

    hm = ctx.hands[tid]
    at = hm.iloc[(hm["frame"] - frame).abs().to_numpy().argmin()]
    air = airborne_series(g)
    air_at = float(air[(g["frame"] - frame).abs().to_numpy().argmin()])

    # peak values across the touch window describe the action better than the
    # single nearest frame, which can land between the swing and the contact
    win = hm[(hm["t"] >= t - HAND_WINDOW_S) & (hm["t"] <= t + HAND_WINDOW_S)]
    speed = float(np.nanmax(win["hand_speed"])) if len(win) else np.nan
    height = float(np.nanmax(win["hand_height"])) if len(win) else float(at["hand_height"])

    present = [v for v in (height, at["hand_spread"], air_at, speed)
               if np.isfinite(v)]
    pose_seen = len(present) / 4.0
    # how much of the pose was visible AND how sure we are it is the right
    # player — a perfectly clear pose belonging to the wrong body is worse than
    # useless, so the two multiply rather than average
    confidence = round(pose_seen * attribution, 4)

    return ContactFeatures(
        t=float(t), frame=frame, track_id=tid,
        side=side_of(float(court[1])),
        x=float(court[0]), y=float(court[1]),
        zone=zone_for(float(court[0]), float(court[1])),
        net_dist=dist_from_net(float(court[1])),
        strength=float(strength),
        hand_height=_nan_to(height, 0.0),
        hand_spread=_nan_to(float(at["hand_spread"]), 0.5),
        airborne=_nan_to(air_at, 0.0),
        hand_speed=_nan_to(speed, 0.0),
        confidence=confidence,
    )


def rally_features(rally, strengths: dict, tracks: pd.DataFrame, mapper,
                   fps: float) -> list[ContactFeatures]:
    """Every attributable contact in one rally, in order.

    `mapper` is anything with a `to_court(points, frames)` method — a bare
    `CourtCalibration` or a `CourtMapper`, whichever the session needs.

    Each contact is attributed using the previous one's court position, so the
    ball-flight constraint has something to work from. The chain starts
    unconstrained: the serve has no predecessor, and its own geometry (behind
    the endline) identifies it anyway.
    """
    ctx = RallyTracks(tracks, rally, fps)
    out: list[ContactFeatures] = []
    prev_court, prev_t = None, None
    for t in rally.contacts:
        f = contact_features(t, strengths.get(round(t, 3), 0.5), ctx, mapper,
                             prev_court, prev_t)
        if f is None:
            continue
        out.append(f)
        prev_court, prev_t = np.array([f.x, f.y]), f.t
    return out


def _nan_to(v: float, default: float) -> float:
    return float(v) if np.isfinite(v) else default


def is_airborne(f: ContactFeatures) -> bool:
    return f.airborne >= JUMP_THRESHOLD


def behind_endline(f: ContactFeatures) -> bool:
    """True when the contact happened past the player's own endline — the
    geometric signature of a serve."""
    return f.y < 0.0 or f.y > 2 * NET_Y


def to_frame(features: list[ContactFeatures]) -> pd.DataFrame:
    cols = list(ContactFeatures.__annotations__)
    return pd.DataFrame([asdict(f) for f in features], columns=cols)
