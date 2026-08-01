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


def attribute_contact(t: float, tracks: pd.DataFrame, fps: float,
                      window_s: float = HAND_WINDOW_S) -> int | None:
    """Which track made the touch heard at time `t`.

    Chosen as the player whose hands are moving fastest around the sound. That
    is the same corroboration idea dinkiq used, but where dinkiq only needed to
    ask "did the subject swing?", here it has to pick one of twelve — so the
    answer is a comparison between players, and ties or a silent field return
    None rather than a guess.
    """
    f_lo = int((t - window_s) * fps)
    f_hi = int((t + window_s) * fps)
    win = tracks[(tracks["frame"] >= f_lo) & (tracks["frame"] <= f_hi)]
    if win.empty:
        return None
    best_id, best_speed = None, 0.0
    for tid, g in win.groupby("track_id"):
        if len(g) < 2:
            continue
        hm = hand_metrics(g, fps)
        peak = np.nanmax(hm["hand_speed"].to_numpy()) if len(hm) else np.nan
        if np.isfinite(peak) and peak > best_speed:
            best_id, best_speed = int(tid), float(peak)
    return best_id


def contact_features(t: float, strength: float, tracks: pd.DataFrame,
                     calib, fps: float) -> ContactFeatures | None:
    """Full feature row for one contact, or None if no player can be attributed."""
    tid = attribute_contact(t, tracks, fps)
    if tid is None:
        return None
    g = tracks[tracks["track_id"] == tid].sort_values("frame")
    if g.empty:
        return None

    frame = int(round(t * fps))
    near = g.iloc[(g["frame"] - frame).abs().to_numpy().argmin()]
    court = calib.to_court(feet_px(pd.DataFrame([near])))[0]

    hm = hand_metrics(g, fps)
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
    confidence = len(present) / 4.0

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
