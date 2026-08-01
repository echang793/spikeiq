"""Jump height from ankle keypoints.

Measuring a vertical distance from a single camera needs a vertical ruler, and
the court homography is not one: it maps the *ground plane*, so converting a
vertical pixel rise with it overstates height by a factor that depends on the
camera tilt. The player's own standing height is a far better ruler — it is
genuinely vertical, and because it is measured in pixels at the player's own
court depth it self-corrects for perspective as they move up and down the court.

So: rise in pixels x (real standing height in metres / standing height in
pixels). The one input required from the user is their height, which they know
exactly; everything else is measured.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from contacts import _mean_xy

DEFAULT_HEIGHT_M = 1.80     # only a placeholder; sessions should set the real one
MIN_JUMP_M = 0.15           # below this it is a hop, a step, or keypoint jitter
MIN_SEPARATION_S = 0.5      # two jumps cannot peak closer together than this
GROUNDED_PCTL = 60          # ankle percentile taken to be "standing on the floor"


@dataclass
class Jump:
    t: float                # time of the peak
    height_m: float
    takeoff_t: float
    track_id: int

    def as_dict(self) -> dict:
        return {"t": round(self.t, 3), "height_m": round(self.height_m, 3),
                "takeoff_t": round(self.takeoff_t, 3), "track_id": self.track_id}


LOCAL_WINDOW_S = 2.5   # both the floor line and the pixel ruler are local to this


def standing_height_px(df: pd.DataFrame) -> float:
    """Bounding-box height in pixels while the player is on the floor.

    Taken from the *tallest* boxes rather than the median: a crouched defender
    or a partly-occluded box would read as a shorter person and inflate every
    jump measured against it.
    """
    h = (df["y2"] - df["y1"]).to_numpy(dtype=float)
    h = h[np.isfinite(h) & (h > 0)]
    if len(h) == 0:
        return float("nan")
    return float(np.percentile(h, 90))


def ankle_series(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Ankle y in pixels over time, missing detections left as NaN."""
    df = df.sort_values("frame").reset_index(drop=True)
    ank = _mean_xy(df, "lank", "rank")
    return pd.DataFrame({
        "frame": df["frame"].to_numpy(),
        "t": df["frame"].to_numpy() / fps,
        "y": ank[:, 1],
    })


def _window_rows(s: pd.DataFrame, fps: float) -> int:
    """How many rows LOCAL_WINDOW_S covers, given the tracking stride."""
    if len(s) < 2:
        return 1
    step = float(np.median(np.diff(s["frame"].to_numpy()))) or 1.0
    return max(3, int(round(LOCAL_WINDOW_S * fps / step)))


def _upper_envelope(t: np.ndarray, v: np.ndarray, win: int) -> np.ndarray:
    """A line that locally hugs the TOP of a noisy series.

    A rolling quantile is not good enough for this. While a player runs from the
    net to the endline their ankle y climbs the frame at a rate comparable to a
    jump, so any quantile taken over a window that is long enough to contain
    standing frames is dragged badly by the trend inside that window.

    So the trend is removed instead of averaged over: fit a straight line to the
    window, discard the samples sitting above it (in image terms, the airborne
    ones) and refit on what is left. What comes back is the line the grounded
    samples lie on, which is the floor even when the floor is moving.

    Used twice — for the ankle floor and for the standing-height ruler — since
    both are "the value when he is upright and on the ground, right here".
    """
    n = len(v)
    out = np.full(n, np.nan)
    half = max(2, win // 2)
    ok = np.isfinite(v)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        idx = np.flatnonzero(ok[lo:hi]) + lo
        if len(idx) < 3:
            continue
        tt, vv = t[idx], v[idx]
        fit = np.polyfit(tt, vv, 1)
        resid = vv - np.polyval(fit, tt)
        # keep the samples at or below the trend line: those are the grounded
        # ones. Refit only if enough of them survive to define a line.
        keep = resid >= 0
        if keep.sum() >= 3 and keep.sum() < len(idx):
            fit = np.polyfit(tt[keep], vv[keep], 1)
        out[i] = np.polyval(fit, t[i])
    return out


def detect_jumps(df: pd.DataFrame, fps: float,
                 subject_height_m: float = DEFAULT_HEIGHT_M) -> list[Jump]:
    """Every jump in one player's track, in chronological order.

    Both the floor line and the pixel-to-metre ruler are computed LOCALLY, over
    a rolling window rather than the whole track. That is not a refinement, it
    is the difference between right and wrong: a player at the far endline is a
    third the pixel height of the same player at the net, and their ankles sit
    a couple of hundred pixels higher in the frame. Measured against
    whole-track constants, simply walking up the court reads as a huge jump —
    which is exactly what showed up on the first real clip.
    """
    if df.empty:
        return []
    df = df.sort_values("frame").reset_index(drop=True)
    track_id = int(df["track_id"].iloc[0])

    s = ankle_series(df, fps)
    y = s["y"].to_numpy()
    if np.isnan(y).all():
        return []

    win = _window_rows(s, fps)
    t = s["t"].to_numpy()

    # the local ruler: how tall this player is in pixels right here on the court
    box_h = (df["y2"] - df["y1"]).to_numpy(dtype=float)
    box_h = np.where(np.isfinite(box_h) & (box_h > 0), box_h, np.nan)
    local_px = _upper_envelope(t, box_h, win)
    fallback = standing_height_px(df)
    local_px = np.where(np.isfinite(local_px) & (local_px > 0), local_px, fallback)

    # the local floor: where his ankles sit when he is standing, right here.
    # Image y grows downward, so "on the floor" is the LARGE-y side.
    floor = _upper_envelope(t, y, win)
    floor = np.where(np.isfinite(floor), floor,
                     np.nanpercentile(y, GROUNDED_PCTL))

    with np.errstate(invalid="ignore"):
        rise_m = (floor - y) * (subject_height_m / local_px)

    jumps: list[Jump] = []
    for i in _local_peaks(rise_m, min_value=MIN_JUMP_M):
        span = _airborne_span(rise_m, i)
        if span is None or not _hang_time_is_physical(rise_m[i], span, fps):
            continue
        if _has_discontinuity(rise_m, t, span):
            continue
        t_peak = float(s["t"].iloc[i])
        if jumps and t_peak - jumps[-1].t < MIN_SEPARATION_S:
            if rise_m[i] <= jumps[-1].height_m:
                continue
            jumps.pop()
        jumps.append(Jump(t=t_peak, height_m=float(rise_m[i]),
                          takeoff_t=float(s["t"].iloc[span[0]]), track_id=track_id))
    return jumps


GRAVITY = 9.81
HANG_TOLERANCE = (0.45, 2.2)   # measured / expected hang time must fall in here
GROUND_FRACTION = 0.25         # back down to this share of peak = on the floor
MAX_AIR_S = 1.4


def _airborne_span(rise: np.ndarray, peak_i: int) -> tuple[int, int] | None:
    """Frames from takeoff to landing around a peak, or None if he never lands.

    The "never lands" case is the important one: when the subject's stitched
    chain switches to a different tracker id, the ankle baseline steps to a new
    level and stays there. That looks exactly like a jump at the step, except
    the player never comes back down — so requiring a landing is what separates
    a real jump from a tracking discontinuity.
    """
    threshold = rise[peak_i] * GROUND_FRACTION
    start = peak_i
    while start > 0 and np.isfinite(rise[start]) and rise[start] > threshold:
        start -= 1
    if start == 0 and np.isfinite(rise[0]) and rise[0] > threshold:
        return None                      # already airborne at the first frame
    end = peak_i
    n = len(rise)
    while end < n - 1 and np.isfinite(rise[end]) and rise[end] > threshold:
        end += 1
    if end >= n - 1 and np.isfinite(rise[-1]) and rise[-1] > threshold:
        return None                      # never lands
    return start, end


MAX_VERTICAL_MPS = 4.0   # ankle vertical speed peaks around 2-3 m/s in a jump


def _has_discontinuity(rise: np.ndarray, t: np.ndarray,
                       span: tuple[int, int]) -> bool:
    """True when the "jump" contains a step no body could make.

    When the subject's stitched chain switches to a different tracker id, the
    ankle position moves half a metre between two adjacent frames. A real
    take-off is fast but not that fast, so a vertical speed well above anything
    human is the signature of a tracking discontinuity rather than a jump.
    """
    lo, hi = max(0, span[0] - 1), min(len(rise) - 1, span[1] + 1)
    seg, ts = rise[lo:hi + 1], t[lo:hi + 1]
    if len(seg) < 2:
        return False
    dt = np.diff(ts)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.abs(np.diff(seg)) / np.where(dt > 0, dt, np.nan)
    return bool(np.nanmax(speed, initial=0.0) > MAX_VERTICAL_MPS)


def _hang_time_is_physical(height_m: float, span: tuple[int, int],
                           fps: float) -> bool:
    """Check the hang time against the height the way gravity would.

    A body that rises h metres is airborne for about 2*sqrt(2h/g) seconds, so
    height and hang time are not independent measurements — they are one
    measurement made twice. Tracker noise satisfies neither relation, and the
    tolerance is wide because at stride 2 on 30 fps footage a jump is only a
    handful of samples.
    """
    air_s = (span[1] - span[0]) / fps
    if air_s <= 0 or air_s > MAX_AIR_S:
        return False
    expected = 2.0 * np.sqrt(2.0 * max(height_m, 1e-3) / GRAVITY)
    lo, hi = HANG_TOLERANCE
    return lo * expected <= air_s <= hi * expected


def _local_peaks(v: np.ndarray, min_value: float) -> list[int]:
    out = []
    for i in range(1, len(v) - 1):
        if not np.isfinite(v[i]) or v[i] < min_value:
            continue
        left = v[i - 1] if np.isfinite(v[i - 1]) else -np.inf
        right = v[i + 1] if np.isfinite(v[i + 1]) else -np.inf
        if v[i] >= left and v[i] >= right:
            out.append(i)
    return out


def detect_jumps_per_rally(tracks: pd.DataFrame, rallies,
                           ids_by_rally: dict[int, set[int]], fps: float,
                           subject_height_m: float = DEFAULT_HEIGHT_M
                           ) -> list[Jump]:
    """Jumps across a whole match, measured one rally at a time.

    Run over the concatenated match instead, the local baseline fit would span
    the dead ball between rallies — where the player has walked somewhere else
    entirely — and manufacture jumps out of the discontinuity. Each rally is its
    own measurement.
    """
    out: list[Jump] = []
    for rally in rallies:
        ids = ids_by_rally.get(rally.index, set())
        if not ids:
            continue
        f0, f1 = int(rally.start * fps), int(rally.end * fps)
        seg = tracks[(tracks["track_id"].isin(ids))
                     & (tracks["frame"] >= f0) & (tracks["frame"] <= f1)]
        if seg.empty:
            continue
        out.extend(detect_jumps(seg, fps, subject_height_m))
    return sorted(out, key=lambda j: j.t)


def jump_at(jumps: list[Jump], t: float, window_s: float = 0.45) -> Jump | None:
    """The jump whose peak is nearest a contact time, if there is one close by.

    Used to attach an approach-jump height to an attack and a block-jump height
    to a block; a contact with no jump near it simply has no height, which is
    the correct answer for a standing set or a floor dig.
    """
    best, best_dt = None, window_s
    for j in jumps:
        dt = abs(j.t - t)
        if dt <= best_dt:
            best, best_dt = j, dt
    return best


def summarise(jumps: list[Jump]) -> dict:
    if not jumps:
        return {"count": 0}
    h = np.array([j.height_m for j in jumps])
    return {
        "count": len(jumps),
        "best_m": round(float(h.max()), 2),
        "median_m": round(float(np.median(h)), 2),
        "mean_m": round(float(h.mean()), 2),
    }


def jumps_to_json(jumps: list[Jump]) -> list[dict]:
    return [j.as_dict() for j in jumps]
