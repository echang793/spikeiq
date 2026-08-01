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


def detect_jumps(df: pd.DataFrame, fps: float,
                 subject_height_m: float = DEFAULT_HEIGHT_M) -> list[Jump]:
    """Every jump in one player's track, tallest first is NOT assumed — order
    is chronological so callers can align jumps with contacts."""
    if df.empty:
        return []
    track_id = int(df["track_id"].iloc[0])
    px_per_m = standing_height_px(df)
    if not np.isfinite(px_per_m) or px_per_m <= 0:
        return []
    scale = subject_height_m / px_per_m

    s = ankle_series(df, fps)
    y = s["y"].to_numpy()
    if np.isnan(y).all():
        return []
    # the floor line: where the ankles sit most of the time. A high percentile
    # because image y grows downward, so "on the floor" is a LARGE y.
    floor = float(np.nanpercentile(y, GROUNDED_PCTL))
    rise_m = (floor - y) * scale

    jumps: list[Jump] = []
    for i in _local_peaks(rise_m, min_value=MIN_JUMP_M):
        t_peak = float(s["t"].iloc[i])
        if jumps and t_peak - jumps[-1].t < MIN_SEPARATION_S:
            if rise_m[i] <= jumps[-1].height_m:
                continue
            jumps.pop()
        jumps.append(Jump(t=t_peak, height_m=float(rise_m[i]),
                          takeoff_t=_takeoff_time(s, rise_m, i), track_id=track_id))
    return jumps


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


def _takeoff_time(s: pd.DataFrame, rise: np.ndarray, peak_i: int) -> float:
    """Walk back from the peak to the last frame the player was on the floor."""
    i = peak_i
    while i > 0 and np.isfinite(rise[i]) and rise[i] > MIN_JUMP_M / 3:
        i -= 1
    return float(s["t"].iloc[i])


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
