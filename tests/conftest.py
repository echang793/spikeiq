import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# A plausible elevated-corner framing: the far endline is shorter and higher in
# frame than the near one, which is exactly the perspective every real session
# has and the reason the homography is not a similarity transform.
FAR_LEFT = [520.0, 300.0]
FAR_RIGHT = [1180.0, 300.0]
NEAR_RIGHT = [1600.0, 940.0]
NEAR_LEFT = [180.0, 940.0]


@pytest.fixture
def corners_px() -> list[list[float]]:
    return [FAR_LEFT, FAR_RIGHT, NEAR_RIGHT, NEAR_LEFT]


@pytest.fixture
def calib(corners_px):
    from court import CourtCalibration
    return CourtCalibration(corners_px)


def tracks_frame(frame: int, track_id: int, cx: float, cy: float,
                 h: float = 120.0, w: float = 50.0, conf: float = 0.9) -> dict:
    """One synthetic track row with all keypoint columns filled in."""
    from tracking import KEYPOINTS
    row = {
        "frame": frame, "track_id": track_id,
        "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2,
        "conf": conf,
    }
    for name in KEYPOINTS:
        row[f"{name}x"] = cx
        row[f"{name}y"] = cy
    return row


@pytest.fixture
def make_tracks():
    import pandas as pd

    def _make(rows: list[dict]):
        from tracking import COLUMNS
        return pd.DataFrame(rows, columns=COLUMNS)

    return _make


def px_for(calib, x: float, y: float):
    """Court metres -> image pixels, by inverting the calibration homography."""
    import cv2
    import numpy as np
    pt = np.array([[[x, y]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, np.linalg.inv(calib.H)).reshape(2)


@pytest.fixture
def make_windowed_tracks(calib):
    """Tracks with the shape `run_tracking` actually writes: frames ONLY inside
    rally windows, so consecutive rows straddle hundreds of frames of dead ball.

    Every fixture in the first version of this suite used contiguous frames, and
    that is exactly why a bug that truncated the subject to rally one shipped
    under a green suite. Anything that consumes `tracks.parquet` should be
    tested against this shape, not a continuous one.

    `positions` maps rally index -> {track_id: (court_x, court_y)}.
    """
    import pandas as pd

    def _make(rallies, positions: dict, fps: float = 30.0, stride: int = 2):
        from tracking import COLUMNS
        rows = []
        for rally in rallies:
            per_track = positions.get(rally.index, {})
            f0, f1 = int(rally.start * fps), int(rally.end * fps)
            for frame in range(f0 - f0 % stride, f1 + 1, stride):
                for tid, (cx, cy) in per_track.items():
                    px, py = px_for(calib, cx, cy)
                    rows.append(tracks_frame(frame, tid, float(px),
                                             float(py) - 60.0, h=120.0))
        return pd.DataFrame(rows, columns=COLUMNS)

    return _make


@pytest.fixture
def match_rallies():
    """Four rallies spread over a realistic match timeline, with the long dead
    ball between them that breaks naive frame-gap stitching."""
    from rallies import Rally
    return [Rally(index=i, start=10.0 + i * 30.0, end=18.0 + i * 30.0,
                  serving_side="near" if i % 2 == 0 else "far")
            for i in range(4)]


@pytest.fixture
def synth_audio(tmp_path):
    """Write a WAV with narrowband whistle tones and broadband contact pops.

    Returns (path, whistle_times, contact_times) so tests can assert against
    ground truth rather than against whatever the detector happens to output.
    """
    import soundfile as sf

    sr = 22050
    dur = 30.0
    t = np.arange(int(sr * dur)) / sr
    rng = np.random.default_rng(0)
    y = 0.004 * rng.standard_normal(len(t))  # gym noise floor

    whistles = [(2.0, 2.35), (12.0, 12.4)]
    for w0, w1 in whistles:
        seg = (t >= w0) & (t < w1)
        y[seg] += 0.5 * np.sin(2 * np.pi * 3000 * t[seg])

    contacts = [4.0, 5.2, 6.1, 7.5, 14.0, 15.3, 16.4]
    for c in contacts:
        i = int(c * sr)
        n = int(0.012 * sr)
        env = np.exp(-np.linspace(0, 7, n))
        y[i:i + n] += 0.6 * env * rng.standard_normal(n)

    path = tmp_path / "audio.wav"
    sf.write(path, y, sr)
    return path, [w for w, _ in whistles], contacts
