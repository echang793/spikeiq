"""Camera motion: mapping each frame back onto the calibration frame.

The court homography is fitted once, on one frame. If the camera moves, every
court coordinate derived from a later frame is wrong — and wrong quietly, with no
error anywhere. This module measures the motion so `court.CourtMapper` can undo
it: `court = H_ref · W_t · p`, where `W_t` warps frame *t* into the reference.

Design choices worth keeping:

- **Every frame solves directly against the reference frame.** The obvious
  alternative, chaining frame-to-frame, accumulates drift over thousands of
  frames and needs loop closure to fix. Footage that stays roughly on one view
  always overlaps the reference, so there is nothing to gain from chaining and a
  whole class of error to avoid.
- **Players are masked out.** Twelve moving bodies are a large share of the
  matched features, and they move coherently, so an unmasked solve drifts toward
  the players' motion rather than the court's. `run_tracking` already hands the
  player boxes to `frame_cb`, so the masks cost nothing.
- **Failure is reported, not smoothed over.** A frame that cannot be solved gets
  no homography, and `CourtMapper` turns that into NaN, which every downstream
  filter already drops. Reusing a stale warp would be the one genuinely dangerous
  option.
- **Fixed cameras pay nothing.** If the measured motion never exceeds a few
  pixels the session is classified `fixed` and the whole mechanism is skipped.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ORB on a downscaled frame: the solve only needs the coarse geometry of the
# background, and full resolution costs several times as much for no benefit.
WORK_WIDTH = 720
N_FEATURES = 1200
MIN_MATCHES = 12          # below this a RANSAC homography is not worth trusting
MIN_INLIERS = 10
RANSAC_PX = 3.0
BOX_PAD = 8               # grow player boxes slightly; feature points cling to edges

# A session whose corner motion never exceeds this is treated as a fixed camera.
FIXED_PX = 2.5
# Beyond this the footage is panning further than a solve-against-reference design
# can honestly cover, and says so rather than approximating.
LARGE_MOTION_PX = 260.0

REGIMES = ("fixed", "handheld", "handheld_with_cuts", "unsolved")


@dataclass
class CameraTrack:
    """Per-frame warp back to the reference frame, plus how much to believe it."""

    warps: dict[int, np.ndarray] = field(default_factory=dict)
    confidence: dict[int, float] = field(default_factory=dict)
    motion_px: dict[int, float] = field(default_factory=dict)
    cuts: list[int] = field(default_factory=list)
    reference_frame: int = 0
    is_identity: bool = False

    @classmethod
    def identity(cls) -> "CameraTrack":
        """A fixed camera. `CourtMapper` short-circuits on this."""
        return cls(is_identity=True)

    def warp_for(self, frame: int) -> np.ndarray | None:
        if self.is_identity:
            return np.eye(3)
        return self.warps.get(int(frame))

    def confidence_for(self, frame: int) -> float:
        if self.is_identity:
            return 1.0
        return self.confidence.get(int(frame), 0.0)

    @property
    def solved_fraction(self) -> float:
        total = len(self.confidence)
        if self.is_identity:
            return 1.0
        if not total:
            return 0.0
        return len(self.warps) / total

    @property
    def max_motion_px(self) -> float:
        return max(self.motion_px.values(), default=0.0)

    def regime(self) -> str:
        """What kind of footage this turned out to be."""
        if self.is_identity:
            return "fixed"
        if not self.warps:
            return "unsolved"
        if self.cuts:
            return "handheld_with_cuts"
        if self.max_motion_px <= FIXED_PX:
            return "fixed"
        return "handheld"

    def summary(self) -> dict:
        return {
            "regime": self.regime(),
            "reference_frame": self.reference_frame,
            "frames_attempted": len(self.confidence),
            "frames_solved": len(self.warps),
            "solved_fraction": round(self.solved_fraction, 3),
            "max_motion_px": round(self.max_motion_px, 1),
            "median_motion_px": round(
                float(np.median(list(self.motion_px.values()))), 1)
            if self.motion_px else 0.0,
            "cuts": self.cuts,
            "beyond_supported_motion": self.max_motion_px > LARGE_MOTION_PX,
        }

    # --- persistence --------------------------------------------------------
    # Independent of the court fit, so this belongs in the cached tier next to
    # tracks.parquet: re-calibrating the court must not mean re-solving the camera.
    def to_frame(self) -> pd.DataFrame:
        rows = []
        for frame in sorted(self.confidence):
            warp = self.warps.get(frame)
            flat = (warp.reshape(-1).tolist() if warp is not None
                    else [np.nan] * 9)
            rows.append([frame, self.confidence[frame],
                         self.motion_px.get(frame, np.nan)] + flat)
        cols = (["frame", "confidence", "motion_px"]
                + [f"h{i}" for i in range(9)])
        return pd.DataFrame(rows, columns=cols)

    def save(self, path: Path) -> None:
        self.to_frame().to_parquet(path, index=False)
        path.with_suffix(".json").write_text(json.dumps(self.summary(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "CameraTrack":
        df = pd.read_parquet(path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        track = cls(reference_frame=int(meta.get("reference_frame", 0)),
                    cuts=list(meta.get("cuts", [])))
        for row in df.itertuples(index=False):
            frame = int(row.frame)
            track.confidence[frame] = float(row.confidence)
            track.motion_px[frame] = float(row.motion_px)
            flat = np.array([getattr(row, f"h{i}") for i in range(9)], dtype=float)
            if np.isfinite(flat).all():
                track.warps[frame] = flat.reshape(3, 3)
        if meta.get("regime") == "fixed" and not track.warps:
            track.is_identity = True
        return track


def _prepare(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Grey, downscaled working image plus the scale it was reduced by."""
    h, w = frame.shape[:2]
    scale = WORK_WIDTH / float(w) if w > WORK_WIDTH else 1.0
    if scale != 1.0:
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(grey), scale


def _player_mask(shape, boxes, scale: float) -> np.ndarray:
    """255 where the solver may look — everywhere except the players."""
    mask = np.full(shape[:2], 255, dtype=np.uint8)
    if boxes is None:
        return mask
    h, w = shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        a = max(0, int(x1 * scale) - BOX_PAD)
        b = max(0, int(y1 * scale) - BOX_PAD)
        c = min(w, int(x2 * scale) + BOX_PAD)
        d = min(h, int(y2 * scale) + BOX_PAD)
        if c > a and d > b:
            mask[b:d, a:c] = 0
    return mask


class CameraSolver:
    """Feeds off `run_tracking`'s `frame_cb`, so it rides the same decode.

    Usage mirrors dinkiq's CutDetector: construct with the reference frame, pass
    every processed frame to `update`, then read `track`.
    """

    def __init__(self, reference: np.ndarray, reference_frame: int = 0,
                 detect_cuts: bool = True):
        self.orb = cv2.ORB_create(nfeatures=N_FEATURES)
        self.ref_grey, self.scale = _prepare(reference)
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(self.ref_grey, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.track = CameraTrack(reference_frame=reference_frame)
        self.corners = _frame_corners(self.ref_grey.shape)
        self._cutter = _CutDetector() if detect_cuts else None

    def update(self, frame_idx: int, frame: np.ndarray, boxes=None) -> None:
        if self._cutter is not None and self._cutter.update(frame_idx, frame):
            self.track.cuts.append(frame_idx)

        grey, scale = _prepare(frame)
        warp, conf = self._solve(grey, scale, frame.shape, boxes)
        self.track.confidence[frame_idx] = conf
        if warp is None:
            return
        motion = _corner_motion(warp, self.corners, self.scale)
        # A warp claiming the camera moved further than the frame is wide is not
        # a warp worth keeping, however many inliers agreed. Seen on cropped
        # footage where most of the background texture was gone: a handful of
        # frames produced confident nonsense in the thousands of pixels. Dropping
        # them turns that into an honest unsolved frame instead of a flagged
        # absurdity carried through to the report.
        limit = _absurd_motion_limit(frame.shape)
        if not np.isfinite(motion) or motion > limit:
            self.track.confidence[frame_idx] = 0.0
            return
        self.track.warps[frame_idx] = warp
        self.track.motion_px[frame_idx] = motion

    def _solve(self, grey, scale, shape, boxes):
        if self.ref_desc is None or len(self.ref_kp) < MIN_MATCHES:
            return None, 0.0
        mask = _player_mask(grey.shape, boxes, scale)
        kp, desc = self.orb.detectAndCompute(grey, mask)
        if desc is None or len(kp) < MIN_MATCHES:
            return None, 0.0
        matches = self.matcher.match(desc, self.ref_desc)
        if len(matches) < MIN_MATCHES:
            return None, 0.0
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in matches]
                         ).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if H is None or inliers is None:
            return None, 0.0
        n_in = int(inliers.sum())
        if n_in < MIN_INLIERS:
            return None, 0.0
        # H maps working-resolution frame pixels to working-resolution reference
        # pixels; rescale so it operates on full-resolution coordinates, which is
        # what every caller actually has
        H_full = _rescale(H, src_scale=scale, dst_scale=self.scale)
        if not np.isfinite(H_full).all():
            return None, 0.0
        confidence = min(1.0, n_in / max(len(matches), 1))
        return H_full, round(confidence, 3)

    def finish(self) -> CameraTrack:
        """Collapse to an identity track when the camera never actually moved,
        so a fixed-tripod session carries no per-frame machinery at all.

        Never collapses when a cut was seen. A cut means the camera was
        physically moved or the recording restarted, and calling that "fixed"
        would apply the reference homography to an entirely different view — the
        exact silent failure this module exists to prevent. Likewise, unsolved
        frames must keep their per-frame handling so they can stay NaN.
        """
        if self.track.cuts:
            return self.track
        if not self.track.warps or len(self.track.warps) < len(self.track.confidence):
            return self.track
        if self.track.max_motion_px <= FIXED_PX:
            self.track.is_identity = True
        return self.track


def _rescale(H: np.ndarray, src_scale: float, dst_scale: float) -> np.ndarray:
    """Convert a homography between downscaled images into a full-resolution one."""
    S_src = np.diag([src_scale, src_scale, 1.0])
    S_dst = np.diag([dst_scale, dst_scale, 1.0])
    return np.linalg.inv(S_dst) @ H @ S_src


def _absurd_motion_limit(shape) -> float:
    """Motion beyond this cannot be a real camera still matching the reference."""
    h, w = shape[:2]
    return float(np.hypot(w, h))


def _frame_corners(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def _corner_motion(warp: np.ndarray, corners_work: np.ndarray,
                   scale: float) -> float:
    """How far the frame corners move under this warp, in full-res pixels.

    A single scalar for "how much has the camera shifted", used both to classify
    the regime and to notice footage that pans beyond what this design covers.
    """
    corners_full = corners_work / scale
    moved = cv2.perspectiveTransform(
        corners_full.reshape(-1, 1, 2).astype(np.float32), warp).reshape(-1, 2)
    return float(np.max(np.hypot(*(moved - corners_full).T)))


class _CutDetector:
    """HSV-histogram correlation between consecutive processed frames.

    Ported from dinkiq/src/ball.py, which built it for broadcast angle changes.
    Here it catches a stop/restart or the camera being physically repositioned,
    both of which invalidate the reference match rather than merely shifting it.
    """

    CORR_THRESHOLD = 0.6

    def __init__(self):
        self._prev = None

    def update(self, frame_idx: int, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self._prev is not None:
            corr = cv2.compareHist(self._prev, hist, cv2.HISTCMP_CORREL)
            cut = corr < self.CORR_THRESHOLD
        self._prev = hist
        return cut
