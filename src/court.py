"""Volleyball court geometry and pixel->court-coordinate homography.

Coordinate system (METRES): x in [0, 9] across the width, y in [0, 18] along the
length, net at y = 9. Attack ("3-metre") lines at y = 6 and y = 12.

Orientation is the camera's: x increases to the right of frame, y increases
towards the camera. The half at y < 9 is the FAR side, y > 9 the NEAR side.

Calibration points are clicked in one consistent order — far-left, far-right,
near-right, near-left (as seen from the camera) — first for the four outer court
corners, then for the four points where the attack lines meet the sidelines.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COURT_W = 9.0
COURT_L = 18.0
NET_Y = 9.0
ATTACK_LINE_OFFSET = 3.0  # attack line is 3 m from the net on each side

# Front row is between the net and the attack line; back row behind it.
FAR_ATTACK_Y = NET_Y - ATTACK_LINE_OFFSET    # 6.0
NEAR_ATTACK_Y = NET_Y + ATTACK_LINE_OFFSET   # 12.0

# Court-space targets for the four clicked outer corners, in click order:
# far-left=(0,0), far-right=(9,0), near-right=(9,18), near-left=(0,18)
CORNER_TARGETS = np.array(
    [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
    dtype=np.float32,
)
# Attack-line / sideline intersections, same click order:
# far-left=(0,6), far-right=(9,6), near-right=(9,12), near-left=(0,12)
ATTACK_TARGETS = np.array(
    [[0.0, FAR_ATTACK_Y], [COURT_W, FAR_ATTACK_Y],
     [COURT_W, NEAR_ATTACK_Y], [0.0, NEAR_ATTACK_Y]],
    dtype=np.float32,
)

# --- named landmarks -------------------------------------------------------
# Every point a user can be asked to identify, with its court coordinate. Four
# non-collinear ones are enough for a homography, so a session only needs
# whichever of these happen to be in frame — which is what makes a partly
# cropped court usable at all.
#
# Deliberately floor-plane only. The net top is 2.43 m up and is not on the
# plane this homography models. Net POST bases are on the floor but excluded
# too: the legal offset from the sideline is anywhere from 0.5 to 1.0 m, so
# assuming a value would inject a silent error into every position downstream.
LANDMARKS: dict[str, tuple[float, float]] = {
    "corner_far_left": (0.0, 0.0),
    "corner_far_right": (COURT_W, 0.0),
    "corner_near_right": (COURT_W, COURT_L),
    "corner_near_left": (0.0, COURT_L),
    "attack_far_left": (0.0, FAR_ATTACK_Y),
    "attack_far_right": (COURT_W, FAR_ATTACK_Y),
    "attack_near_right": (COURT_W, NEAR_ATTACK_Y),
    "attack_near_left": (0.0, NEAR_ATTACK_Y),
    "centre_left": (0.0, NET_Y),
    "centre_right": (COURT_W, NET_Y),
}

LANDMARK_LABELS: dict[str, str] = {
    "corner_far_left": "far endline, left sideline",
    "corner_far_right": "far endline, right sideline",
    "corner_near_right": "near endline, right sideline",
    "corner_near_left": "near endline, left sideline",
    "attack_far_left": "far attack line, left sideline",
    "attack_far_right": "far attack line, right sideline",
    "attack_near_right": "near attack line, right sideline",
    "attack_near_left": "near attack line, left sideline",
    "centre_left": "under the net, left sideline",
    "centre_right": "under the net, right sideline",
}

# The click order the fixed 8-point UI used, kept so stored calibrations and
# the old API shape keep working.
LEGACY_CORNER_ORDER = ["corner_far_left", "corner_far_right",
                       "corner_near_right", "corner_near_left"]
LEGACY_ATTACK_ORDER = ["attack_far_left", "attack_far_right",
                       "attack_near_right", "attack_near_left"]

MIN_LANDMARKS = 4

# The seven lines of the court model, as pairs of court-space endpoints. Used
# to render the model over a frame and (in courtfind) to score a hypothesis.
COURT_LINES: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((0.0, 0.0), (COURT_W, 0.0)),                    # far endline
    ((0.0, COURT_L), (COURT_W, COURT_L)),            # near endline
    ((0.0, 0.0), (0.0, COURT_L)),                    # left sideline
    ((COURT_W, 0.0), (COURT_W, COURT_L)),            # right sideline
    ((0.0, FAR_ATTACK_Y), (COURT_W, FAR_ATTACK_Y)),  # far attack line
    ((0.0, NEAR_ATTACK_Y), (COURT_W, NEAR_ATTACK_Y)),  # near attack line
    ((0.0, NET_Y), (COURT_W, NET_Y)),                # centre line
]


@dataclass
class CalibrationQuality:
    """How much the fit can be trusted, measured at fit time.

    Reprojection error alone is misleading here. With exactly four landmarks the
    fit is determined and the error is zero by construction, which says nothing
    about whether the court extrapolates sensibly beyond them — and extrapolation
    is the whole point when part of the court is out of frame. `corner_shift_m`
    is the honest measure: refit without each landmark in turn and see how far
    the court's own corners move.
    """
    n_landmarks: int
    reprojection_px: float
    corner_shift_m: float | None       # None when a leave-one-out refit is impossible
    hull_coverage: float               # share of the court inside the clicked hull
    region_extrapolation: dict[str, float]   # region -> distance outside hull, in hull widths
    convex: bool

    @property
    def determined_only(self) -> bool:
        """True when there are exactly the minimum landmarks, so reprojection
        error is zero for free and tells us nothing."""
        return self.n_landmarks <= MIN_LANDMARKS

    def trusted(self, region: str, limit: float = 0.35) -> bool:
        """Whether a court region was measured rather than guessed at."""
        return self.region_extrapolation.get(region, 1.0) <= limit

    def as_dict(self) -> dict:
        return {
            "n_landmarks": self.n_landmarks,
            "reprojection_px": round(self.reprojection_px, 2),
            "corner_shift_m": (None if self.corner_shift_m is None
                               else round(self.corner_shift_m, 3)),
            "hull_coverage": round(self.hull_coverage, 3),
            "region_extrapolation": {k: round(v, 3)
                                     for k, v in self.region_extrapolation.items()},
            "convex": self.convex,
            "determined_only": self.determined_only,
        }


# Court regions whose trust is tracked separately. A near-court hitting
# percentage is perfectly good with the far endline guessed; a far-court zone
# distribution is not, so they cannot share one verdict.
REGIONS: dict[str, tuple[float, float]] = {
    "far_court": (COURT_W / 2, COURT_L * 0.17),
    "far_attack": (COURT_W / 2, FAR_ATTACK_Y),
    "net": (COURT_W / 2, NET_Y),
    "near_attack": (COURT_W / 2, NEAR_ATTACK_Y),
    "near_court": (COURT_W / 2, COURT_L * 0.83),
}


class CourtCalibration:
    """Homography from identified court landmarks.

    Any four non-collinear landmarks from `LANDMARKS` will do; more are fitted by
    least squares and make the result more stable. Points near the net are worth
    the most, for the same reason dinkiq anchors on the kitchen line: endline
    corners are often distant or cropped, while the attack and centre lines sit
    where attacking and blocking metrics actually live.

    A homography extrapolates the whole court from whatever was given, so a
    partly cropped court needs no special machinery here — only the honesty of
    `quality` about how far that extrapolation is being trusted.
    """

    def __init__(self, landmarks: dict[str, list[float]] | list | None = None,
                 legacy_attack_px: list[list[float]] | None = None, *,
                 corners_px: list[list[float]] | None = None,
                 attack_px: list[list[float]] | None = None):
        """`landmarks` is a name -> pixel mapping.

        For compatibility it also accepts the old positional shape — a list of
        four corners, optionally followed by four attack-line points — because
        calibrations stored by the previous UI must keep loading. The two shapes
        are a dict and a list, so telling them apart is unambiguous.
        """
        if isinstance(landmarks, (list, tuple)):
            corners_px = list(landmarks)
            attack_px = attack_px if legacy_attack_px is None else legacy_attack_px
            landmarks = None
        if landmarks is None:
            landmarks = _from_legacy(corners_px, attack_px)
        unknown = set(landmarks) - set(LANDMARKS)
        if unknown:
            raise ValueError(f"unknown landmarks: {sorted(unknown)}")
        if len(landmarks) < MIN_LANDMARKS:
            raise ValueError(
                f"need at least {MIN_LANDMARKS} landmarks to locate the court, "
                f"got {len(landmarks)}")
        self.landmarks = {k: [float(v[0]), float(v[1])]
                          for k, v in landmarks.items()}
        self.H = _fit(self.landmarks)
        self.quality = _assess(self.landmarks, self.H)
        if not self.quality.convex:
            raise ValueError(
                "those points do not describe a court — the fitted court folds "
                "over on itself, so at least one landmark is misplaced")

    # legacy accessors, still used by court_px_width and older callers
    @property
    def corners_px(self) -> list[list[float]]:
        """The four court corners in image pixels, extrapolated if not clicked."""
        return [p.tolist() for p in self.court_to_px(CORNER_TARGETS)]

    @property
    def attack_px(self) -> list[list[float]]:
        return [p.tolist() for p in self.court_to_px(ATTACK_TARGETS)]

    def to_court(self, points_px: np.ndarray, frames=None) -> np.ndarray:
        """Map (N,2) pixel points to (N,2) court metres.

        `frames` is accepted and ignored so that a bare calibration and a
        `CourtMapper` present the same interface. A fixed camera has one
        homography for every frame, so there is nothing for it to do here — and
        callers should not have to know which regime they are in.
        """
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        if pts.size == 0:
            return np.zeros((0, 2))
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def court_to_px(self, points_m: np.ndarray) -> np.ndarray:
        """Inverse map: court metres back to image pixels. Used to render the
        court model over a frame, which is how a wrong fit becomes obvious."""
        pts = np.asarray(points_m, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def model_lines_px(self) -> list[list[list[float]]]:
        """The seven court lines in image pixels, for drawing over the frame."""
        out = []
        for a, b in COURT_LINES:
            pa, pb = self.court_to_px(np.array([a, b]))
            out.append([pa.tolist(), pb.tolist()])
        return out

    def metres_per_pixel(self, x_px: float, y_px: float) -> float:
        """Local vertical scale at a pixel: metres of court per pixel of image.

        Perspective makes this vary hugely between the near and far endline, so
        jump heights (which are measured in pixels of vertical displacement)
        must be converted using the scale *where the player is standing*, not a
        single frame-wide constant. Estimated by projecting a 1-pixel vertical
        step and measuring how far it moves in court space.
        """
        a, b = self.to_court(np.array([[x_px, y_px], [x_px, y_px - 1.0]]))
        return float(np.hypot(*(b - a)))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "version": 2,
            "landmarks": self.landmarks,
            "quality": self.quality.as_dict(),
        }, indent=1))

    @classmethod
    def load(cls, path: Path) -> "CourtCalibration":
        data = json.loads(path.read_text())
        if "landmarks" in data:
            return cls(data["landmarks"])
        # version 1: a fixed 4-or-8 point click order
        return cls(corners_px=data["corners_px"], attack_px=data.get("attack_px"))


class CourtMapper:
    """Pixels to court metres, for footage where the camera may be moving.

    The court homography is fitted once, on the calibration frame. When the
    camera drifts, a point in frame *t* must first be warped back into that
    reference frame, so the full mapping is `H_ref · W_t`. A fixed-camera session
    passes no camera track at all and gets `W_t = I`, which keeps one code path
    for both regimes instead of two that can disagree.

    Frames the camera solver could not solve come back as NaN rather than being
    mapped with a stale homography. That is deliberate: every consumer already
    filters through `on_court`, and NaN fails those comparisons, so an unsolved
    frame drops out instead of contributing a confident wrong position.
    """

    def __init__(self, calibration: CourtCalibration, camera=None):
        self.calibration = calibration
        self.camera = camera

    @property
    def moving(self) -> bool:
        return self.camera is not None and not self.camera.is_identity

    def to_court(self, points_px, frames=None) -> np.ndarray:
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return pts.reshape(0, 2)
        if frames is None or not self.moving:
            return self.calibration.to_court(pts)

        frames = np.asarray(frames).reshape(-1)
        if len(frames) != len(pts):
            raise ValueError("frames must line up with points")
        out = np.full((len(pts), 2), np.nan)
        for frame in np.unique(frames):
            mask = frames == frame
            warp = self.camera.warp_for(int(frame))
            if warp is None:
                continue      # unsolved: stays NaN, and gets filtered downstream
            H = self.calibration.H @ warp
            sub = cv2.perspectiveTransform(
                pts[mask].astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
            out[mask] = sub
        return out

    def metres_per_pixel(self, x_px: float, y_px: float) -> float:
        return self.calibration.metres_per_pixel(x_px, y_px)

    @property
    def quality(self) -> CalibrationQuality:
        return self.calibration.quality


def _from_legacy(corners_px, attack_px) -> dict[str, list[float]]:
    """Map the old fixed click order onto landmark names.

    Kept so calibrations stored by the previous UI keep working — a session
    should not have to be re-calibrated because the code learned a new trick.
    """
    if corners_px is None or len(corners_px) != 4:
        raise ValueError("need exactly 4 court corners")
    out = {name: list(map(float, pt))
           for name, pt in zip(LEGACY_CORNER_ORDER, corners_px)}
    if attack_px is not None:
        if len(attack_px) != 4:
            raise ValueError("attack_px must have exactly 4 points when given")
        out.update({name: list(map(float, pt))
                    for name, pt in zip(LEGACY_ATTACK_ORDER, attack_px)})
    return out


def _fit(landmarks: dict[str, list[float]]) -> np.ndarray:
    """Least-squares homography from image pixels to court metres."""
    names = sorted(landmarks)
    src = np.array([landmarks[n] for n in names], dtype=np.float32)
    dst = np.array([LANDMARKS[n] for n in names], dtype=np.float32)
    if len(names) == 4:
        H = cv2.getPerspectiveTransform(src, dst)
    else:
        H, _ = cv2.findHomography(src, dst, method=0)   # least squares
    if H is None or not np.isfinite(H).all():
        raise ValueError("degenerate calibration points")
    return H.astype(np.float64)


def _assess(landmarks: dict[str, list[float]], H: np.ndarray) -> CalibrationQuality:
    names = sorted(landmarks)
    src = np.array([landmarks[n] for n in names], dtype=np.float32)
    dst = np.array([LANDMARKS[n] for n in names], dtype=np.float64)
    got = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    # residuals are in metres; convert to a pixel-ish figure via the local scale
    # so the number means something to a human looking at a frame
    reproj_m = float(np.mean(np.hypot(*(got - dst).T)))

    corner_shift = _leave_one_out_shift(landmarks) if len(names) > 4 else None
    hull = _clicked_hull_m(landmarks, H)
    coverage, extrapolation = _region_trust(hull)
    return CalibrationQuality(
        n_landmarks=len(names),
        reprojection_px=reproj_m,
        corner_shift_m=corner_shift,
        hull_coverage=coverage,
        region_extrapolation=extrapolation,
        convex=_is_convex_court(H),
    )


def _leave_one_out_shift(landmarks: dict[str, list[float]]) -> float | None:
    """Worst movement of a court corner when any one landmark is dropped.

    This is the number that matters for a partial court: it measures how much
    the fit depends on each individual click, and therefore how much faith the
    extrapolated part of the court deserves.
    """
    base = _fit(landmarks)
    base_corners = cv2.perspectiveTransform(
        np.array(_court_corner_px(base), dtype=np.float32).reshape(-1, 1, 2),
        base).reshape(-1, 2)
    worst = 0.0
    for drop in landmarks:
        subset = {k: v for k, v in landmarks.items() if k != drop}
        if len(subset) < MIN_LANDMARKS:
            continue
        try:
            alt = _fit(subset)
        except ValueError:
            return None
        moved = cv2.perspectiveTransform(
            np.array(_court_corner_px(base), dtype=np.float32).reshape(-1, 1, 2),
            alt).reshape(-1, 2)
        worst = max(worst, float(np.max(np.hypot(*(moved - base_corners).T))))
    return worst


def _court_corner_px(H: np.ndarray) -> list[list[float]]:
    return cv2.perspectiveTransform(
        CORNER_TARGETS.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2).tolist()


def _clicked_hull_m(landmarks: dict[str, list[float]],
                    H: np.ndarray) -> np.ndarray:
    """Convex hull of the clicked landmarks, in court metres."""
    pts = np.array([LANDMARKS[n] for n in sorted(landmarks)], dtype=np.float32)
    if len(pts) < 3:
        return pts
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2)


def _region_trust(hull_m: np.ndarray) -> tuple[float, dict[str, float]]:
    """How far each court region sits outside the clicked hull.

    Distance is normalised by the hull's own size, so "0.5" means half a hull
    width beyond the evidence — a scale-free way of saying how much of this is
    extrapolation.
    """
    if len(hull_m) < 3:
        return 0.0, {name: 1.0 for name in REGIONS}
    hull = hull_m.astype(np.float32)
    scale = max(float(np.hypot(*(hull.max(axis=0) - hull.min(axis=0)))), 1e-6)
    extrapolation = {}
    for name, point in REGIONS.items():
        # negative inside, positive outside; measureDist gives signed distance
        d = cv2.pointPolygonTest(hull, (float(point[0]), float(point[1])), True)
        extrapolation[name] = max(0.0, -d) / scale if d < 0 else 0.0
    hull_area = float(cv2.contourArea(hull))
    coverage = min(1.0, hull_area / (COURT_W * COURT_L))
    return coverage, extrapolation


def _is_convex_court(H: np.ndarray) -> bool:
    """The court must map to a convex, correctly-wound quadrilateral.

    A flipped or folded homography is easy to detect here and impossible to
    diagnose 200 lines downstream, where it just produces coordinates that look
    like a player standing in the stands.
    """
    px = np.array(_court_corner_px(H), dtype=np.float32)
    if not np.isfinite(px).all():
        return False
    return bool(cv2.isContourConvex(px.reshape(-1, 1, 2).astype(np.float32)))


def court_px_width(corners_px: list[list[float]]) -> float:
    """Average of the two endline pixel lengths — the pixel-to-metre scale for
    this session's actual framing. Screen-space speeds are meaningless as raw
    px/s across sessions shot at different distances, so every such threshold
    is normalised by this.
    """
    (flx, fly), (frx, fry), (nrx, nry), (nlx, nly) = corners_px
    far = float(np.hypot(frx - flx, fry - fly))
    near = float(np.hypot(nrx - nlx, nry - nly))
    return (far + near) / 2.0


def side_of(y: float) -> str:
    """Which half of the court a court-space y falls in."""
    return "far" if y < NET_Y else "near"


def dist_from_net(y: float) -> float:
    """Absolute distance in metres from the net plane."""
    return abs(y - NET_Y)


def in_front_row(y: float) -> bool:
    """True when the position is between the net and that side's attack line."""
    return dist_from_net(y) <= ATTACK_LINE_OFFSET


def on_court(x: float, y: float, margin: float = 1.5) -> bool:
    """True for positions on or near the court — margin (metres) admits the
    free zone, where servers stand and defenders chase balls.
    """
    return -margin <= x <= COURT_W + margin and -margin <= y <= COURT_L + margin


def zone_for(x: float, y: float) -> int | None:
    """Volleyball rotational zone 1-6 for a court position, or None if off court.

    Zones are numbered from each team's own point of view, so the mapping is
    mirrored between the two halves:

        1 right back   2 right front   3 middle front
        4 left front   5 left back     6 middle back

    A player on the NEAR side faces -y, so their right hand is +x; a player on
    the FAR side faces +y, so their right hand is -x. Getting this mirroring
    wrong would silently swap every outside hitter with every opposite.
    """
    if not on_court(x, y, margin=0.0):
        return None
    front = in_front_row(y)
    if y >= NET_Y:                      # near side, faces -y, right = +x
        col = 2 if x >= 6.0 else (1 if x >= 3.0 else 0)   # 0=left 1=mid 2=right
    else:                               # far side, faces +y, right = -x
        col = 0 if x >= 6.0 else (1 if x >= 3.0 else 2)
    if front:
        return {0: 4, 1: 3, 2: 2}[col]
    return {0: 5, 1: 6, 2: 1}[col]


ZONE_ROLES = {
    1: "right back",
    2: "right front",
    3: "middle front",
    4: "left front",
    5: "left back",
    6: "middle back",
}

# The specialist position a player is most likely filling when they play out of
# a given zone, once post-serve switching has happened. Used for the per-role
# breakdown; `rotation.py` owns the caveats.
ZONE_SPECIALIST = {
    4: "outside hitter",
    3: "middle blocker",
    2: "opposite / setter",
    5: "left back",
    6: "middle back",
    1: "right back",
}
