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


class CourtCalibration:
    """Homography from clicked reference points.

    The 4 outer corners are required. The 4 attack-line intersections are
    optional but strongly improve accuracy, for the same reason dinkiq anchors
    on the kitchen line: baseline corners are often estimated (far away, or
    cropped), while the attack lines sit near the net where they are clearly
    visible — and near the net is exactly where attacking and blocking metrics
    live. Eight correspondences give a least-squares fit anchored there.
    """

    def __init__(self, corners_px: list[list[float]],
                 attack_px: list[list[float]] | None = None):
        if len(corners_px) != 4:
            raise ValueError("need exactly 4 court corners")
        if attack_px is not None and len(attack_px) != 4:
            raise ValueError("attack_px must have exactly 4 points when given")
        self.corners_px = corners_px
        self.attack_px = attack_px
        src = np.array(corners_px, dtype=np.float32)
        if attack_px is None:
            self.H = cv2.getPerspectiveTransform(src, CORNER_TARGETS)
        else:
            src8 = np.vstack([src, np.array(attack_px, dtype=np.float32)])
            dst8 = np.vstack([CORNER_TARGETS, ATTACK_TARGETS])
            self.H, _ = cv2.findHomography(src8, dst8, method=0)  # least squares
            if self.H is None:
                raise ValueError("degenerate calibration points")
        self.H = self.H.astype(np.float64)

    def to_court(self, points_px: np.ndarray) -> np.ndarray:
        """Map (N,2) pixel points to (N,2) court metres."""
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

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
        path.write_text(json.dumps(
            {"corners_px": self.corners_px, "attack_px": self.attack_px}))

    @classmethod
    def load(cls, path: Path) -> "CourtCalibration":
        data = json.loads(path.read_text())
        return cls(data["corners_px"], data.get("attack_px"))


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
