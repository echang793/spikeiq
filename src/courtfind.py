"""Propose a court from an image, by fitting the model rather than reading lines.

The obvious approach — detect lines, decide which are the volleyball court — does
not survive an indoor gym, where the same floor carries basketball and badminton
lines and nothing in a line's appearance says which sport it belongs to.

So no line is ever classified. Instead a candidate homography is scored by
rendering the **whole** court model — four boundary lines, two attack lines, one
centre line — and measuring how much of it lands on detected line pixels. A stray
basketball line can coincide with one or two volleyball lines by chance; it cannot
produce a 9x18 rectangle *and* attack lines at exactly 3 m *and* a centre line at
9 m all at once. The rigidity of the full model at fixed ratios is what rejects
the distractors, and it is the only reason this can work at all.

It will still fail: shallow viewing angles, worn floors, players standing on the
lines, or a crop holding too little of the model. Every one of those ends in a
support score below `MIN_SUPPORT` and a returned `None`. That is the contract —
this is a time-saver in front of manual calibration, never a replacement for it,
and a wrong court accepted silently would be worse than no proposal at all.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from court import (COURT_LINES, COURT_L, COURT_W, LANDMARKS, CourtCalibration)

WORK_WIDTH = 960
MIN_SEGMENT_PX = 40          # shorter segments are floor scuff and shoe marks
COLLINEAR_DEG = 2.5          # segments this aligned lie on the same marking
COLLINEAR_PX = 6.0           # ...and this close in offset
MAX_LINES = 12               # distinct lines to consider; a court has seven
MIN_SUPPORT = 0.55           # below this we report failure instead of a court
MIN_VISIBLE = 0.35           # and below this too little of the model is in frame
SUPPORT_PX = 6.0             # a model line within this many pixels counts as met
SAMPLES_PER_LINE = 40
MAX_HYPOTHESES = 4000        # cap the search; pruning normally keeps it far lower


@dataclass
class CourtProposal:
    landmarks: dict[str, list[float]]
    support: float             # 0-1, how much of the model sits on real lines
    visible: float             # 0-1, how much of the model is inside the frame
    n_segments: int

    def as_dict(self) -> dict:
        return {"landmarks": self.landmarks, "support": round(self.support, 3),
                "visible": round(self.visible, 3), "n_segments": self.n_segments}

    def calibration(self) -> CourtCalibration:
        return CourtCalibration(self.landmarks)


def detect(image: np.ndarray, inside_hint: tuple[float, float] | None = None
           ) -> CourtProposal | None:
    """Best court proposal for this frame, or None when nothing scores well.

    `inside_hint` is an optional single point the user says is inside the court.
    It is a cheap disambiguator and much stronger than anything else available
    when two rectangles fit the floor markings about equally well.
    """
    work, scale = _prepare(image)
    segments = _segments(work)
    if len(segments) < 4:
        return None

    support_map = _support_map(work.shape, segments)
    lines = distinct_lines(segments)
    if len(lines) < 4:
        return None

    hint = None
    if inside_hint is not None:
        hint = (inside_hint[0] * scale, inside_hint[1] * scale)

    best = None
    for quad in _hypotheses(lines, work.shape, hint):
        scored = _score(quad, support_map, work.shape)
        if scored is None:
            continue
        if best is None or _better(scored, best):
            best = (scored[0], scored[1], scored[2], quad)

    if best is None:
        return None
    refined = _refine(best, support_map, work.shape)
    support, visible, H, _quad = refined

    # the contract: below the bar, say so
    if support < MIN_SUPPORT or visible < MIN_VISIBLE:
        return None

    landmarks = _landmarks_from_H(H, scale)
    return CourtProposal(landmarks=landmarks, support=support, visible=visible,
                         n_segments=len(segments))


def _better(scored, best) -> bool:
    return (round(scored[0], 4), round(scored[1], 4)) > (round(best[0], 4),
                                                         round(best[1], 4))


# --- image preparation ------------------------------------------------------

def _prepare(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = WORK_WIDTH / float(w) if w > WORK_WIDTH else 1.0
    if scale != 1.0:
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(grey), scale


def _segments(grey: np.ndarray) -> np.ndarray:
    """Long straight segments as (x1,y1,x2,y2). LSD where available."""
    try:
        lsd = cv2.createLineSegmentDetector()
        found = lsd.detect(grey)[0]
        segs = (found.reshape(-1, 4) if found is not None
                else np.empty((0, 4), np.float32))
    except (AttributeError, cv2.error):
        segs = np.empty((0, 4), np.float32)
    if len(segs) == 0:
        edges = cv2.Canny(grey, 60, 180)
        found = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                                minLineLength=MIN_SEGMENT_PX, maxLineGap=12)
        segs = (found.reshape(-1, 4).astype(np.float32) if found is not None
                else np.empty((0, 4), np.float32))
    if len(segs) == 0:
        return segs
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[lengths >= MIN_SEGMENT_PX]


def _support_map(shape, segments: np.ndarray) -> np.ndarray:
    """Distance to the nearest detected line pixel, per pixel."""
    mask = np.zeros(shape[:2], np.uint8)
    for x1, y1, x2, y2 in segments:
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)


# --- hypotheses -------------------------------------------------------------

def distinct_lines(segments: np.ndarray, limit: int = MAX_LINES) -> list[np.ndarray]:
    """Collapse segments onto the distinct straight lines they lie on.

    Grouping by image angle instead — "a court's lines form two families" — is
    wrong, and wrong in a way that quietly discarded the correct court: that
    holds in the court plane, but under perspective the two sidelines converge
    and sit tens of degrees apart in the image, so they land in different angular
    families. Taking the two largest families then kept the five cross lines and
    a single sideline, and every pair drawn from one sideline is degenerate.

    Working from distinct lines needs no assumption about direction at all. A
    detector finds a handful of them, so enumerating pairs stays cheap.
    """
    lines: list[tuple[np.ndarray, float]] = []      # (normalised line, length)
    for seg in segments:
        line = _line_through(seg)
        norm = np.hypot(line[0], line[1])
        if norm < 1e-8:
            continue
        line = line / norm
        if line[0] < 0 or (line[0] == 0 and line[1] < 0):
            line = -line                            # fix the sign for comparison
        length = float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]))
        for i, (existing, existing_len) in enumerate(lines):
            if _same_line(line, existing):
                if length > existing_len:
                    lines[i] = (line, length)       # keep the longest witness
                break
        else:
            lines.append((line, length))
    lines.sort(key=lambda pair: -pair[1])
    return [line for line, _ in lines[:limit]]


def _same_line(a: np.ndarray, b: np.ndarray) -> bool:
    """Two normalised lines describing the same painted marking."""
    angle = abs(np.degrees(np.arctan2(a[1], a[0]) - np.arctan2(b[1], b[0])))
    angle = min(angle, 180.0 - angle)
    return angle <= COLLINEAR_DEG and abs(a[2] - b[2]) <= COLLINEAR_PX


def _line_through(seg) -> np.ndarray:
    """Homogeneous line through a segment."""
    p = np.array([seg[0], seg[1], 1.0])
    q = np.array([seg[2], seg[3], 1.0])
    return np.cross(p, q)


def _intersect(a: np.ndarray, b: np.ndarray):
    p = np.cross(a, b)
    if abs(p[2]) < 1e-8:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]])


def _hypotheses(lines: list[np.ndarray], shape, hint):
    """Candidate court quadrilaterals, pruned hard before any scoring.

    Any two lines can be the sidelines and any two others the endlines, so this
    enumerates pairs-of-pairs. No assumption about which is which — the model
    score decides, which is the point.
    """
    count = 0
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k in (i, j):
                    continue
                for m in range(k + 1, n):
                    if m in (i, j):
                        continue
                    quad = _quad_from(lines[i], lines[j], lines[k], lines[m])
                    if quad is None or not _plausible(quad, shape, hint):
                        continue
                    count += 1
                    if count > MAX_HYPOTHESES:
                        return
                    yield quad


def _quad_from(l1, l2, c1, c2):
    """The four corners where two side lines meet two cross lines.

    The intersections are then sorted into a proper cycle. Taking them in the
    order the lines were paired can produce a bowtie, which fails the convexity
    prune — and that silently threw away the correct court on the first run of
    this.
    """
    # Structured order, kept deliberately: corners 0,1 lie on cross line c1 and
    # 2,3 on c2; corners 0,3 lie on side l1 and 1,2 on l2. Re-sorting these by
    # angle would throw that away and force trying all eight corner assignments
    # instead of four.
    pts = [_intersect(l1, c1), _intersect(l2, c1),
           _intersect(l2, c2), _intersect(l1, c2)]
    if any(p is None for p in pts):
        return None
    quad = np.array(pts, dtype=np.float32)
    if not np.isfinite(quad).all():
        return None
    return quad


def _plausible(quad: np.ndarray, shape, hint) -> bool:
    h, w = shape[:2]
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return False
    area = abs(cv2.contourArea(quad))
    if area < 0.03 * w * h or area > 4.0 * w * h:
        return False
    # corners wildly outside the frame mean this is not the court we can see
    if np.any(np.abs(quad[:, 0]) > 4 * w) or np.any(np.abs(quad[:, 1]) > 4 * h):
        return False
    if hint is not None:
        if cv2.pointPolygonTest(quad, (float(hint[0]), float(hint[1])), False) < 0:
            return False
    return True


# --- scoring ----------------------------------------------------------------

# The court's cross lines, by their y in metres. Any two of these bound a
# rectangle whose sides are the sidelines, so a detected quad might be any of
# them — which is what makes a cropped court findable: with the far endline out
# of frame, the visible quad is (centre line, near endline) or (attack, attack),
# never the full boundary, and insisting on the boundary is what made an earlier
# version fit a plausible wrong court instead.
CROSS_YS = (0.0, COURT_L / 2 - 3.0, COURT_L / 2, COURT_L / 2 + 3.0, COURT_L)
CROSS_PAIRS = [(a, b) for i, a in enumerate(CROSS_YS) for b in CROSS_YS[i + 1:]]


def _targets():
    """Candidate court-space rectangles a detected quad could be."""
    for y_a, y_b in CROSS_PAIRS:
        yield np.array([[0.0, y_a], [COURT_W, y_a],
                        [COURT_W, y_b], [0.0, y_b]], np.float32)


def _homography_for(quad_pair):
    """Homographies mapping a detected quad onto each candidate rectangle.

    `quad_pair` carries which corners came from which line pair, so the only real
    freedom left is which cross line is the nearer one and which sideline is
    x = 0 — four combinations, not the eight that discarding that structure would
    force. The model score decides between them, since the attack and centre
    lines only land on real markings under the right assignment.
    """
    quad = quad_pair
    for target in _targets():
        for flip_cross in (False, True):
            for flip_side in (False, True):
                src = quad.copy()
                if flip_cross:
                    src = src[[3, 2, 1, 0]]
                if flip_side:
                    src = src[[1, 0, 3, 2]]
                H = cv2.getPerspectiveTransform(src.astype(np.float32), target)
                if H is not None and np.isfinite(H).all():
                    yield H


def _score(quad: np.ndarray, support_map: np.ndarray, shape):
    """Best (support, visible, H) over the quad's possible interpretations.

    Ties on support are broken by how much of the model is visible, which favours
    the interpretation that explains the most of the frame rather than one that
    survives by pushing most of the court out of shot.
    """
    best = None
    for H in _homography_for(quad):
        if not matches_convention(H, shape) or not court_is_sane(H, shape):
            continue
        scored = _score_one(H, support_map, shape)
        if scored is None:
            continue
        key = (round(scored[0], 4), round(scored[1], 4))
        if best is None or key > (round(best[0], 4), round(best[1], 4)):
            best = (scored[0], scored[1], H)
    return best


def matches_convention(H: np.ndarray, shape) -> bool:
    """Whether this homography follows the documented court orientation.

    A volleyball court is symmetric: swap the ends, or mirror left for right, and
    the seven lines land in exactly the same places. So the markings alone cannot
    say which end is which — four different homographies score a perfect 1.0 on a
    clean court, and three of them are 450 to 985 pixels wrong.

    Geometry settles it, using the convention `court.py` already states: y grows
    towards the camera and x grows to the right of frame. The near endline is
    therefore lower in the image than the far one, and court x = 9 is to the right
    of x = 0. Without this check auto-detection would mirror the court roughly
    three times out of four, quietly swapping every left-front hitter with a
    right-front one.
    """
    Hinv = np.linalg.inv(H)
    probes = np.array([[COURT_W / 2, 0.0], [COURT_W / 2, COURT_L],
                       [0.0, COURT_L / 2], [COURT_W, COURT_L / 2]], np.float32)
    px = cv2.perspectiveTransform(probes.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    if not np.isfinite(px).all():
        return False
    far_mid, near_mid, left_mid, right_mid = px
    return bool(near_mid[1] > far_mid[1] and right_mid[0] > left_mid[0])


MIN_COURT_AREA = 0.05        # of the frame; smaller is a collapsed fit
MAX_COURT_AREA = 6.0
MIN_COURT_SPAN = 0.20        # endline-to-endline, as a share of frame height
MIN_CROSS_GAP_PX = 5.0       # adjacent cross lines must be distinguishable
MIN_LINES_SUPPORTED = 5      # of the seven, each on real markings
PER_LINE_SUPPORT = 0.6


def in_front_of_camera(H: np.ndarray) -> bool:
    """Whether the whole court lies on one side of the horizon.

    This is the check that actually kills degenerate fits, and it is the one I
    was missing. A plane seen by a camera maps to the image with a consistent
    sign in the third homogeneous coordinate; when the horizon line crosses the
    court, part of it is behind the camera. `perspectiveTransform` divides by that
    coordinate regardless and hands back perfectly plausible-looking pixels, so
    on real footage a near-singular fit put all seven model lines into a small
    cluster of strong edges around the net and scored 0.99 — with an implied court
    that was a self-intersecting sliver running off the top of the frame.

    Checking the sign directly is exact, cheap, and needs no thresholds.
    """
    Hinv = np.linalg.inv(H)
    pts = np.array([[0.0, 0.0, 1.0], [COURT_W, 0.0, 1.0],
                    [COURT_W, COURT_L, 1.0], [0.0, COURT_L, 1.0],
                    [COURT_W / 2, COURT_L / 2, 1.0]])
    w = (Hinv @ pts.T)[2]
    if not np.isfinite(w).all():
        return False
    if np.abs(w).min() < 1e-9:
        return False
    return bool(np.all(w > 0) or np.all(w < 0))


def court_is_sane(H: np.ndarray, shape) -> bool:
    """Reject fits that are geometrically degenerate rather than merely poor.

    Found on real footage: a homography that collapses the whole court onto the
    net line scores a perfect 1.0. Every model line piles onto one strong
    horizontal edge, every sample lands on a marking, and the support metric —
    which only asks "is this sample near a line" — is delighted. The court it
    implies runs off the top of the frame into the wall banner.

    So the court the homography implies has to be checked as a shape: a convex
    quadrilateral of sensible area, spanning enough of the frame end to end, with
    its five cross lines actually distinguishable from one another.
    """
    if not in_front_of_camera(H):
        return False
    h, w = shape[:2]
    Hinv = np.linalg.inv(H)
    corners = np.array([[0.0, 0.0], [COURT_W, 0.0],
                        [COURT_W, COURT_L], [0.0, COURT_L]], np.float32)
    quad = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    if not np.isfinite(quad).all():
        return False
    if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return False
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    if not (MIN_COURT_AREA * w * h <= area <= MAX_COURT_AREA * w * h):
        return False

    mids = cv2.perspectiveTransform(
        np.array([[[COURT_W / 2, y]] for y in CROSS_YS], np.float32),
        Hinv).reshape(-1, 2)
    if not np.isfinite(mids).all():
        return False
    if float(np.hypot(*(mids[-1] - mids[0]))) < MIN_COURT_SPAN * h:
        return False
    gaps = np.hypot(*np.diff(mids, axis=0).T)
    return bool(gaps.min() >= MIN_CROSS_GAP_PX)


def _score_one(H: np.ndarray, support_map: np.ndarray, shape):
    """How much of the rendered court model lands on detected line pixels.

    Normalised by how much of the model is inside the frame, so a partly cropped
    court is not penalised for being partly cropped — it is scored on what can
    actually be seen.
    """
    h, w = shape[:2]
    Hinv = np.linalg.inv(H)
    matched = 0
    seen = 0
    inside_total = 0
    for a, b in COURT_LINES:
        ends = cv2.perspectiveTransform(
            np.array([[a], [b]], np.float32), Hinv).reshape(-1, 2)
        if not np.isfinite(ends).all():
            continue
        ts = np.linspace(0.0, 1.0, SAMPLES_PER_LINE)
        pts = np.array([[a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
                        for t in ts], np.float32)
        px = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        ok = ((px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
              & np.isfinite(px).all(axis=1))
        n_inside = int(ok.sum())
        inside_total += n_inside
        if n_inside < SAMPLES_PER_LINE * 0.2:
            continue          # too little of this line is in frame to judge
        seen += 1
        if _line_is_on_a_marking(px[ok], support_map):
            matched += 1
    total = SAMPLES_PER_LINE * len(COURT_LINES)
    if seen == 0:
        return None
    # Support is the share of the model's OWN lines that sit on real markings,
    # not the share of samples that happen to be near some edge. On real footage
    # the sample-based version scored an extremely oblique wrong pose at 1.0,
    # because a gym scene is full of edges — net tape, banners, sponsor boards —
    # and a foreshortened model can pass through them without ever coinciding
    # with a court line. Judging line by line is what makes the model's rigidity
    # actually bite.
    if matched < min(MIN_LINES_SUPPORTED, seen):
        return None
    return matched / seen, inside_total / total


def _line_is_on_a_marking(px: np.ndarray, support_map: np.ndarray) -> bool:
    """Whether a projected model line runs ALONG a detected marking.

    Requires most of the line's length to be close to detected line pixels —
    checked over the whole span rather than pixel by pixel, so a line that merely
    crosses a few edges cannot qualify.
    """
    if len(px) < 4:
        return False
    xs = np.clip(px[:, 0].astype(int), 0, support_map.shape[1] - 1)
    ys = np.clip(px[:, 1].astype(int), 0, support_map.shape[0] - 1)
    close = support_map[ys, xs] <= SUPPORT_PX
    return bool(close.mean() >= PER_LINE_SUPPORT)


def _refine(best, support_map: np.ndarray, shape):
    """Nudge the corners to maximise the same score.

    Cheap coordinate descent. The hypothesis is already close; this only cleans up
    error in the line intersections, which are sensitive when two lines meet at a
    shallow angle near the edge of frame.
    """
    support, visible, H, quad = best
    for step in (4.0, 2.0, 1.0):
        improved = True
        while improved:
            improved = False
            for corner in range(4):
                for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    trial = quad.copy()
                    trial[corner] += (dx, dy)
                    if not _plausible(trial, shape, None):
                        continue
                    scored = _score(trial, support_map, shape)
                    if scored and _better(scored, (support, visible)):
                        support, visible, H = scored
                        quad = trial
                        improved = True
    return support, visible, H, quad


def _landmarks_from_H(H: np.ndarray, scale: float) -> dict[str, list[float]]:
    """Every landmark's pixel position, from the winning homography.

    Derived from H rather than from the quad, because the quad is not necessarily
    the court boundary — it might be the pair of attack lines, or the centre line
    and an endline, which is exactly what lets a cropped court be found. All ten
    landmarks still follow, the same "a homography gives you the whole court"
    property that makes a partly visible court workable.
    """
    Hinv = np.linalg.inv(H)
    out = {}
    for name, xy in LANDMARKS.items():
        px = cv2.perspectiveTransform(
            np.array([[xy]], np.float32), Hinv).reshape(2)
        out[name] = [float(px[0] / scale), float(px[1] / scale)]
    return out
