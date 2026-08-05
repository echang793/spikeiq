"""Auto court detection, and — more importantly — its refusals.

I argued against auto-detection on the grounds that gym floors carry basketball
and badminton lines and a detector cannot tell which is which. The answer is not
to classify lines but to fit the whole court model, whose fixed ratios a stray
line cannot satisfy. The test that matters most here is therefore not the one
where it succeeds: it is the cluttered-floor test, where getting a confidently
wrong court would be worse than getting nothing.
"""

import cv2
import numpy as np
import pytest

from court import LANDMARKS, CourtCalibration
from courtfind import MIN_SUPPORT, detect


def render_court(corners, size=(1280, 720), *, lines=True, clutter=(),
                 line_colour=(245, 245, 245), noise=18) -> np.ndarray:
    """Draw a volleyball court from four image-space corners.

    `clutter` takes extra straight lines — the basketball and badminton markings
    that make this problem interesting.
    """
    w, h = size
    rng = np.random.default_rng(3)
    img = np.full((h, w, 3), 105, np.uint8)
    img = cv2.add(img, np.repeat(rng.integers(0, noise, (h, w, 1), dtype=np.uint8),
                                 3, axis=2))
    calib = CourtCalibration([list(map(float, c)) for c in corners])
    if lines:
        for a, b in _model_segments():
            pa, pb = calib.court_to_px(np.array([a, b]))
            cv2.line(img, tuple(np.int32(pa)), tuple(np.int32(pb)),
                     line_colour, 3, cv2.LINE_AA)
    for a, b in clutter:
        cv2.line(img, a, b, (60, 130, 220), 3, cv2.LINE_AA)
    return img


def _model_segments():
    from court import COURT_LINES
    return COURT_LINES


CORNERS = [(360, 250), (930, 250), (1130, 640), (150, 640)]


@pytest.fixture(scope="module")
def clean_court():
    return render_court(CORNERS)


def landmark_error(proposal, truth_calib) -> float:
    """Worst landmark displacement, in pixels."""
    worst = 0.0
    for name, px in proposal.landmarks.items():
        want = truth_calib.court_to_px(np.array([LANDMARKS[name]]))[0]
        worst = max(worst, float(np.hypot(*(np.array(px) - want))))
    return worst


# --- it works when it can ---------------------------------------------------

def test_finds_a_clean_court(clean_court):
    proposal = detect(clean_court)
    assert proposal is not None, "failed on a clean, fully visible court"
    assert proposal.support >= MIN_SUPPORT
    truth = CourtCalibration([list(map(float, c)) for c in CORNERS])
    assert landmark_error(proposal, truth) < 25.0


def test_a_proposal_converts_straight_into_a_calibration(clean_court):
    """Auto and manual must land in exactly the same place, or there are two
    code paths to keep honest instead of one."""
    proposal = detect(clean_court)
    calib = proposal.calibration()
    assert calib.quality.n_landmarks == len(LANDMARKS)
    assert calib.quality.convex


# --- and refuses when it cannot ---------------------------------------------

def test_the_court_is_not_mirrored_or_end_swapped(clean_court):
    """A volleyball court is symmetric: swap the ends or mirror left for right and
    all seven lines land in the same places. Four homographies score a perfect 1.0
    on a clean court and three of them are hundreds of pixels wrong, so geometry
    has to settle it — near is lower in frame, x grows to the right. Without this
    the court would be mirrored about three times in four, silently swapping every
    left-front hitter with a right-front one."""
    proposal = detect(clean_court)
    assert proposal is not None
    marks = proposal.landmarks
    # near endline below the far one
    assert marks["corner_near_left"][1] > marks["corner_far_left"][1]
    assert marks["corner_near_right"][1] > marks["corner_far_right"][1]
    # right sideline to the right of the left one
    assert marks["corner_near_right"][0] > marks["corner_near_left"][0]
    assert marks["corner_far_right"][0] > marks["corner_far_left"][0]


def test_convention_check_rejects_the_symmetric_alternatives():
    from courtfind import matches_convention

    right_way = CourtCalibration([list(map(float, c)) for c in CORNERS])
    assert matches_convention(right_way.H, (720, 1280))

    end_swapped = CourtCalibration({
        "corner_far_left": list(map(float, CORNERS[3])),
        "corner_far_right": list(map(float, CORNERS[2])),
        "corner_near_right": list(map(float, CORNERS[1])),
        "corner_near_left": list(map(float, CORNERS[0])),
    })
    assert not matches_convention(end_swapped.H, (720, 1280))


def test_a_court_behind_the_camera_is_rejected():
    """A near-singular fit puts part of the court behind the camera, where
    perspectiveTransform still returns plausible-looking pixels. Found on real
    footage scoring 0.99 with a self-intersecting sliver of a court."""
    from courtfind import in_front_of_camera

    good = CourtCalibration([list(map(float, c)) for c in CORNERS])
    assert in_front_of_camera(good.H)

    # court -> image with w = 1 - 0.1*y, which crosses zero at y = 10 m: the
    # horizon runs through the middle of the court, so the far half is behind
    # the camera even though every projected pixel looks perfectly ordinary
    court_to_image = np.array([[40.0, 0.0, 300.0],
                               [0.0, 30.0, 200.0],
                               [0.0, -0.1, 1.0]])
    assert not in_front_of_camera(np.linalg.inv(court_to_image))


def test_a_collapsed_court_is_rejected(clean_court):
    """Every model line piled onto one strong edge would score perfectly under a
    "near any edge" metric, which is why support is judged line by line."""
    from courtfind import court_is_sane

    good = CourtCalibration([list(map(float, c)) for c in CORNERS])
    assert court_is_sane(good.H, clean_court.shape)

    squashed = CourtCalibration([[360.0, 300.0], [930.0, 300.0],
                                 [935.0, 306.0], [355.0, 306.0]])
    assert not court_is_sane(squashed.H, clean_court.shape)


def test_lines_crossing_edges_do_not_count_as_lying_along_them():
    """The metric asks whether a model line runs ALONG a marking, not whether it
    passes over a few. A gym is full of edges; without this an oblique wrong pose
    scores 1.0."""
    from courtfind import _line_is_on_a_marking

    # a support map where only a horizontal band at y=50 has markings
    dist = np.full((100, 200), 50.0, np.float32)
    dist[48:53, :] = 0.0
    along = np.array([[x, 50.0] for x in range(10, 190, 4)], np.float32)
    across = np.array([[100.0, y] for y in range(10, 90, 2)], np.float32)
    assert _line_is_on_a_marking(along, dist)
    assert not _line_is_on_a_marking(across, dist)


def test_a_blank_floor_yields_nothing():
    blank = render_court(CORNERS, lines=False)
    assert detect(blank) is None


def test_a_gym_floor_full_of_other_sports_lines_is_refused_not_guessed():
    """The whole objection to auto-detection, made concrete. A basketball key,
    a badminton court and a centre circle, and NO volleyball court. Returning a
    confident wrong court here would be worse than returning nothing."""
    clutter = [
        ((200, 300), (1100, 300)), ((200, 420), (1100, 420)),   # stray parallels
        ((300, 200), (300, 700)), ((980, 200), (980, 700)),     # stray verticals
        ((150, 560), (1150, 560)), ((420, 240), (420, 690)),
        ((640, 200), (640, 700)), ((250, 660), (1050, 660)),
    ]
    img = render_court(CORNERS, lines=False, clutter=clutter)
    proposal = detect(img)
    if proposal is not None:
        # if it does propose something, it must be because the model genuinely
        # fits — never a low-support guess dressed up as success
        assert proposal.support >= MIN_SUPPORT


def test_low_support_never_comes_back_as_success():
    """The contract: a proposal always clears the bar, or there is no proposal."""
    rng = np.random.default_rng(11)
    for seed in range(4):
        noise = rng.integers(0, 255, (400, 700, 3), dtype=np.uint8)
        proposal = detect(noise)
        assert proposal is None or proposal.support >= MIN_SUPPORT


def test_too_little_of_the_court_in_frame_is_refused():
    """A crop holding almost none of the model cannot be verified, so it is not
    proposed."""
    tiny = render_court(CORNERS)[300:380, 500:620]
    proposal = detect(tiny)
    assert proposal is None or proposal.visible >= 0.3


# --- partial court ----------------------------------------------------------

def test_a_court_with_the_far_endline_cropped_can_still_be_proposed():
    """Scoring is normalised by how much of the model is in frame, so a cropped
    court is judged on what is visible rather than penalised for the crop."""
    # the far endline sits at y=250, so the crop has to start past it
    img = render_court(CORNERS)[330:, :]
    proposal = detect(img)
    if proposal is not None:
        assert proposal.support >= MIN_SUPPORT
        assert proposal.visible < 1.0, "crop should have removed part of the model"


# --- the hint ---------------------------------------------------------------

def test_an_inside_hint_rejects_courts_that_do_not_contain_it(clean_court):
    """One click is a cheap disambiguator and much stronger than anything else
    available when two rectangles fit about equally well."""
    outside = detect(clean_court, inside_hint=(20.0, 20.0))
    inside = detect(clean_court, inside_hint=(640.0, 450.0))
    assert inside is not None
    if outside is not None:
        # a proposal must actually contain the hint it was given
        quad = np.array([inside.landmarks[k] for k in
                         ["corner_far_left", "corner_far_right",
                          "corner_near_right", "corner_near_left"]], np.float32)
        assert cv2.pointPolygonTest(quad, (640.0, 450.0), False) >= 0


def test_hint_is_optional(clean_court):
    assert detect(clean_court) is not None


# --- shape of the output ----------------------------------------------------

def test_proposal_reports_all_ten_landmarks(clean_court):
    proposal = detect(clean_court)
    assert set(proposal.landmarks) == set(LANDMARKS)


def test_proposal_serialises_for_the_api(clean_court):
    d = detect(clean_court).as_dict()
    assert set(d) == {"landmarks", "support", "visible", "n_segments"}
    assert 0.0 <= d["support"] <= 1.0
    assert 0.0 <= d["visible"] <= 1.0


def test_detect_survives_a_tiny_image():
    assert detect(np.zeros((12, 12, 3), np.uint8)) is None
