"""Camera motion solving, against synthetic warps with known ground truth.

This is the one place where correctness can be proved rather than argued: warp a
real frame by a homography we chose, and the solver must recover its inverse. If
it does, handheld support works; if it does not, no amount of downstream care
helps.
"""

import cv2
import numpy as np
import pytest

from camera import (FIXED_PX, CameraSolver, CameraTrack, _corner_motion,
                    _frame_corners, _player_mask, _rescale)
from court import LANDMARKS, CourtCalibration, CourtMapper


@pytest.fixture(scope="module")
def gym_frame():
    """A textured synthetic 'gym': court lines, wall banners, floor speckle.

    Needs real texture in the background — a flat image has no features to match
    and would make the solver look broken when it is only starved.
    """
    rng = np.random.default_rng(7)
    img = np.full((720, 1280, 3), 60, np.uint8)
    img[:260] = (40, 70, 45)                      # back wall
    img[260:] = (110, 95, 80)                     # floor
    # speckle gives ORB something to lock onto, as a real floor does
    noise = rng.integers(0, 45, (720, 1280, 1), dtype=np.uint8)
    img = cv2.add(img, np.repeat(noise, 3, axis=2))
    # court lines
    for a, b in [((180, 640), (1100, 640)), ((300, 340), (980, 340)),
                 ((300, 340), (180, 640)), ((980, 340), (1100, 640)),
                 ((250, 470), (1030, 470))]:
        cv2.line(img, a, b, (245, 245, 245), 3)
    # wall detail: strong, distinctive corners well away from the court
    for x in range(60, 1240, 150):
        cv2.rectangle(img, (x, 70), (x + 90, 150), (200, 60, 60), -1)
        cv2.putText(img, "AC", (x + 8, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (250, 250, 250), 2)
    return img


def shift_homography(dx: float, dy: float, rot_deg: float = 0.0,
                     scale: float = 1.0, centre=(640, 360)) -> np.ndarray:
    M = cv2.getRotationMatrix2D(centre, rot_deg, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return np.vstack([M, [0, 0, 1]])


def warp(img, H):
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))


# --- the core claim ---------------------------------------------------------

@pytest.mark.parametrize("dx,dy,rot", [
    (18.0, -11.0, 0.0),
    (-30.0, 22.0, 1.5),
    (45.0, 0.0, -2.5),
])
def test_solver_recovers_the_warp_that_was_applied(gym_frame, dx, dy, rot):
    """Ground truth is known by construction, so this is a real measurement of
    accuracy rather than a consistency check."""
    truth = shift_homography(dx, dy, rot)
    moved = warp(gym_frame, truth)

    solver = CameraSolver(gym_frame, detect_cuts=False)
    solver.update(10, moved)
    track = solver.track
    recovered = track.warp_for(10)
    assert recovered is not None, "failed to solve a modest shift"

    # the solver maps moved -> reference, so it should invert `truth`
    probe = np.float32([[300, 400], [900, 500], [640, 620]]).reshape(-1, 1, 2)
    moved_pts = cv2.perspectiveTransform(probe, truth)
    back = cv2.perspectiveTransform(moved_pts, recovered).reshape(-1, 2)
    assert np.allclose(back, probe.reshape(-1, 2), atol=3.0), back


def test_court_positions_survive_a_moving_camera(gym_frame):
    """The whole feature, end to end: court coordinates read off a warped frame
    must match those from the unwarped one."""
    marks = {name: [float(v) for v in
                    CourtCalibration([[300, 340], [980, 340],
                                      [1100, 640], [180, 640]])
                    .court_to_px(np.array([xy]))[0]]
             for name, xy in LANDMARKS.items()}
    calib = CourtCalibration(marks)

    truth = shift_homography(25.0, -14.0, 1.0)
    solver = CameraSolver(gym_frame, detect_cuts=False)
    solver.update(50, warp(gym_frame, truth))
    mapper = CourtMapper(calib, solver.finish())

    # a player standing at a known court spot, seen in the moved frame
    court_pt = np.array([[4.5, 12.0]])
    px_ref = calib.court_to_px(court_pt)
    px_moved = cv2.perspectiveTransform(
        px_ref.reshape(-1, 1, 2).astype(np.float32), truth).reshape(-1, 2)

    got = mapper.to_court(px_moved, [50])
    assert np.allclose(got, court_pt, atol=0.35), got
    # and ignoring the motion would have been visibly wrong
    naive = calib.to_court(px_moved)
    assert not np.allclose(naive, court_pt, atol=0.35)


def test_players_are_masked_out_of_the_solve(gym_frame):
    """Twelve bodies moving together will drag an unmasked solve toward the
    players' motion instead of the court's. This is the test that makes the
    masking worth having."""
    moving = gym_frame.copy()
    boxes = []
    for i, x in enumerate(range(200, 1000, 110)):
        # big, high-contrast blobs: an unmasked solver finds them irresistible
        cv2.rectangle(moving, (x + 60, 380), (x + 150, 620), (20, 20, 230), -1)
        boxes.append([x + 60, 380, x + 150, 620])
    # same players, displaced, camera unmoved
    displaced = gym_frame.copy()
    shifted_boxes = []
    for x in range(200, 1000, 110):
        cv2.rectangle(displaced, (x + 130, 380), (x + 220, 620), (20, 20, 230), -1)
        shifted_boxes.append([x + 130, 380, x + 220, 620])

    solver = CameraSolver(moving, detect_cuts=False)
    solver.update(1, displaced, boxes=shifted_boxes)
    warp_m = solver.track.warp_for(1)
    assert warp_m is not None
    motion = _corner_motion(warp_m, _frame_corners(solver.ref_grey.shape),
                            solver.scale)
    assert motion < 6.0, f"masked solve drifted {motion:.1f}px on a static camera"


def test_player_mask_blanks_the_boxes_and_nothing_else():
    mask = _player_mask((100, 200), [[50, 20, 90, 60]], scale=1.0)
    assert mask[40, 70] == 0        # inside the box
    assert mask[5, 5] == 255        # far outside
    assert mask.mean() > 100        # most of the frame is still usable


def test_no_boxes_leaves_the_whole_frame_usable():
    assert _player_mask((50, 50), None, 1.0).min() == 255


# --- regime detection -------------------------------------------------------

def test_an_unmoved_camera_is_classified_fixed(gym_frame):
    """A fixed tripod must not pay for any of this."""
    solver = CameraSolver(gym_frame, detect_cuts=False)
    for i in range(4):
        solver.update(i, gym_frame)
    track = solver.finish()
    assert track.regime() == "fixed"
    assert track.is_identity
    assert track.max_motion_px <= FIXED_PX


def test_identity_track_short_circuits_the_mapper(gym_frame):
    calib = CourtCalibration([[300, 340], [980, 340], [1100, 640], [180, 640]])
    mapper = CourtMapper(calib, CameraTrack.identity())
    assert not mapper.moving
    pts = np.array([[640.0, 600.0]])
    assert np.allclose(mapper.to_court(pts, [99]), calib.to_court(pts))


def test_real_motion_is_classified_handheld(gym_frame):
    solver = CameraSolver(gym_frame, detect_cuts=False)
    for i, dx in enumerate([0.0, 12.0, 26.0, 40.0]):
        solver.update(i, warp(gym_frame, shift_homography(dx, -dx / 3)))
    track = solver.finish()
    assert track.regime() == "handheld"
    assert not track.is_identity
    assert track.max_motion_px > FIXED_PX


def test_a_cut_is_detected_and_reported(gym_frame):
    """A stop/restart or repositioning invalidates the reference match rather
    than merely shifting it, so it has to be visible in the summary."""
    other = np.full_like(gym_frame, 30)
    other[:, :] = (10, 200, 220)
    solver = CameraSolver(gym_frame)
    solver.update(0, gym_frame)
    solver.update(1, other)
    track = solver.finish()
    assert track.cuts == [1]
    assert track.regime() == "handheld_with_cuts"


# --- honest failure ---------------------------------------------------------

def test_an_unsolvable_frame_is_left_unsolved_not_guessed(gym_frame):
    """Reusing a stale warp is the one genuinely dangerous option here."""
    solver = CameraSolver(gym_frame, detect_cuts=False)
    solver.update(0, gym_frame)
    solver.update(1, np.full_like(gym_frame, 128))   # featureless
    assert solver.track.warp_for(1) is None
    assert solver.track.confidence_for(1) == 0.0
    assert solver.track.solved_fraction < 1.0


def test_unsolved_frames_become_nan_and_drop_out(gym_frame):
    """NaN is chosen deliberately: every consumer filters through `on_court`,
    and NaN fails those comparisons, so an unsolved frame disappears instead of
    contributing a confident wrong position."""
    from court import on_court
    calib = CourtCalibration([[300, 340], [980, 340], [1100, 640], [180, 640]])
    track = CameraTrack(warps={5: np.eye(3)}, confidence={5: 0.9, 6: 0.0},
                        motion_px={5: 30.0})
    mapper = CourtMapper(calib, track)
    got = mapper.to_court(np.array([[640.0, 600.0], [640.0, 600.0]]), [5, 6])
    assert np.isfinite(got[0]).all()
    assert np.isnan(got[1]).all()
    assert not on_court(*got[1])


def test_an_absurd_warp_is_dropped_rather_than_flagged(gym_frame):
    """Seen on cropped footage where most background texture was gone: a few
    frames produced confident nonsense in the thousands of pixels. A warp claiming
    the camera moved further than the frame is wide is not worth keeping, however
    many inliers agreed."""
    from camera import _absurd_motion_limit

    limit = _absurd_motion_limit(gym_frame.shape)
    assert limit > 1000  # frame diagonal for a 1280x720 frame

    solver = CameraSolver(gym_frame, detect_cuts=False)
    # a warp far beyond anything physical, injected past the feature solve
    solver.track.confidence[7] = 0.9
    huge = shift_homography(5000.0, 0.0)
    motion = _corner_motion(huge, _frame_corners(solver.ref_grey.shape),
                            solver.scale)
    assert motion > limit


def test_beyond_supported_motion_is_flagged():
    """The honest boundary of a solve-against-reference design."""
    track = CameraTrack(warps={1: shift_homography(400, 0)},
                        confidence={1: 0.8}, motion_px={1: 400.0})
    assert track.summary()["beyond_supported_motion"] is True
    assert CameraTrack(warps={1: np.eye(3)}, confidence={1: 0.9},
                       motion_px={1: 20.0}).summary()[
                           "beyond_supported_motion"] is False


def test_mapper_rejects_mismatched_frames():
    calib = CourtCalibration([[300, 340], [980, 340], [1100, 640], [180, 640]])
    mapper = CourtMapper(calib, CameraTrack(warps={1: np.eye(3)},
                                            confidence={1: 1.0},
                                            motion_px={1: 30.0}))
    with pytest.raises(ValueError, match="line up"):
        mapper.to_court(np.array([[1.0, 2.0], [3.0, 4.0]]), [1])


# --- persistence ------------------------------------------------------------

def test_track_round_trips_through_parquet(tmp_path):
    track = CameraTrack(warps={2: shift_homography(5, 5)},
                        confidence={2: 0.7, 3: 0.0},
                        motion_px={2: 7.1}, cuts=[3], reference_frame=2)
    path = tmp_path / "camera.parquet"
    track.save(path)
    back = CameraTrack.load(path)
    assert np.allclose(back.warp_for(2), track.warp_for(2))
    assert back.warp_for(3) is None
    assert back.cuts == [3]
    assert back.reference_frame == 2


def test_rescale_maps_between_resolutions():
    """The solve happens on a downscaled image; callers hold full-resolution
    pixels, so getting this conversion wrong scales every position."""
    H = shift_homography(10.0, 0.0)
    full = _rescale(H, src_scale=0.5, dst_scale=0.5)
    pt = np.float32([[[100.0, 100.0]]])
    got = cv2.perspectiveTransform(pt, full).reshape(2)
    assert np.allclose(got, [120.0, 100.0], atol=1e-3)


def test_summary_reports_what_happened(gym_frame):
    solver = CameraSolver(gym_frame, detect_cuts=False)
    solver.update(0, gym_frame)
    solver.update(1, warp(gym_frame, shift_homography(20, -8)))
    s = solver.finish().summary()
    assert s["frames_attempted"] == 2
    assert s["frames_solved"] >= 1
    assert 0.0 < s["solved_fraction"] <= 1.0
    assert s["regime"] in ("fixed", "handheld")
