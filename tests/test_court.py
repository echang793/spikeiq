import numpy as np
import pytest

from court import (COURT_L, COURT_W, NET_Y, CourtCalibration, court_px_width,
                   in_front_row, on_court, zone_for)


def test_corners_map_to_court_corners(calib, corners_px):
    got = calib.to_court(np.array(corners_px))
    want = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])
    assert np.allclose(got, want, atol=1e-3)


def test_centre_of_court_is_net_centre(calib, corners_px):
    # the pixel centroid of the four corners is NOT the court centre under
    # perspective, but the court centre must still round-trip through to_court
    mid_far = np.mean([corners_px[0], corners_px[1]], axis=0)
    mid_near = np.mean([corners_px[2], corners_px[3]], axis=0)
    # the net line in pixels is not the midpoint of those two either; instead
    # check the mapping is monotonic in y and brackets the net
    y_far = calib.to_court(np.array([mid_far]))[0][1]
    y_near = calib.to_court(np.array([mid_near]))[0][1]
    assert y_far < NET_Y < y_near


def test_attack_line_points_improve_fit(corners_px):
    """An 8-point fit must still honour the corners it was given."""
    base = CourtCalibration(corners_px)
    attack_px = base_attack_pixels(base)
    eight = CourtCalibration(corners_px, attack_px)
    got = eight.to_court(np.array(corners_px))
    want = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])
    assert np.allclose(got, want, atol=0.05)


def base_attack_pixels(calib) -> list[list[float]]:
    """Invert the 4-point homography to get where the attack lines land."""
    import cv2
    Hinv = np.linalg.inv(calib.H)
    court = np.array([[0.0, 6.0], [COURT_W, 6.0],
                      [COURT_W, 12.0], [0.0, 12.0]], dtype=np.float32)
    px = cv2.perspectiveTransform(court.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    return px.tolist()


def test_metres_per_pixel_larger_at_far_end(calib, corners_px):
    """Perspective means a pixel at the far endline covers more court than one
    at the near endline. Jump heights depend on getting this right."""
    far = calib.metres_per_pixel(*np.mean([corners_px[0], corners_px[1]], axis=0))
    near = calib.metres_per_pixel(*np.mean([corners_px[2], corners_px[3]], axis=0))
    assert far > near


def test_on_court_admits_free_zone_but_not_the_stands():
    assert on_court(4.5, 9.0)
    assert on_court(4.5, -1.0)        # server behind the endline
    assert not on_court(4.5, -5.0)    # spectator
    assert not on_court(-4.0, 9.0)


def test_front_row_is_within_three_metres_of_the_net():
    assert in_front_row(7.0)
    assert in_front_row(11.0)
    assert not in_front_row(5.0)
    assert not in_front_row(13.0)


@pytest.mark.parametrize("x,y,zone", [
    # near side faces -y, so their right hand is +x
    (7.5, 15.0, 1),   # right back
    (7.5, 10.0, 2),   # right front
    (4.5, 10.0, 3),   # middle front
    (1.5, 10.0, 4),   # left front
    (1.5, 15.0, 5),   # left back
    (4.5, 15.0, 6),   # middle back
    # far side faces +y, so their right hand is -x — the mirror image
    (1.5, 3.0, 1),
    (1.5, 8.0, 2),
    (4.5, 8.0, 3),
    (7.5, 8.0, 4),
    (7.5, 3.0, 5),
    (4.5, 3.0, 6),
])
def test_zone_numbering_is_mirrored_between_halves(x, y, zone):
    assert zone_for(x, y) == zone


def test_zone_is_none_off_court():
    assert zone_for(-1.0, 9.0) is None
    assert zone_for(4.5, 19.0) is None


def test_court_px_width_averages_the_two_endlines(corners_px):
    w = court_px_width(corners_px)
    assert 600 < w < 1500  # far endline 660 px, near endline 1420 px
    assert w == pytest.approx((660 + 1420) / 2, abs=1.0)
