"""Calibration from any subset of named landmarks, and how far to trust it.

A homography extrapolates the whole court from four points, so a partly cropped
court needs no special machinery — what it needs is an honest measure of how much
of the result is measurement and how much is extrapolation. These tests pin that
measure down, because it is the thing every partial-court metric decision rests
on.
"""

import cv2
import numpy as np
import pytest

from court import (COURT_L, COURT_W, LANDMARKS, MIN_LANDMARKS, CourtCalibration,
                   court_px_width)


def full_landmarks(calib) -> dict[str, list[float]]:
    """Pixel positions of all 10 landmarks, from a known-good calibration."""
    return {name: [float(v) for v in calib.court_to_px(np.array([xy]))[0]]
            for name, xy in LANDMARKS.items()}


NEAR_HALF = ["corner_near_left", "corner_near_right", "attack_near_left",
             "attack_near_right", "centre_left", "centre_right"]


def test_four_landmarks_are_enough(calib):
    marks = full_landmarks(calib)
    picked = {k: marks[k] for k in
              ["corner_far_left", "corner_far_right",
               "corner_near_right", "corner_near_left"]}
    fit = CourtCalibration(picked)
    got = fit.to_court(np.array(list(picked.values())))
    want = np.array([LANDMARKS[k] for k in picked])
    assert np.allclose(got, want, atol=1e-3)


def test_fewer_than_four_is_refused(calib):
    marks = full_landmarks(calib)
    with pytest.raises(ValueError, match="at least"):
        CourtCalibration({k: marks[k] for k in list(marks)[:3]})


def test_unknown_landmark_names_are_refused(calib):
    marks = full_landmarks(calib)
    marks["net_top_left"] = [10.0, 10.0]
    with pytest.raises(ValueError, match="unknown landmarks"):
        CourtCalibration(marks)


def test_any_four_non_collinear_landmarks_work(calib):
    """Not just corners. Whichever four happen to be in frame."""
    marks = full_landmarks(calib)
    picked = {k: marks[k] for k in ["centre_left", "centre_right",
                                    "corner_near_left", "corner_near_right"]}
    fit = CourtCalibration(picked)
    assert fit.quality.n_landmarks == 4


def test_collinear_landmarks_are_rejected(calib):
    """Four points along one sideline cannot locate a court, and must fail at
    fit time rather than produce coordinates that look plausible."""
    marks = full_landmarks(calib)
    picked = {k: marks[k] for k in ["corner_far_left", "attack_far_left",
                                    "centre_left", "attack_near_left"]}
    with pytest.raises((ValueError, cv2.error)):
        CourtCalibration(picked)


# --- the partial-court case -------------------------------------------------

def test_near_half_only_still_recovers_the_far_corners(calib):
    """The plan's headline claim: with only the near half clickable, the far
    endline is extrapolated. Tolerance is stated rather than assumed."""
    marks = full_landmarks(calib)
    partial = CourtCalibration({k: marks[k] for k in NEAR_HALF})
    truth = np.array([LANDMARKS["corner_far_left"], LANDMARKS["corner_far_right"]])
    got = partial.to_court(np.array([marks["corner_far_left"],
                                     marks["corner_far_right"]]))
    assert np.allclose(got, truth, atol=0.5), got


def test_click_noise_hurts_the_extrapolated_half_far_more(calib):
    """The real cost of a partial court is not the maths, it is that human
    clicks are a few pixels off and extrapolation amplifies that. A perfect
    synthetic fit would hide this entirely, so the clicks are jittered.
    """
    marks = full_landmarks(calib)
    rng = np.random.default_rng(0)
    noisy = {k: [v[0] + rng.normal(0, 3.0), v[1] + rng.normal(0, 3.0)]
             for k, v in marks.items() if k in NEAR_HALF}
    partial = CourtCalibration(noisy)

    def error_at(name):
        got = partial.to_court(np.array([marks[name]]))[0]
        return float(np.hypot(*(got - np.array(LANDMARKS[name]))))

    near_err = np.mean([error_at("corner_near_left"),
                        error_at("corner_near_right")])
    far_err = np.mean([error_at("corner_far_left"),
                       error_at("corner_far_right")])
    assert far_err > near_err * 2, (near_err, far_err)
    # and the quality object must have predicted that this would happen
    assert not partial.quality.trusted("far_court")


def test_partial_court_is_less_stable_than_a_full_one(calib):
    """Leave-one-out shift is the number that matters for extrapolation, and it
    must visibly degrade when the evidence is clustered in one half."""
    marks = full_landmarks(calib)
    full = CourtCalibration(marks)
    partial = CourtCalibration({k: marks[k] for k in NEAR_HALF})
    assert partial.quality.corner_shift_m > full.quality.corner_shift_m


def test_regions_beyond_the_clicked_points_are_flagged_as_extrapolated(calib):
    marks = full_landmarks(calib)
    partial = CourtCalibration({k: marks[k] for k in NEAR_HALF})
    q = partial.quality
    assert q.trusted("near_court")
    assert q.trusted("net")
    assert not q.trusted("far_court")
    assert q.region_extrapolation["far_court"] > q.region_extrapolation["near_court"]


def test_a_full_court_trusts_every_region(calib):
    q = CourtCalibration(full_landmarks(calib)).quality
    assert all(q.trusted(r) for r in q.region_extrapolation)
    assert q.hull_coverage > 0.95


def test_hull_coverage_reflects_how_much_was_actually_seen(calib):
    marks = full_landmarks(calib)
    full = CourtCalibration(marks)
    partial = CourtCalibration({k: marks[k] for k in NEAR_HALF})
    assert partial.quality.hull_coverage < full.quality.hull_coverage


def test_four_points_report_that_reprojection_error_means_nothing(calib):
    """With an exactly determined fit the error is zero by construction, so the
    quality object has to say so rather than letting 0.00 read as perfect."""
    marks = full_landmarks(calib)
    four = CourtCalibration({k: marks[k] for k in
                             ["corner_far_left", "corner_far_right",
                              "corner_near_right", "corner_near_left"]})
    assert four.quality.determined_only
    assert four.quality.corner_shift_m is None
    assert not CourtCalibration(marks).quality.determined_only


def test_a_misplaced_landmark_that_folds_the_court_is_refused(calib):
    """Swapping two corners folds the quadrilateral. Caught at fit time, where
    it is diagnosable, rather than downstream where it looks like a player
    standing in the stands."""
    marks = full_landmarks(calib)
    picked = {
        "corner_far_left": marks["corner_far_right"],
        "corner_far_right": marks["corner_far_left"],
        "corner_near_right": marks["corner_near_right"],
        "corner_near_left": marks["corner_near_left"],
    }
    with pytest.raises(ValueError, match="fold|not describe a court"):
        CourtCalibration(picked)


# --- compatibility ----------------------------------------------------------

def test_the_old_positional_shape_still_works(corners_px):
    """Calibrations stored by the previous UI must keep loading."""
    fit = CourtCalibration(corners_px)
    got = fit.to_court(np.array(corners_px))
    want = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])
    assert np.allclose(got, want, atol=1e-3)


def test_a_version_1_file_loads_as_landmarks(tmp_path, corners_px):
    import json
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"corners_px": corners_px, "attack_px": None}))
    fit = CourtCalibration.load(path)
    assert set(fit.landmarks) == {"corner_far_left", "corner_far_right",
                                  "corner_near_right", "corner_near_left"}


def test_save_and_load_round_trips_landmarks(tmp_path, calib):
    marks = full_landmarks(calib)
    path = tmp_path / "calibration.json"
    CourtCalibration(marks).save(path)
    back = CourtCalibration.load(path)
    assert set(back.landmarks) == set(marks)
    assert np.allclose(back.H, CourtCalibration(marks).H)


def test_corners_px_is_available_even_when_not_clicked(calib):
    """court_px_width and the old callers ask for the four corners; with a
    partial court those are extrapolated rather than missing."""
    marks = full_landmarks(calib)
    partial = CourtCalibration({k: marks[k] for k in NEAR_HALF})
    assert len(partial.corners_px) == 4
    assert court_px_width(partial.corners_px) > 0


def test_model_lines_render_all_seven(calib):
    lines = CourtCalibration(full_landmarks(calib)).model_lines_px()
    assert len(lines) == 7
    assert all(len(seg) == 2 and len(seg[0]) == 2 for seg in lines)


def test_court_to_px_inverts_to_court(calib):
    pts = np.array([[4.5, 9.0], [1.0, 15.0], [8.0, 3.0]])
    px = calib.court_to_px(pts)
    assert np.allclose(calib.to_court(px), pts, atol=1e-3)


def test_min_landmarks_is_four():
    assert MIN_LANDMARKS == 4
