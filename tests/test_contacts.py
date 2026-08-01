import numpy as np
import pandas as pd
import pytest

from contacts import (attribute_contact, airborne_series, behind_endline,
                      contact_features, hand_metrics, is_airborne,
                      torso_length, _mean_xy)
from tracking import COLUMNS


def pose_row(frame, track_id, *, cx=800.0, feet_y=900.0, torso=100.0,
             wrist_dy=40.0, wrist_dx=20.0, ankle_y=None):
    """A track row with anatomically ordered keypoints.

    Image y grows downward: shoulders sit above hips, so shoulder y is the
    smaller number. `wrist_dy` is measured DOWN from the shoulders, so a
    negative value puts the hands overhead.
    """
    sh_y = feet_y - 2.2 * torso
    hip_y = sh_y + torso
    # every offset scales with torso so that a "far court" player is a uniform
    # shrink of a near one — otherwise perspective tests measure the fixture
    head = 0.4 * torso
    row = {
        "frame": frame, "track_id": track_id,
        "x1": cx - 0.4 * torso, "y1": sh_y - head,
        "x2": cx + 0.4 * torso, "y2": feet_y,
        "conf": 0.9,
        "nosex": cx, "nosey": sh_y - 0.25 * torso,
        "lshx": cx - 25.0, "lshy": sh_y, "rshx": cx + 25.0, "rshy": sh_y,
        "lwx": cx - wrist_dx, "lwy": sh_y + wrist_dy,
        "rwx": cx + wrist_dx, "rwy": sh_y + wrist_dy,
        "lhipx": cx - 15.0, "lhipy": hip_y, "rhipx": cx + 15.0, "rhipy": hip_y,
        "lankx": cx - 10.0, "lanky": ankle_y if ankle_y else feet_y,
        "rankx": cx + 10.0, "ranky": ankle_y if ankle_y else feet_y,
    }
    return row


def frame_of(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_mean_xy_uses_the_visible_side_when_one_is_occluded():
    """Averaging a detected keypoint with a (0,0) 'missing' one would drag the
    midpoint towards the origin — with twelve players that is most frames."""
    df = frame_of([pose_row(0, 1)])
    df.loc[0, ["lshx", "lshy"]] = 0.0
    got = _mean_xy(df, "lsh", "rsh")[0]
    assert got.tolist() == [df.loc[0, "rshx"], df.loc[0, "rshy"]]


def test_mean_xy_is_nan_when_both_sides_are_missing():
    df = frame_of([pose_row(0, 1)])
    df.loc[0, ["lshx", "lshy", "rshx", "rshy"]] = 0.0
    assert np.isnan(_mean_xy(df, "lsh", "rsh")[0]).all()


def test_torso_length_is_shoulder_to_hip():
    df = frame_of([pose_row(0, 1, torso=120.0)])
    assert torso_length(df)[0] == pytest.approx(120.0, abs=1.0)


def test_hand_height_is_positive_when_the_hands_are_overhead():
    overhead = frame_of([pose_row(f, 1, wrist_dy=-60.0) for f in range(3)])
    low = frame_of([pose_row(f, 1, wrist_dy=+70.0) for f in range(3)])
    assert hand_metrics(overhead, 30.0)["hand_height"].iloc[0] > 0.5
    assert hand_metrics(low, 30.0)["hand_height"].iloc[0] < -0.5


def test_hand_spread_is_measured_in_torso_lengths():
    """Scale-free by design: a far-court player is a third the pixel size of a
    near-court one, so raw-pixel spread would mean different things."""
    near = frame_of([pose_row(f, 1, torso=120.0, wrist_dx=60.0) for f in range(3)])
    far = frame_of([pose_row(f, 1, torso=40.0, wrist_dx=20.0) for f in range(3)])
    a = hand_metrics(near, 30.0)["hand_spread"].iloc[0]
    b = hand_metrics(far, 30.0)["hand_spread"].iloc[0]
    assert a == pytest.approx(b, abs=0.02)


def test_airborne_series_detects_a_rise_above_the_standing_baseline():
    rows = [pose_row(f, 1) for f in range(20)]
    for f in range(8, 12):                       # four frames off the floor
        rows[f] = pose_row(f, 1, ankle_y=900.0 - 70.0)
    air = airborne_series(frame_of(rows))
    assert air[9] > 0.5
    assert abs(air[0]) < 0.1


def test_airborne_baseline_is_per_track_not_per_frame():
    """A far-court player stands higher in the image than a near-court one; a
    shared baseline would report them as permanently airborne."""
    deep = frame_of([pose_row(f, 1, feet_y=400.0, torso=45.0) for f in range(15)])
    assert np.nanmax(np.abs(airborne_series(deep))) < 0.1


def test_is_airborne_threshold():
    from contacts import ContactFeatures
    base = dict(t=0.0, frame=0, track_id=1, side="near", x=4.5, y=14.0, zone=6,
                net_dist=5.0, strength=0.5, hand_height=0.0, hand_spread=0.5,
                hand_speed=1.0, confidence=1.0)
    assert is_airborne(ContactFeatures(**base, airborne=0.6))
    assert not is_airborne(ContactFeatures(**base, airborne=0.05))


def test_behind_endline_flags_both_serving_ends():
    from contacts import ContactFeatures
    base = dict(t=0.0, frame=0, track_id=1, side="near", x=4.5, zone=None,
                net_dist=10.0, strength=0.9, hand_height=0.8, hand_spread=1.0,
                airborne=0.0, hand_speed=6.0, confidence=1.0)
    assert behind_endline(ContactFeatures(**base, y=19.0))
    assert behind_endline(ContactFeatures(**base, y=-0.5))
    assert not behind_endline(ContactFeatures(**base, y=14.0))


def test_attribute_contact_picks_the_player_whose_hands_move_fastest():
    rows = []
    for f in range(0, 20):
        rows.append(pose_row(f, 1, cx=400.0))                 # standing still
        rows.append(pose_row(f, 2, cx=800.0, wrist_dx=20.0 + 25.0 * f))  # swinging
    assert attribute_contact(0.33, frame_of(rows), fps=30.0) == 2


def test_attribute_contact_returns_none_when_nobody_is_tracked():
    rows = [pose_row(f, 1, cx=400.0) for f in range(5)]
    assert attribute_contact(50.0, frame_of(rows), fps=30.0) is None


def test_contact_features_places_the_touch_on_the_court(calib):
    import cv2
    Hinv = np.linalg.inv(calib.H)
    px = cv2.perspectiveTransform(
        np.array([[[4.5, 14.0]]], dtype=np.float32), Hinv).reshape(2)
    rows = [pose_row(f, 1, cx=float(px[0]), feet_y=float(px[1]),
                     wrist_dx=20.0 + 20.0 * f) for f in range(10)]
    f = contact_features(0.15, 0.8, frame_of(rows), calib, fps=30.0)
    assert f is not None
    assert f.side == "near"
    assert f.x == pytest.approx(4.5, abs=0.4)
    assert f.y == pytest.approx(14.0, abs=0.4)
    assert f.zone == 6
    assert 0.0 < f.confidence <= 1.0


def test_contact_features_returns_none_when_unattributable(calib):
    empty = frame_of([])
    assert contact_features(1.0, 0.5, empty, calib, fps=30.0) is None
