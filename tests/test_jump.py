import numpy as np
import pytest
from test_contacts import frame_of, pose_row

from jump import (Jump, detect_jumps, jump_at, standing_height_px, summarise)


def jumping_track(peak_rise_px=70.0, n=40, at=(10, 16), track_id=1, torso=100.0,
                  feet_y=900.0):
    """A track that stands still, jumps once, and lands.

    Ankle y follows a parabola through the airborne frames; image y grows
    downward, so rising means subtracting.
    """
    rows = []
    lo, hi = at
    for f in range(n):
        ankle = feet_y
        if lo <= f < hi:
            phase = (f - lo) / (hi - lo)
            ankle = feet_y - peak_rise_px * np.sin(np.pi * phase)
        rows.append(pose_row(f, track_id, torso=torso, feet_y=feet_y,
                             ankle_y=ankle))
    return frame_of(rows)


def test_standing_height_px_uses_the_tall_boxes():
    """A crouched or clipped box would read as a shorter person and inflate
    every jump measured against it."""
    rows = [pose_row(f, 1, torso=100.0) for f in range(10)]
    df = frame_of(rows)
    df.loc[0:3, "y1"] = df.loc[0:3, "y2"] - 80.0   # four badly-cropped boxes
    assert standing_height_px(df) > 200.0


def test_detects_a_single_jump():
    jumps = detect_jumps(jumping_track(), fps=30.0, subject_height_m=1.80)
    assert len(jumps) == 1
    assert jumps[0].track_id == 1


def test_jump_height_is_scaled_by_the_players_real_height():
    """The ruler is the player's own standing height in pixels, so a taller
    person measured over the same pixel rise gets a taller jump."""
    track = jumping_track(peak_rise_px=70.0)
    short = detect_jumps(track, fps=30.0, subject_height_m=1.60)[0]
    tall = detect_jumps(track, fps=30.0, subject_height_m=2.00)[0]
    assert tall.height_m > short.height_m
    assert tall.height_m / short.height_m == pytest.approx(2.00 / 1.60, rel=0.01)


def test_jump_height_is_perspective_free():
    """The same real jump filmed at the far endline is fewer pixels of rise but
    also fewer pixels of standing height — the ratio must cancel out."""
    near = jumping_track(peak_rise_px=70.0, torso=100.0, feet_y=900.0)
    far = jumping_track(peak_rise_px=70.0 / 3, torso=100.0 / 3, feet_y=350.0)
    a = detect_jumps(near, fps=30.0, subject_height_m=1.8)[0].height_m
    b = detect_jumps(far, fps=30.0, subject_height_m=1.8)[0].height_m
    assert a == pytest.approx(b, rel=0.05)


def test_standing_still_is_not_a_jump():
    rows = [pose_row(f, 1) for f in range(40)]
    assert detect_jumps(frame_of(rows), fps=30.0) == []


def test_keypoint_jitter_is_not_a_jump():
    rng = np.random.default_rng(0)
    rows = [pose_row(f, 1, ankle_y=900.0 + rng.normal(0, 3.0)) for f in range(40)]
    assert detect_jumps(frame_of(rows), fps=30.0) == []


def test_two_separate_jumps_are_both_kept():
    import pandas as pd
    a = jumping_track(at=(5, 11), n=20)
    b = jumping_track(at=(5, 11), n=20)
    b = b.assign(frame=b["frame"] + 40)
    jumps = detect_jumps(pd.concat([a, b], ignore_index=True), fps=30.0)
    assert len(jumps) == 2


def test_takeoff_precedes_the_peak():
    j = detect_jumps(jumping_track(), fps=30.0)[0]
    assert j.takeoff_t < j.t


def test_jump_at_matches_a_contact_to_its_jump():
    jumps = [Jump(t=2.00, height_m=0.55, takeoff_t=1.8, track_id=1),
             Jump(t=9.00, height_m=0.40, takeoff_t=8.8, track_id=1)]
    assert jump_at(jumps, 2.10).height_m == 0.55
    assert jump_at(jumps, 9.20).height_m == 0.40


def test_jump_at_returns_none_for_a_grounded_contact():
    """A standing set or a floor dig genuinely has no jump, and reporting the
    nearest one seconds away would invent a number."""
    jumps = [Jump(t=2.0, height_m=0.55, takeoff_t=1.8, track_id=1)]
    assert jump_at(jumps, 6.0) is None


def test_summarise_reports_best_and_median():
    jumps = [Jump(t=float(i), height_m=h, takeoff_t=0.0, track_id=1)
             for i, h in enumerate([0.40, 0.55, 0.50])]
    out = summarise(jumps)
    assert out["count"] == 3
    assert out["best_m"] == 0.55
    assert out["median_m"] == 0.50


def test_summarise_on_no_jumps():
    assert summarise([])["count"] == 0


def test_empty_track_yields_no_jumps():
    assert detect_jumps(frame_of([]), fps=30.0) == []
