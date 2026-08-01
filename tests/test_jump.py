import numpy as np
import pytest
from test_contacts import frame_of, pose_row

from jump import (Jump, detect_jumps, jump_at, standing_height_px, summarise)


def air_frames(peak_rise_px, torso, fps=30.0, height_m=1.80):
    """How many frames a rise of this size must last to be a real jump.

    Height and hang time are the same measurement twice over: a body that rises
    h metres is airborne for 2*sqrt(2h/g). The detector checks that relation, so
    a fixture that ignores it produces a physically impossible jump and is
    testing nothing.
    """
    height_m_actual = peak_rise_px * (height_m / (2.6 * torso))
    hang_s = 2.0 * np.sqrt(2.0 * height_m_actual / 9.81)
    # widen for the threshold crossing: the span is measured where the arch is
    # above a quarter of its peak, which is ~84 % of the full flight
    return int(round(hang_s * fps / 0.839))


def jumping_track(peak_rise_px=70.0, n=None, start=10, track_id=1, torso=100.0,
                  feet_y=900.0, span=None, fps=30.0):
    """A track that stands still, jumps once, and lands.

    Ankle y follows an arch through the airborne frames; image y grows downward,
    so rising means subtracting. The flight time defaults to whatever gravity
    says it should be for the height.
    """
    lo = start
    hi = lo + (span if span is not None else air_frames(peak_rise_px, torso, fps))
    n = n if n is not None else hi + 12
    rows = []
    for f in range(n):
        ankle = feet_y
        if lo <= f < hi:
            phase = (f - lo) / (hi - lo)
            ankle = feet_y - peak_rise_px * np.sin(np.pi * phase)
        rows.append(pose_row(f, track_id, torso=torso, feet_y=feet_y,
                             ankle_y=ankle))
    return frame_of(rows)


def stepped_track(offset_px=70.0, n=40, at=15, torso=100.0, feet_y=900.0):
    """The tracker-discontinuity case: the ankle baseline steps to a new level
    partway through and never comes back. This is what a stitched subject chain
    looks like when it switches to a different person, and before the physical
    gate it read as a huge jump."""
    return frame_of([
        pose_row(f, 1, torso=torso, feet_y=feet_y,
                 ankle_y=feet_y - (offset_px if f >= at else 0.0))
        for f in range(n)])


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
    a = jumping_track(start=5, n=40)
    b = jumping_track(start=5, n=40).assign(frame=lambda d: d["frame"] + 40)
    jumps = detect_jumps(pd.concat([a, b], ignore_index=True), fps=30.0)
    assert len(jumps) == 2


def test_walking_up_the_court_is_not_a_jump():
    """Found on the first real clip: a player moving from the near court to the
    far court rises hundreds of pixels up the frame and shrinks as they go. Read
    against whole-track constants that scored as a 0.79 m vertical, so both the
    floor line and the pixel ruler have to be local."""
    rows = []
    for f in range(60):
        depth = f / 59.0                     # near court -> far court
        torso = 100.0 - 65.0 * depth         # shrinks with distance
        feet = 900.0 - 520.0 * depth         # rises up the frame
        rows.append(pose_row(f, 1, torso=torso, feet_y=feet))
    assert detect_jumps(frame_of(rows), fps=30.0, subject_height_m=1.83) == []


def test_a_real_jump_survives_while_the_player_is_crossing_the_court():
    """The local baseline must not be so aggressive that it eats a genuine jump
    taken on the move."""
    rows = []
    span = air_frames(40.0, 60.0)
    for f in range(60):
        depth = f / 59.0
        torso = 100.0 - 65.0 * depth
        feet = 900.0 - 520.0 * depth
        ankle = feet
        if 25 <= f < 25 + span:
            ankle = feet - 40.0 * np.sin(np.pi * (f - 25) / span)
        rows.append(pose_row(f, 1, torso=torso, feet_y=feet, ankle_y=ankle))
    assert len(detect_jumps(frame_of(rows), fps=30.0, subject_height_m=1.83)) == 1


def test_a_tracker_id_switch_is_not_a_jump():
    """The subject's stitched chain hops between tracker ids; each hop steps the
    ankle baseline. Requiring a landing — and a hang time gravity agrees with —
    is what stops that reading as a 0.8 m vertical."""
    assert detect_jumps(stepped_track(), fps=30.0) == []


def test_an_impossibly_brief_rise_is_rejected():
    """A half-metre rise that lasts a fifth of a second is not something a body
    can do, whatever the pixels say."""
    quick = jumping_track(peak_rise_px=70.0, span=6, n=40)
    assert detect_jumps(quick, fps=30.0) == []


def test_a_rise_that_never_comes_down_is_rejected():
    rows = jumping_track(peak_rise_px=70.0, span=200, n=30)
    assert detect_jumps(rows, fps=30.0) == []


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
