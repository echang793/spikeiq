import numpy as np
import pytest
from conftest import tracks_frame

from tracking import (assign_sides, feet_px, pick_subject, stitch_subject,
                      wanted_frames)


def test_wanted_frames_without_windows_is_every_stride():
    got = wanted_frames(100, 30.0, 2, None)
    assert got.tolist() == list(range(0, 100, 2))


def test_wanted_frames_restricted_to_rally_windows():
    # 30 fps, keep only 1.0-2.0 s => frames 30..60, plus bridge samples
    got = wanted_frames(300, 30.0, 2, [(1.0, 2.0)], bridge_fps=None)
    assert got.min() == 30 and got.max() == 60
    assert all(f % 2 == 0 for f in got.tolist())


def test_wanted_frames_skips_dead_time_between_rallies():
    got = set(wanted_frames(600, 30.0, 1, [(0.0, 1.0), (5.0, 6.0)],
                            bridge_fps=None).tolist())
    assert 15 in got and 165 in got
    assert 90 not in got  # 3.0 s — between the two rallies


def test_bridge_frames_sample_the_dead_ball_sparsely():
    """Dead time is sampled, not skipped: without these frames the subject's
    identity cannot survive to the next rally. Sparse enough to stay cheap."""
    dense = wanted_frames(3000, 30.0, 2, [(0.0, 1.0)], bridge_fps=2.0)
    sparse_only = [f for f in dense.tolist() if f > 60]
    assert sparse_only, "dead ball was skipped entirely"
    gaps = set(np.diff(sparse_only).tolist())
    assert len(gaps) == 1, f"uneven bridge spacing: {gaps}"
    spacing = gaps.pop()
    assert 30.0 / spacing == pytest.approx(2.0, abs=0.3)   # ~2 fps


def test_bridging_costs_a_small_fraction_of_full_tracking():
    """The whole point is that identity survives for a modest surcharge, not
    that we track the dead ball properly."""
    windows = [(t, t + 8.0) for t in range(10, 300, 30)]
    rallies_only = len(wanted_frames(9000, 30.0, 2, windows, bridge_fps=None))
    bridged = len(wanted_frames(9000, 30.0, 2, windows, bridge_fps=2.0))
    assert 1.0 < bridged / rallies_only < 1.5


def test_model_defaults_to_the_larger_pose_model(monkeypatch):
    """The nano model benchmarks ~1.9x faster and lost nothing on the test
    clip, but that clip is large near-court players. Under-detecting a small
    far-court figure breaks subject resolution and attribution, which costs
    more than the wait, so the safe model stays the default."""
    from tracking import DEFAULT_MODEL, model_name
    monkeypatch.delenv("SPIKEIQ_MODEL", raising=False)
    assert model_name() == DEFAULT_MODEL == "yolov8s-pose.pt"


def test_model_can_be_switched_by_environment(monkeypatch):
    from tracking import model_name
    monkeypatch.setenv("SPIKEIQ_MODEL", "yolov8n-pose.pt")
    assert model_name() == "yolov8n-pose.pt"


def test_bridge_gap_allowance_exceeds_the_bridge_spacing():
    """A stitching allowance smaller than the sample spacing guarantees the
    chain snaps at the first dead ball — the original bug."""
    from tracking import bridge_gap_frames
    assert bridge_gap_frames(30.0, 2.0) > 30.0 / 2.0
    assert bridge_gap_frames(60.0, 2.0) > 60.0 / 2.0


def test_feet_px_is_bottom_centre(make_tracks):
    df = make_tracks([tracks_frame(0, 1, 100.0, 200.0, h=100.0, w=40.0)])
    assert feet_px(df).tolist() == [[100.0, 250.0]]


def test_pick_subject_matches_the_nearest_track_to_the_click(make_tracks):
    df = make_tracks([
        tracks_frame(0, 1, 100.0, 100.0),
        tracks_frame(0, 2, 500.0, 100.0),
        tracks_frame(0, 3, 900.0, 100.0),
    ])
    assert pick_subject(df, (510.0, 105.0), near_frame=0) == 2


def test_pick_subject_is_scoped_to_the_clicked_frame(make_tracks):
    """A different player drifting under the click point 20 s later must not
    steal the identity — with twelve players that happens constantly."""
    df = make_tracks([
        tracks_frame(0, 1, 100.0, 100.0),
        tracks_frame(600, 2, 101.0, 100.0),
    ])
    assert pick_subject(df, (100.0, 100.0), near_frame=0) == 1


def test_stitch_subject_follows_an_id_switch(make_tracks):
    """The tracker drops id 1 and re-ids the same player as 7 a few frames
    later, a few pixels away — the chain must survive it."""
    rows = [tracks_frame(f, 1, 100.0 + f, 200.0) for f in range(0, 20)]
    rows += [tracks_frame(f, 7, 100.0 + f, 200.0) for f in range(22, 40)]
    rows += [tracks_frame(f, 3, 900.0, 500.0) for f in range(0, 40)]  # someone else
    df = make_tracks(rows)
    out = stitch_subject(df, 1)
    assert set(out["track_id"]) == {1, 7}
    assert 3 not in set(out["track_id"])
    assert out["frame"].is_monotonic_increasing


def test_stitch_subject_refuses_a_teleport(make_tracks):
    """A candidate on the far side of the frame is a different person, no
    matter that its id starts right when ours ends."""
    rows = [tracks_frame(f, 1, 100.0, 200.0) for f in range(0, 20)]
    rows += [tracks_frame(f, 7, 1500.0, 900.0) for f in range(21, 40)]
    df = make_tracks(rows)
    assert set(stitch_subject(df, 1)["track_id"]) == {1}


def test_stitch_subject_deduplicates_overlapping_fragments(make_tracks):
    rows = [tracks_frame(f, 1, 100.0, 200.0) for f in range(0, 20)]
    rows += [tracks_frame(f, 7, 101.0, 200.0) for f in range(15, 30)]
    df = make_tracks(rows)
    out = stitch_subject(df, 1)
    assert out["frame"].duplicated().sum() == 0


def test_stitch_subject_rejects_an_unknown_id(make_tracks):
    df = make_tracks([tracks_frame(0, 1, 100.0, 100.0)])
    with pytest.raises(ValueError):
        stitch_subject(df, 99)


def test_assign_sides_splits_teams_across_the_net(calib, make_tracks, corners_px):
    """Two players, one in each half, must land on opposite sides."""
    import cv2
    Hinv = np.linalg.inv(calib.H)

    def px_for(x, y):
        pt = np.array([[[x, y]]], dtype=np.float32)
        return cv2.perspectiveTransform(pt, Hinv).reshape(2)

    rows = []
    fx, fy = px_for(4.5, 4.0)    # far half
    nx, ny = px_for(4.5, 14.0)   # near half
    for f in range(30):
        rows.append(tracks_frame(f, 1, fx, fy - 60.0, h=120.0))
        rows.append(tracks_frame(f, 2, nx, ny - 60.0, h=120.0))
    sides = assign_sides(make_tracks(rows), calib)
    assert sides[1] == "far"
    assert sides[2] == "near"


def test_assign_sides_ignores_brief_and_off_court_tracks(calib, make_tracks):
    rows = [tracks_frame(f, 9, 5.0, 5.0) for f in range(3)]  # 3 frames, top-left
    assert assign_sides(make_tracks(rows), calib) == {}
