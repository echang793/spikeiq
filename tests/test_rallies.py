import numpy as np
from conftest import tracks_frame

from audio import Contact, Whistle
from rallies import (Rally, assign_winners, detect_serving_side, segment_rallies,
                     subject_side, summarise)


def whistle_pair(start, end):
    return [Whistle(start, start + 0.3, 3000.0), Whistle(end, end + 0.3, 3000.0)]


def build_match(n=8, cycle=25.0):
    """A referee-officiated match: whistle, contacts, whistle, dead air, repeat."""
    whistles, contacts = [], []
    for i in range(n):
        t0 = 10.0 + i * cycle
        whistles += whistle_pair(t0, t0 + 8.0)
        contacts += [Contact(t0 + 0.5 + j * 1.2, 0.8) for j in range(6)]
    return whistles, contacts


def test_whistle_segmentation_finds_every_rally():
    whistles, contacts = build_match(n=8)
    rallies = segment_rallies(whistles, contacts)
    assert len(rallies) == 8
    assert all(r.source == "whistle" for r in rallies)


def test_intervals_with_no_contacts_are_not_rallies():
    """A timeout is bracketed by whistles too — only contacts tell them apart."""
    whistles, contacts = build_match(n=4)
    whistles += whistle_pair(500.0, 530.0)  # 30 s timeout, no ball touched
    rallies = segment_rallies(whistles, contacts)
    assert len(rallies) == 4
    assert all(r.duration < 45.0 for r in rallies)


def test_an_ace_is_still_a_rally():
    """One contact, one whistle pair. Aces are the single most interesting
    serve outcome; a min-contacts rule would silently delete them."""
    whistles = whistle_pair(10.0, 12.0) + whistle_pair(40.0, 42.0) \
        + whistle_pair(70.0, 72.0) + whistle_pair(100.0, 102.0)
    contacts = [Contact(10.6, 1.0), Contact(40.6, 1.0),
                Contact(70.6, 1.0), Contact(100.6, 1.0)]
    rallies = segment_rallies(whistles, contacts)
    assert len(rallies) == 4
    assert all(len(r.contacts) == 1 for r in rallies)


def test_falls_back_to_gap_segmentation_without_a_referee():
    contacts = []
    for i in range(5):
        t0 = 10.0 + i * 25.0
        contacts += [Contact(t0 + j * 1.0, 0.8) for j in range(5)]
    rallies = segment_rallies([], contacts)
    assert len(rallies) == 5
    assert all(r.source == "gap" for r in rallies)


def test_gap_fallback_ignores_an_isolated_bang():
    contacts = [Contact(1.0, 0.5)]  # a dropped ball in warmup
    contacts += [Contact(30.0 + j * 1.0, 0.8) for j in range(5)]
    rallies = segment_rallies([], contacts)
    assert len(rallies) == 1


def test_long_dead_time_starts_a_new_set():
    whistles, contacts = build_match(n=3)
    later_w, later_c = build_match(n=3, cycle=25.0)
    shift = 600.0
    whistles += [Whistle(w.start + shift, w.end + shift, w.peak_hz) for w in later_w]
    contacts += [Contact(c.t + shift, c.strength) for c in later_c]
    rallies = segment_rallies(whistles, contacts)
    assert {r.set_index for r in rallies} == {0, 1}
    assert sum(1 for r in rallies if r.set_index == 0) == 3


def test_assign_winners_reads_the_next_serve():
    rallies = [
        Rally(0, 0.0, 5.0, serving_side="near"),
        Rally(1, 10.0, 15.0, serving_side="near"),   # near served again => near won r0
        Rally(2, 20.0, 25.0, serving_side="far"),    # far serves now => far won r1
    ]
    assign_winners(rallies)
    assert rallies[0].winner == "near"
    assert rallies[1].winner == "far"
    assert rallies[2].winner is None  # no next serve to read


def test_winners_are_not_chained_across_a_set_break():
    """Teams switch ends between sets, so the next serve says nothing about who
    won the last rally of the previous set."""
    rallies = [
        Rally(0, 0.0, 5.0, set_index=0, serving_side="near"),
        Rally(1, 700.0, 705.0, set_index=1, serving_side="far"),
    ]
    assign_winners(rallies)
    assert rallies[0].winner is None


def test_unknown_serving_side_propagates_as_unknown_not_a_guess():
    rallies = [Rally(0, 0.0, 5.0, serving_side="near"),
               Rally(1, 10.0, 15.0, serving_side=None)]
    assign_winners(rallies)
    assert rallies[0].winner is None


def _px_for(calib, x, y):
    import cv2
    Hinv = np.linalg.inv(calib.H)
    pt = np.array([[[x, y]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, Hinv).reshape(2)


def test_detect_serving_side_finds_the_player_behind_the_endline(calib, make_tracks):
    rows = []
    sx, sy = _px_for(calib, 4.5, 19.2)   # 1.2 m behind the near endline
    ox, oy = _px_for(calib, 4.5, 5.0)    # an opponent, inside the far half
    for f in range(20):
        rows.append(tracks_frame(f, 1, sx, sy - 60.0, h=120.0))
        rows.append(tracks_frame(f, 2, ox, oy - 60.0, h=120.0))
    rally = Rally(0, 0.0, 5.0)
    assert detect_serving_side(rally, make_tracks(rows), calib, fps=30.0) == "near"


def test_detect_serving_side_returns_none_when_nobody_is_behind_a_line(calib, make_tracks):
    rows = []
    ax, ay = _px_for(calib, 4.5, 5.0)
    bx, by = _px_for(calib, 4.5, 14.0)
    for f in range(20):
        rows.append(tracks_frame(f, 1, ax, ay - 60.0, h=120.0))
        rows.append(tracks_frame(f, 2, bx, by - 60.0, h=120.0))
    rally = Rally(0, 0.0, 5.0)
    assert detect_serving_side(rally, make_tracks(rows), calib, fps=30.0) is None


def test_subject_side_is_computed_per_rally(calib, make_tracks):
    """The subject's own tracked position decides his side, so the between-set
    end switch needs no special handling."""
    fx, fy = _px_for(calib, 4.5, 4.0)
    nx, ny = _px_for(calib, 4.5, 14.0)
    first = make_tracks([tracks_frame(f, 1, fx, fy - 60.0, h=120.0) for f in range(30)])
    second = make_tracks([tracks_frame(f, 1, nx, ny - 60.0, h=120.0)
                          for f in range(30)])
    r = Rally(0, 0.0, 1.0)
    assert subject_side(r, first, calib, fps=30.0) == "far"
    assert subject_side(r, second, calib, fps=30.0) == "near"


def test_summarise_reports_nothing_useful_without_rallies():
    out = summarise([])
    assert out["rally_count"] == 0
    assert "note" in out


def test_summarise_counts_sets_and_scored_rallies():
    rallies = [Rally(0, 0.0, 5.0, contacts=[1.0, 2.0], serving_side="near"),
               Rally(1, 10.0, 18.0, contacts=[11.0], serving_side="far")]
    assign_winners(rallies)
    out = summarise(rallies)
    assert out["rally_count"] == 2
    assert out["set_count"] == 1
    assert out["scored_rallies"] == 1
    assert out["serving_side_known"] == 2
