import cv2
import numpy as np
from conftest import tracks_frame

from rallies import Rally
from rotation import (RallyRole, back_row_attack, group_by_role, rally_role,
                      rotation_coverage, summarise)


def px_for(calib, x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, np.linalg.inv(calib.H)).reshape(2)


def subject_at(calib, make_tracks, positions, fps=30.0):
    """Track rows placing the subject at (court_x, court_y) for each frame."""
    rows = []
    for f, (x, y) in enumerate(positions):
        px, py = px_for(calib, x, y)
        rows.append(tracks_frame(f, 1, float(px), float(py) - 60.0, h=120.0))
    return make_tracks(rows)


def test_serve_zone_is_taken_before_players_switch(calib, make_tracks):
    """A setter starts in zone 1 and runs to the right front. The rotational
    slot is the starting one; reading it after the switch would call him a
    different position entirely."""
    fps = 30.0
    start = [(7.5, 15.0)] * 20        # zone 1, right back, first 0.66 s
    later = [(7.5, 10.0)] * 100       # zone 2, right front, rest of the rally
    subject = subject_at(calib, make_tracks, start + later)
    role = rally_role(Rally(0, 0.0, 4.0), subject, calib, fps)
    assert role.serve_zone == 1
    assert role.play_zone == 2


def test_row_comes_from_the_rotational_slot_not_where_he_ends_up(calib, make_tracks):
    """A back-row player who runs to the front is still back row — that is what
    makes an attack in front of the 3 m line a fault rather than a stat."""
    fps = 30.0
    subject = subject_at(calib, make_tracks,
                         [(4.5, 15.0)] * 20 + [(4.5, 10.5)] * 100)
    role = rally_role(Rally(0, 0.0, 4.0), subject, calib, fps)
    assert role.serve_zone == 6
    assert role.row == "back"
    assert role.play_zone == 3


def test_specialist_role_comes_from_where_he_actually_plays(calib, make_tracks):
    subject = subject_at(calib, make_tracks, [(1.5, 10.0)] * 60)
    role = rally_role(Rally(0, 0.0, 2.0), subject, calib, 30.0)
    assert role.play_zone == 4
    assert role.role == "outside hitter"


def test_role_is_none_when_the_subject_is_untracked(calib, make_tracks):
    """Better an honest gap than a position invented from no data."""
    subject = subject_at(calib, make_tracks, [(4.5, 14.0)] * 5)
    role = rally_role(Rally(0, 100.0, 105.0), subject, calib, 30.0)
    assert role.serve_zone is None and role.play_zone is None
    assert role.role is None and role.row is None


def test_back_row_attack_is_flagged_only_in_front_of_the_attack_line():
    back = RallyRole(0, 6, 3, "back", "middle blocker", "middle back")
    front = RallyRole(0, 3, 3, "front", "middle blocker", "middle front")
    assert back_row_attack(back, contact_y=10.5)      # inside the 3 m line
    assert not back_row_attack(back, contact_y=13.5)  # legal back-row attack
    assert not back_row_attack(front, contact_y=10.5)


def test_summarise_counts_every_position_played():
    roles = [
        RallyRole(0, 4, 4, "front", "outside hitter", "left front"),
        RallyRole(1, 4, 4, "front", "outside hitter", "left front"),
        RallyRole(2, 3, 3, "front", "middle blocker", "middle front"),
        RallyRole(3, 5, 5, "back", "left back", "left back"),
    ]
    out = summarise(roles)
    assert out["rallies"] == 4
    assert out["primary_role"] == "outside hitter"
    assert out["positions_played"] == 3
    assert out["by_row"] == {"front": 3, "back": 1}


def test_summarise_separates_rallies_with_no_role():
    roles = [RallyRole(0, 4, 4, "front", "outside hitter", "left front"),
             RallyRole(1, None, None, None, None, None)]
    out = summarise(roles)
    assert out["rallies"] == 2
    assert out["rallies_with_role"] == 1


def test_group_by_role_indexes_rallies_for_slicing():
    roles = [RallyRole(0, 4, 4, "front", "outside hitter", "left front"),
             RallyRole(1, 3, 3, "front", "middle blocker", "middle front"),
             RallyRole(2, 4, 4, "front", "outside hitter", "left front")]
    assert group_by_role(roles) == {"outside hitter": [0, 2], "middle blocker": [1]}


def test_rotation_coverage_flags_a_lopsided_sample():
    """Five rallies all served from one slot cannot support per-rotation claims,
    and the report needs to know that before making any."""
    lopsided = [RallyRole(i, 1, 1, "back", "right back", "right back")
                for i in range(5)]
    assert rotation_coverage(lopsided)["balanced"] is False
    assert rotation_coverage(lopsided)["slots_seen"] == 1

    full = [RallyRole(i, (i % 6) + 1, 3, "front", "middle blocker", "x")
            for i in range(30)]
    assert rotation_coverage(full)["balanced"] is True
    assert rotation_coverage(full)["slots_seen"] == 6


def test_rotation_coverage_with_nothing_known():
    assert rotation_coverage([])["slots_seen"] == 0
