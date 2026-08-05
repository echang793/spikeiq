import pandas as pd
import pytest

from metrics import (attack_outcome, block_outcome, compute, dig_converted,
                     movement, pass_rating, serve_outcome, set_fed_outcome,
                     set_is_assist, skill_lines)
from rallies import Rally

ME = 1
THEM = 2


def play(action, side, track_id=THEM, t=0.0):
    return {"t": t, "action": action, "track_id": track_id, "side": side,
            "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
            "confidence": 0.8, "touch_index": 1}


def textbook(winner="near", my_side="near", finisher=ME):
    """serve (far) -> pass, set, attack (near). The attack ends the rally."""
    plays = [play("serve", "far"),
             play("pass", my_side, ME),
             play("set", my_side, THEM),
             play("attack", my_side, finisher)]
    return plays, Rally(0, 0.0, 5.0, serving_side="far", winner=winner)


def test_attack_that_ends_the_rally_in_your_favour_is_a_kill():
    plays, _ = textbook(winner="near")
    assert attack_outcome(plays, 3, "near") == "kill"


def test_attack_that_ends_the_rally_against_you_is_an_error():
    plays, _ = textbook(winner="far")
    assert attack_outcome(plays, 3, "far") == "error"


def test_attack_the_opponent_stuffed_is_charged_to_the_attacker():
    plays = [play("serve", "far"), play("pass", "near", ME),
             play("set", "near"), play("attack", "near", ME),
             play("block", "far")]
    assert attack_outcome(plays, 3, "far") == "blocked"


def test_attack_that_stayed_in_play_is_neither_kill_nor_error():
    plays = [play("serve", "far"), play("pass", "near", ME),
             play("set", "near"), play("attack", "near", ME),
             play("dig", "far"), play("set", "far"), play("attack", "far")]
    assert attack_outcome(plays, 3, "near") == "in_play"


def test_attack_outcome_is_unknown_without_a_rally_winner():
    """The last rally of a set has no next serve to read the winner from —
    it must not be silently scored as an error."""
    plays, _ = textbook()
    assert attack_outcome(plays, 3, None) == "unknown"


def test_attack_outcome_is_in_play_without_a_winner_if_more_play_followed():
    """Whether more play followed this attack is knowable from the touch
    sequence alone. An unresolved rally winner must not throw that away — the
    last rally of every set has exactly this shape (a winner-less resolution)
    and it happens every set, not on rare occasions."""
    plays = [play("serve", "far"), play("pass", "near", ME),
             play("set", "near", THEM), play("attack", "near", ME),
             play("dig", "far"), play("set", "far"), play("attack", "far")]
    assert attack_outcome(plays, 3, None) == "in_play"


def test_attack_blocked_charge_still_needs_the_winner():
    """The one sub-case that genuinely can't be settled by touch sequence
    alone: whether the block that immediately followed actually stuffed it."""
    plays = [play("serve", "far"), play("pass", "near", ME),
             play("set", "near", THEM), play("attack", "near", ME),
             play("block", "far")]
    assert attack_outcome(plays, 3, None) == "unknown"


def test_an_untouched_serve_is_an_ace():
    plays = [play("serve", "near", ME)]
    assert serve_outcome(plays, 0, "near") == "ace"
    assert serve_outcome(plays, 0, "far") == "error"


def test_a_returned_serve_is_neither():
    plays = [play("serve", "near", ME), play("pass", "far")]
    assert serve_outcome(plays, 0, "near") == "in_play"


def test_a_returned_serve_is_in_play_even_without_a_winner():
    """The serve was clearly touched — that needs no winner to know."""
    plays = [play("serve", "near", ME), play("pass", "far")]
    assert serve_outcome(plays, 0, None) == "in_play"


def test_an_untouched_serve_with_no_winner_is_unknown():
    plays = [play("serve", "near", ME)]
    assert serve_outcome(plays, 0, None) == "unknown"


@pytest.mark.parametrize("rest,want", [
    ([("set", "near"), ("attack", "near")], 3),   # in system
    ([("attack", "near")], 2),                    # out of system, still attacked
    ([("set", "near"), ("pass", "far")], 2),      # set went over
    ([("pass", "far")], 1),                       # sent straight back over
    ([], 0),                                      # shanked, rally over
])
def test_pass_rating_grades_what_the_pass_made_possible(rest, want):
    plays = [play("serve", "far"), play("pass", "near", ME)]
    plays += [play(a, s) for a, s in rest]
    assert pass_rating(plays, 1) == want


def test_set_is_an_assist_only_when_the_attack_scored():
    scored = [play("pass", "near"), play("set", "near", ME),
              play("attack", "near")]
    assert set_is_assist(scored, 1, "near") is True
    assert set_is_assist(scored, 1, "far") is False


def test_set_fed_outcome_distinguishes_no_kill_from_unknown():
    """'definitely not a kill' and 'can't say' must not collapse to the same
    thing — one is real information for a denominator, the other is a gap."""
    fed_error = [play("pass", "near"), play("set", "near", ME),
                play("attack", "near"), play("block", "far")]
    # the attack is the last touch of the rally either way; only `winner`
    # differs between the "kill" and "unknown" cases below
    fed_last_touch = [play("pass", "near"), play("set", "near", ME),
                      play("attack", "near")]
    no_attack = [play("pass", "near"), play("set", "near", ME),
                play("pass", "far")]
    assert set_fed_outcome(fed_error, 1, "far") == "no_kill"
    assert set_fed_outcome(fed_last_touch, 1, None) == "unknown"
    assert set_fed_outcome(fed_last_touch, 1, "near") == "kill"
    assert set_fed_outcome(no_attack, 1, "near") == "no_attack"
    # set_is_assist must keep returning exactly what it always did
    assert set_is_assist(fed_last_touch, 1, "near") is True
    assert set_is_assist(fed_last_touch, 1, None) is False


def test_block_that_ends_the_rally_is_a_stuff():
    plays = [play("attack", "far"), play("block", "near", ME)]
    assert block_outcome(plays, 1, "near") == "stuff"
    assert block_outcome(plays, 1, "far") == "error"


def test_block_that_kept_the_ball_alive_is_a_touch():
    plays = [play("attack", "far"), play("block", "near", ME),
             play("dig", "far"), play("set", "far")]
    assert block_outcome(plays, 1, "far") == "touch"


def test_block_that_kept_the_ball_alive_is_a_touch_even_without_a_winner():
    """More contacts followed — that alone proves 'touch', no winner needed.
    Before this fix every block in the last rally of a set was marked
    'unknown' regardless of how the touch sequence actually played out."""
    plays = [play("attack", "far"), play("block", "near", ME),
             play("dig", "far"), play("set", "far")]
    assert block_outcome(plays, 1, None) == "touch"


def test_a_terminal_block_with_no_winner_is_unknown():
    plays = [play("attack", "far"), play("block", "near", ME)]
    assert block_outcome(plays, 1, None) == "unknown"


def test_dig_is_converted_when_your_side_gets_a_swing_out_of_it():
    good = [play("attack", "far"), play("dig", "near", ME),
            play("set", "near"), play("attack", "near")]
    bad = [play("attack", "far"), play("dig", "near", ME), play("pass", "far")]
    assert dig_converted(good, 1) is True
    assert dig_converted(bad, 1) is False


def test_hitting_percentage_matches_the_standard_formula():
    """(kills - errors) / attempts, the rate every volleyball box score uses."""
    rallies, plays = [], {}
    for i in range(10):
        winner = "near" if i < 6 else "far"
        p = [play("serve", "far"), play("pass", "near", ME),
             play("set", "near"), play("attack", "near", ME)]
        rallies.append(Rally(i, i * 10.0, i * 10.0 + 5, winner=winner))
        plays[i] = p
    out = skill_lines(rallies, plays, {ME})["attacking"]
    assert out["attempts"] == 10
    assert out["kills"] == 6 and out["errors"] == 4
    assert out["hitting_pct"] == pytest.approx(0.2)


def test_hitting_percentage_can_be_negative():
    rallies, plays = [], {}
    for i in range(5):
        plays[i] = [play("serve", "far"), play("pass", "near", ME),
                    play("set", "near"), play("attack", "near", ME)]
        rallies.append(Rally(i, i * 10.0, i * 10.0 + 5, winner="far"))
    out = skill_lines(rallies, plays, {ME})["attacking"]
    assert out["hitting_pct"] == -1.0


def test_unscored_rallies_are_excluded_from_the_rate_not_counted_as_errors():
    rallies, plays = [], {}
    for i in range(4):
        plays[i] = [play("serve", "far"), play("pass", "near", ME),
                    play("set", "near"), play("attack", "near", ME)]
        rallies.append(Rally(i, i * 10.0, i * 10.0 + 5,
                             winner="near" if i < 2 else None))
    out = skill_lines(rallies, plays, {ME})["attacking"]
    assert out["attempts"] == 4
    assert out["unscored"] == 2
    assert out["hitting_pct"] == 1.0   # 2 kills, 0 errors, over 2 graded swings


def test_low_sample_is_flagged():
    """Two swings at 100 % must not read like form."""
    rallies, plays = [], {}
    for i in range(2):
        plays[i] = [play("attack", "near", ME)]
        rallies.append(Rally(i, i * 10.0, i * 10.0 + 5, winner="near"))
    out = skill_lines(rallies, plays, {ME})["attacking"]
    assert out["kill_pct"] == 1.0
    assert out["low_sample"] is True


def test_blocking_rate_excludes_unresolved_rallies_from_its_denominator():
    """Mirrors how attacking/serving already work: a block whose rally never
    got a winner is unscored, not counted against stuff_pct's denominator."""
    rallies, plays = [], {}
    for i in range(3):
        # each block is the last touch of its own rally, so it needs `winner`
        plays[i] = [play("attack", "far"), play("block", "near", ME)]
        rallies.append(Rally(i, i * 10.0, i * 10.0 + 5, winner="near"))
    for i in range(3, 5):
        plays[i] = [play("attack", "far"), play("block", "near", ME)]
        rallies.append(Rally(i, i * 10.0, i * 10.0 + 5, winner=None))
    out = skill_lines(rallies, plays, {ME})["blocking"]
    assert out["attempts"] == 5
    assert out["unscored"] == 2
    assert out["stuffs"] == 3
    assert out["stuff_pct"] == 1.0        # 3 stuffs / 3 graded, not / 5


def test_setting_rate_excludes_unresolved_rallies_from_its_denominator():
    rallies, plays = [], {}
    for i in range(4):
        plays[i] = [play("pass", "near"), play("set", "near", ME),
                    play("attack", "near")]
        rallies.append(Rally(i, i * 10.0, i * 10.0 + 5,
                             winner="near" if i < 4 else None))
    plays[4] = [play("pass", "near"), play("set", "near", ME),
               play("attack", "near")]
    rallies.append(Rally(4, 40.0, 45.0, winner=None))
    out = skill_lines(rallies, plays, {ME})["setting"]
    assert out["attempts"] == 5
    assert out["unscored"] == 1
    assert out["assists"] == 4
    assert out["assist_pct"] == 1.0       # 4 assists / 4 graded, not / 5


def test_only_the_subjects_touches_are_counted():
    plays = {0: [play("serve", "far", THEM), play("pass", "near", ME),
                 play("set", "near", THEM), play("attack", "near", THEM)]}
    rallies = [Rally(0, 0.0, 5.0, winner="near")]
    out = skill_lines(rallies, plays, {ME})
    assert out["passing"]["attempts"] == 1
    assert out["attacking"]["attempts"] == 0
    assert out["setting"]["attempts"] == 0


def test_a_stitched_subject_chain_counts_as_one_player():
    """Tracker re-ids fragment the subject; every id in his chain is him."""
    plays = {0: [play("pass", "near", 1), play("set", "near", 7),
                 play("attack", "near", 12)]}
    rallies = [Rally(0, 0.0, 5.0, winner="near")]
    out = skill_lines(rallies, plays, {1, 7, 12})
    assert out["passing"]["attempts"] == 1
    assert out["setting"]["attempts"] == 1
    assert out["attacking"]["attempts"] == 1


def test_movement_only_counts_time_while_the_ball_is_in_play():
    pos = pd.DataFrame({"t": [0.0, 1.0, 2.0, 50.0, 51.0],
                        "x": [0.0, 1.0, 2.0, 0.0, 5.0],
                        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "frame": [0, 30, 60, 1500, 1530]})
    out = movement(pos, [Rally(0, 0.0, 2.0)])
    assert out["distance_m"] == pytest.approx(2.0)
    assert out["rallies"] == 1


def test_movement_ignores_tracker_teleports():
    """A 9 m step between adjacent frames is an id switch, not a sprint."""
    pos = pd.DataFrame({"t": [0.0, 0.1, 0.2], "x": [0.0, 0.5, 9.0],
                        "y": [0.0, 0.0, 0.0], "frame": [0, 3, 6]})
    out = movement(pos, [Rally(0, 0.0, 1.0)])
    assert out["distance_m"] == pytest.approx(0.5)


def test_metres_per_minute_only_counts_the_time_that_contributed_distance():
    """The denominator must match the numerator's scope. A rally with no
    position data contributes nothing to `distance_m` and must not inflate
    the time it's divided by either — that would silently understate the
    rate every time the subject has an unresolved rally mixed with tracked
    ones, an independent failure mode from missing distance data itself."""
    # steps under 2 m each (the teleport-guard threshold) so all of it counts
    pos = pd.DataFrame({"t": [0.0, 0.5, 1.0], "x": [0.0, 1.5, 3.0],
                        "y": [0.0, 0.0, 0.0], "frame": [0, 15, 30]})
    rallies = [Rally(0, 0.0, 1.0), Rally(1, 10.0, 70.0)]  # rally 1: no position data
    out = movement(pos, rallies)
    assert out["rallies"] == 1
    assert out["distance_m"] == pytest.approx(3.0)
    # 3 m over the 1 s that actually produced it, not over 61 s of both rallies
    assert out["metres_per_minute_in_play"] == pytest.approx(180.0)


def test_back_row_attacks_in_front_of_the_line_are_counted_as_faults():
    """A back-row player taking off inside the 3 m line has given the point
    away. It is a fault, not a swing, and the report should say so."""
    from rotation import RallyRole

    back = RallyRole(0, 6, 3, "back", "middle blocker", "middle back")
    plays = {0: [dict(play("attack", "near", ME), y=10.5)]}   # inside the line
    rallies = [Rally(0, 0.0, 5.0, winner="far")]
    out = skill_lines(rallies, plays, {ME}, roles_by_rally={0: back})
    assert out["attacking"]["back_row_faults"] == 1


def test_a_legal_back_row_attack_is_not_a_fault():
    from rotation import RallyRole

    back = RallyRole(0, 6, 3, "back", "middle blocker", "middle back")
    plays = {0: [dict(play("attack", "near", ME), y=13.5)]}   # behind the line
    rallies = [Rally(0, 0.0, 5.0, winner="near")]
    out = skill_lines(rallies, plays, {ME}, roles_by_rally={0: back})
    assert out["attacking"]["back_row_faults"] == 0


def test_a_front_row_attack_at_the_net_is_never_a_fault():
    from rotation import RallyRole

    front = RallyRole(0, 4, 4, "front", "outside hitter", "left front")
    plays = {0: [dict(play("attack", "near", ME), y=10.0)]}
    rallies = [Rally(0, 0.0, 5.0, winner="near")]
    out = skill_lines(rallies, plays, {ME}, roles_by_rally={0: front})
    assert out["attacking"]["back_row_faults"] == 0


def test_faults_are_zero_when_no_role_was_resolved():
    plays = {0: [dict(play("attack", "near", ME), y=10.5)]}
    out = skill_lines([Rally(0, 0.0, 5.0, winner="far")], plays, {ME})
    assert out["attacking"]["back_row_faults"] == 0


def test_compute_slices_the_same_stats_by_position():
    rallies, plays = [], {}
    for i in range(4):
        plays[i] = [play("serve", "far"), play("pass", "near", ME),
                    play("set", "near"), play("attack", "near", ME)]
        rallies.append(Rally(i, i * 10.0, i * 10.0 + 5, winner="near"))
    groups = {"outside hitter": [0, 1], "middle blocker": [2, 3]}
    out = compute(rallies, plays, {ME}, groups)
    assert out["overall"]["attacking"]["kills"] == 4
    assert out["by_role"]["outside hitter"]["attacking"]["kills"] == 2
    assert out["by_role"]["middle blocker"]["attacking"]["kills"] == 2
    assert out["by_role"]["outside hitter"]["rallies"] == 2
    assert out["coverage"]["subject_touches"] == 8
