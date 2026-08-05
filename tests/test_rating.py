import pytest

from feedback import build, per_role_notes, tips_for
from rating import (RUBRIC, band_for, estimate, extract_dimension_values,
                    interp_band, next_anchor_target, strengths_and_weaknesses)


def metrics(hitting=0.15, rating_val=2.15, ace=0.08, err=0.11, stuff=0.16,
            conv=0.52, assist=0.30, low=False):
    return {
        "attacking": {"hitting_pct": hitting, "attempts": 40, "kills": 14,
                      "errors": 8, "low_sample": low},
        "passing": {"rating": rating_val, "attempts": 30, "shank_pct": 0.05,
                    "low_sample": low},
        "serving": {"ace_pct": ace, "error_pct": err, "attempts": 25,
                    "low_sample": low},
        "blocking": {"stuff_pct": stuff, "attempts": 20, "stuffs": 3,
                     "low_sample": low},
        "defense": {"conversion_pct": conv, "digs": 25, "low_sample": low},
        "setting": {"assist_pct": assist, "attempts": 20, "low_sample": low},
        "coverage": {"rallies_with_winner": 60},
    }


def test_interp_band_hits_the_anchors_exactly():
    anchors = RUBRIC["attacking"]["anchors"]
    assert interp_band(0.15, anchors) == pytest.approx(3.0)
    assert interp_band(0.25, anchors) == pytest.approx(4.0)


def test_interp_band_interpolates_between_anchors():
    anchors = RUBRIC["attacking"]["anchors"]
    assert 3.0 < interp_band(0.20, anchors) < 4.0


def test_interp_band_clamps_outside_the_anchor_range():
    anchors = RUBRIC["attacking"]["anchors"]
    assert interp_band(-5.0, anchors) == 1.0
    assert interp_band(5.0, anchors) == 5.0


def test_inverted_dimension_scores_lower_error_rates_higher():
    """Service errors are the one dimension where less is better."""
    anchors = RUBRIC["serve_control"]["anchors"]
    assert interp_band(0.05, anchors) > interp_band(0.20, anchors)


def test_next_anchor_target_points_upward():
    anchors = RUBRIC["attacking"]["anchors"]
    value, level = next_anchor_target(0.10, anchors)
    assert value == 0.15 and level == 3.0


def test_next_anchor_target_respects_inverted_dimensions():
    anchors = RUBRIC["serve_control"]["anchors"]
    value, level = next_anchor_target(0.13, anchors, invert=True)
    assert value < 0.13 and level > interp_band(0.13, anchors) - 1


def test_next_anchor_target_is_none_at_the_top():
    assert next_anchor_target(1.0, RUBRIC["attacking"]["anchors"]) is None


def test_band_names_follow_the_club_ladder():
    assert band_for(1.0) == "B"
    assert band_for(3.0) == "A"
    assert band_for(5.0) == "Open"


def test_estimate_lands_on_the_expected_band():
    out = estimate(metrics(), jumps={"best_m": 0.55, "count": 12})
    assert out["band"] == "A"
    assert 2.7 < out["level"] < 3.3
    assert out["confidence"] > 0.5


def test_estimate_without_measurable_dimensions():
    out = estimate({}, jumps={})
    assert out["level"] is None
    assert out["confidence"] == 0.0
    assert "note" in out


def test_low_sample_dimensions_still_appear_but_lower_confidence():
    """Hiding a thin dimension would make a short match look complete."""
    solid = estimate(metrics(low=False), jumps={"best_m": 0.55, "count": 12})
    thin = estimate(metrics(low=True), jumps={"best_m": 0.55, "count": 2})
    assert set(thin["dimensions"]) == set(solid["dimensions"])
    assert thin["confidence"] < solid["confidence"]
    assert thin["solid_dimensions"] == 0


def test_extract_skips_dimensions_with_no_measurement():
    m = metrics()
    m["blocking"]["stuff_pct"] = None
    dims = extract_dimension_values(m, {"best_m": 0.5, "count": 8})
    assert "blocking" not in dims
    assert "attacking" in dims


def test_estimate_carries_the_uncalibrated_caveat():
    """The anchors are informed guesses, not fitted — the output has to say so
    or the number reads as a placement it cannot support."""
    out = estimate(metrics(), jumps={"best_m": 0.55, "count": 12})
    assert "uncalibrated" in out["note"]


def test_strengths_and_weaknesses_are_relative_to_the_player():
    out = estimate(metrics(hitting=0.35, rating_val=1.6),
                   jumps={"best_m": 0.55, "count": 12})
    sw = strengths_and_weaknesses(out)
    assert sw["strengths"][0]["dimension"] == "attacking"
    assert sw["weaknesses"][0]["dimension"] == "passing"
    assert sw["strengths"][0]["delta"] > 0 > sw["weaknesses"][0]["delta"]


def test_strengths_and_weaknesses_never_share_a_dimension_with_few_measured():
    """A match without a block or a jump easily lands at 4-5 of the 8 possible
    dimensions, and the naive ranked[:3]/ranked[-3:] slices overlap below 6 —
    the middle dimension must be left out of both, not double-counted."""
    rating = {"dimensions": {
        "attacking": {"level": 1.0, "label": "a", "low_sample": False, "weight": 1},
        "passing": {"level": 2.0, "label": "b", "low_sample": False, "weight": 1},
        "serving": {"level": 3.0, "label": "c", "low_sample": False, "weight": 1},
        "defense": {"level": 4.0, "label": "d", "low_sample": False, "weight": 1},
    }}
    sw = strengths_and_weaknesses(rating)
    weak_names = {d["dimension"] for d in sw["weaknesses"]}
    strong_names = {d["dimension"] for d in sw["strengths"]}
    assert not (weak_names & strong_names)
    # k = min(3, 4 // 2) = 2: the two weakest and two strongest, no overlap
    assert weak_names == {"attacking", "passing"}
    assert strong_names == {"defense", "serving"}


def test_strengths_and_weaknesses_exclude_the_true_middle_dimension():
    """With an odd count the middle dimension is neither a clear strength nor a
    clear weakness, and must be left out of both rather than assigned to one."""
    rating = {"dimensions": {
        "attacking": {"level": 1.0, "label": "a", "low_sample": False, "weight": 1},
        "passing": {"level": 3.0, "label": "b", "low_sample": False, "weight": 1},
        "serving": {"level": 5.0, "label": "c", "low_sample": False, "weight": 1},
    }}
    sw = strengths_and_weaknesses(rating)
    assert [d["dimension"] for d in sw["weaknesses"]] == ["attacking"]
    assert [d["dimension"] for d in sw["strengths"]] == ["serving"]


def test_strengths_and_weaknesses_are_both_empty_with_one_dimension():
    """With only one measured dimension there is no "relative to himself" to
    report — better an honest empty list than calling the same skill both the
    best and the worst."""
    rating = {"dimensions": {
        "attacking": {"level": 3.0, "label": "a", "low_sample": False, "weight": 1},
    }}
    sw = strengths_and_weaknesses(rating)
    assert sw["strengths"] == [] and sw["weaknesses"] == []


def test_tips_lead_with_what_to_work_on():
    r = estimate(metrics(hitting=-0.10), jumps={"best_m": 0.55, "count": 12})
    tips = tips_for(r, metrics(hitting=-0.10))
    assert tips[0]["dimension"] == "attacking"
    assert tips[0]["priority"] == "work on"
    assert "-0.100" in tips[0]["note"] or "-0.1" in tips[0]["note"]


def test_tips_flag_a_thin_sample_in_the_note_itself():
    r = estimate(metrics(hitting=-0.10, low=True), jumps={"best_m": 0.4, "count": 2})
    tips = tips_for(r, metrics(hitting=-0.10, low=True))
    assert any("Small sample" in t["note"] for t in tips)


def test_tip_text_quotes_the_same_denominator_as_the_percentage_shown():
    """'{value}' and '{attempts}' must describe the same population, or the
    sentence contradicts its own numbers when recomputed."""
    m = metrics(hitting=-0.10)
    m["attacking"] = {"hitting_pct": -0.10, "attempts": 14, "kills": 3,
                      "errors": 4, "unscored": 4, "low_sample": False}
    r = estimate(m, jumps={"best_m": 0.55, "count": 12})
    note = tips_for(r, m)[0]["note"]
    assert "on 10 swings" in note      # 14 attempts - 4 unscored = 10 graded
    assert "on 14 swings" not in note


def test_per_role_note_swing_count_excludes_unscored_touches():
    by_role = {"outside hitter": {
        "rallies": 20,
        "attacking": {"hitting_pct": 0.21, "attempts": 18, "unscored": 3},
    }}
    note = per_role_notes(by_role)[0]["note"]
    assert "15 swings" in note
    assert "18 swings" not in note


def test_per_role_notes_refuse_to_judge_a_thin_position():
    by_role = {"outside hitter": {"rallies": 2, "attacking": {"hitting_pct": 1.0,
                                                              "attempts": 1}}}
    notes = per_role_notes(by_role)
    assert "not enough" in notes[0]["note"]


def test_per_role_notes_summarise_a_real_sample():
    by_role = {"outside hitter": {
        "rallies": 20,
        "attacking": {"hitting_pct": 0.21, "attempts": 18},
        "passing": {"rating": 2.3},
        "blocking": {"stuffs": 2},
    }}
    note = per_role_notes(by_role)[0]["note"]
    assert "outside hitter" in note and "18 swings" in note


def test_examples_point_at_rallies_that_show_the_problem():
    """A tip is advice; the same tip attached to the rallies where it happened
    is something you can watch."""
    from feedback import example_rallies
    from rallies import Rally

    def play(action, track_id=1, side="near"):
        return {"t": 1.0, "action": action, "track_id": track_id, "side": side,
                "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
                "confidence": 0.8, "touch_index": 1}

    rallies = [Rally(0, 0.0, 5.0, winner="near"),     # his attack won it
               Rally(1, 10.0, 15.0, winner="far"),    # his attack lost it
               Rally(2, 20.0, 25.0, winner="far")]    # and again
    plays = {i: [play("set", track_id=2), play("attack")] for i in range(3)}
    got = example_rallies("attacking", rallies, plays, {1})
    assert got == [1, 2]


def test_examples_ignore_other_players_mistakes():
    from feedback import example_rallies
    from rallies import Rally

    plays = {0: [{"t": 1.0, "action": "attack", "track_id": 99, "side": "near",
                  "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
                  "confidence": 0.8, "touch_index": 3}]}
    got = example_rallies("attacking", [Rally(0, 0.0, 5.0, winner="far")],
                          plays, {1})
    assert got == []


def test_examples_do_not_flag_a_block_that_kept_the_ball_alive():
    """A block with more contacts after it is 'touch', an ordinary and often
    good outcome, not a demonstrated blocking failure."""
    from feedback import example_rallies
    from rallies import Rally

    def play(action, track_id=1, side="near"):
        return {"t": 1.0, "action": action, "track_id": track_id, "side": side,
                "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
                "confidence": 0.8, "touch_index": 1}

    plays = {0: [play("attack", track_id=2, side="far"),
                play("block", track_id=1, side="near"),
                play("dig", track_id=2, side="far"),
                play("set", track_id=2, side="far")]}
    got = example_rallies("blocking", [Rally(0, 0.0, 5.0, winner="far")],
                          plays, {1})
    assert got == []


def test_examples_flag_a_block_that_was_actually_scored_against():
    from feedback import example_rallies
    from rallies import Rally

    def play(action, track_id=1, side="near"):
        return {"t": 1.0, "action": action, "track_id": track_id, "side": side,
                "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
                "confidence": 0.8, "touch_index": 1}

    plays = {0: [play("attack", track_id=2, side="far"),
                play("block", track_id=1, side="near")]}
    got = example_rallies("blocking", [Rally(0, 0.0, 5.0, winner="far")],
                          plays, {1})
    assert got == [0]


def test_examples_do_not_flag_a_set_whose_fed_attack_never_resolved():
    """The last rally of a set has no winner to read — a set that fed an
    attack we can't score has demonstrated nothing, good or bad."""
    from feedback import example_rallies
    from rallies import Rally

    def play(action, track_id=1, side="near"):
        return {"t": 1.0, "action": action, "track_id": track_id, "side": side,
                "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
                "confidence": 0.8, "touch_index": 1}

    plays = {0: [play("pass", track_id=2), play("set", track_id=1),
                play("attack", track_id=2)]}
    got = example_rallies("setting", [Rally(0, 0.0, 5.0, winner=None)],
                          plays, {1})
    assert got == []


def test_examples_flag_a_set_that_definitely_did_not_lead_to_a_kill():
    from feedback import example_rallies
    from rallies import Rally

    def play(action, track_id=1, side="near"):
        return {"t": 1.0, "action": action, "track_id": track_id, "side": side,
                "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
                "confidence": 0.8, "touch_index": 1}

    plays = {0: [play("pass", track_id=2), play("set", track_id=1),
                play("attack", track_id=2), play("block", track_id=99,
                                                 side="far")]}
    got = example_rallies("setting", [Rally(0, 0.0, 5.0, winner="near")],
                          plays, {1})
    assert got == [0]


def test_build_attaches_examples_only_to_what_to_work_on():
    from rallies import Rally
    m = metrics(hitting=-0.10)
    r = estimate(m, jumps={"best_m": 0.55, "count": 12})
    plays = {0: [{"t": 1.0, "action": "attack", "track_id": 1, "side": "near",
                  "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
                  "confidence": 0.8, "touch_index": 3}]}
    out = build(r, {"overall": m, "by_role": {}},
                [Rally(0, 0.0, 5.0, winner="far")], plays, {1})
    work = [t for t in out["tips"] if t["priority"] == "work on"]
    keep = [t for t in out["tips"] if t["priority"] == "keep"]
    assert any(t.get("examples") for t in work)
    assert all("examples" not in t for t in keep)


def test_build_without_rally_context_still_works():
    """The report must render for a session where plays were never resolved."""
    m = metrics()
    r = estimate(m, jumps={"best_m": 0.55, "count": 12})
    out = build(r, {"overall": m, "by_role": {}})
    assert out["tips"]
    assert all("examples" not in t for t in out["tips"])


def test_build_assembles_the_full_payload():
    m = metrics()
    r = estimate(m, jumps={"best_m": 0.55, "count": 12})
    out = build(r, {"overall": m, "by_role": {}})
    assert out["band"] == "A"
    assert out["tips"] and out["strengths"] and out["weaknesses"]
    assert "uncalibrated" in out["caveat"]
