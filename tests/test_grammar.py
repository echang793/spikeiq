from contacts import ContactFeatures
from grammar import decode, emission_scores


def feat(t, side, *, x=4.5, y=14.0, zone=6, net_dist=5.0, strength=0.5,
         hand_height=0.0, hand_spread=0.5, airborne=0.0, hand_speed=1.0,
         confidence=1.0, track_id=1):
    return ContactFeatures(
        t=t, frame=int(t * 30), track_id=track_id, side=side, x=x, y=y,
        zone=zone, net_dist=net_dist, strength=strength,
        hand_height=hand_height, hand_spread=hand_spread, airborne=airborne,
        hand_speed=hand_speed, confidence=confidence,
    )


def serve_feat(t, side="near"):
    return feat(t, side, y=19.5, net_dist=10.5, zone=None, strength=0.9,
                hand_height=0.9, hand_speed=6.0, hand_spread=1.0)


def platform_feat(t, side, **kw):
    """Forearm contact — the pose a pass and a dig share exactly."""
    return feat(t, side, hand_height=-0.5, hand_spread=0.2, airborne=0.0,
                hand_speed=1.5, strength=0.5, net_dist=5.5, **kw)


def set_feat(t, side, **kw):
    return feat(t, side, hand_height=0.8, hand_spread=0.35, airborne=0.05,
                hand_speed=2.0, strength=0.25, net_dist=2.5, **kw)


def attack_feat(t, side, **kw):
    return feat(t, side, hand_height=1.1, hand_spread=1.2, airborne=0.7,
                hand_speed=9.0, strength=0.95, net_dist=1.2, **kw)


def block_feat(t, side, **kw):
    return feat(t, side, hand_height=1.0, hand_spread=0.5, airborne=0.6,
                hand_speed=2.0, strength=0.2, net_dist=0.4, **kw)


def test_textbook_rally_is_decoded_in_order():
    seq = [serve_feat(0.0, "near"),
           platform_feat(1.0, "far"),
           set_feat(1.8, "far"),
           attack_feat(2.6, "far")]
    assert [p.action for p in decode(seq)] == ["serve", "pass", "set", "attack"]


def test_identical_platform_pose_is_a_pass_after_a_serve():
    seq = [serve_feat(0.0, "near"), platform_feat(1.0, "far")]
    assert decode(seq)[1].action == "pass"


def test_identical_platform_pose_is_a_dig_after_an_attack():
    """Same body, different context. This is the whole reason for the decoder:
    per-contact classification cannot tell these two apart at all."""
    seq = [serve_feat(0.0, "near"),
           platform_feat(1.0, "far"),
           set_feat(1.8, "far"),
           attack_feat(2.6, "far"),
           platform_feat(3.2, "near")]
    plays = decode(seq)
    assert plays[1].action == "pass"
    assert plays[4].action == "dig"


def test_pass_and_dig_are_nearly_indistinguishable_by_pose_alone():
    """Guards the premise above: if these emissions ever diverge sharply,
    the transition model is no longer carrying the disambiguation."""
    scores = emission_scores(platform_feat(0.0, "near"))
    assert abs(scores["pass"] - scores["dig"]) < 0.20


def test_a_rally_opens_with_a_serve():
    plays = decode([serve_feat(0.0, "near")])
    assert plays[0].action == "serve"


def test_block_does_not_consume_one_of_the_three_touches():
    seq = [serve_feat(0.0, "near"),
           platform_feat(1.0, "far"),
           set_feat(1.8, "far"),
           attack_feat(2.6, "far"),
           block_feat(2.8, "near"),
           platform_feat(3.4, "near"),
           set_feat(4.0, "near"),
           attack_feat(4.6, "near")]
    plays = decode(seq)
    assert plays[4].action == "block"
    assert plays[4].touch_index == 0
    # the blocking side still gets a full three touches afterwards
    assert [p.touch_index for p in plays[5:]] == [1, 2, 3]


def test_touch_index_counts_within_a_possession():
    seq = [serve_feat(0.0, "near"),
           platform_feat(1.0, "far"),
           set_feat(1.8, "far"),
           attack_feat(2.6, "far")]
    plays = decode(seq)
    assert [p.touch_index for p in plays] == [1, 1, 2, 3]


def test_four_touches_on_one_side_are_penalised_but_still_decoded():
    """A missed soft touch must not derail the rest of the rally, so an illegal
    fourth touch is penalised rather than declared impossible."""
    seq = [serve_feat(0.0, "near"),
           platform_feat(1.0, "far"),
           set_feat(1.6, "far"),
           set_feat(2.1, "far"),
           attack_feat(2.7, "far")]
    plays = decode(seq)
    assert len(plays) == len(seq)
    assert plays[-1].action == "attack"


def test_confidence_is_scaled_by_how_much_pose_was_visible():
    clear = decode([serve_feat(0.0, "near"), attack_feat(2.0, "far")])
    occluded = decode([serve_feat(0.0, "near"),
                       attack_feat(2.0, "far", confidence=0.25)])
    assert occluded[1].confidence < clear[1].confidence


def test_empty_rally_decodes_to_nothing():
    assert decode([]) == []
