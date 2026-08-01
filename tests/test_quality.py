import pytest

from quality import assess, gate_rating
from rallies import Rally

ME = 1


def play(action="attack", track_id=ME, side="near"):
    return {"t": 0.0, "action": action, "track_id": track_id, "side": side,
            "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
            "confidence": 0.8, "touch_index": 1}


def good_match(n=30, touches_per_rally=4):
    rallies = [Rally(i, i * 30.0, i * 30.0 + 8.0, winner="near")
               for i in range(n)]
    plays = {i: [play() for _ in range(touches_per_rally)] for i in range(n)}
    subject_ids = {i: {ME} for i in range(n)}
    return rallies, plays, subject_ids


def test_a_healthy_match_passes_every_check():
    rallies, plays, ids = good_match()
    q = assess(rallies, plays, ids, audio_contacts=120, subject_touches=120)
    assert q.usable
    assert q.failures == []
    assert q.headline() == ""


def test_a_short_clip_is_not_a_match():
    rallies, plays, ids = good_match(n=3)
    q = assess(rallies, plays, ids, audio_contacts=12, subject_touches=12)
    assert not q.usable
    assert any(c.name == "rally_count" for c in q.failures)


def test_losing_the_subject_in_most_rallies_fails_loudly():
    """The exact shape of the bug that shipped: a full match analysed, but he
    was only identified in the first few rallies."""
    rallies, plays, ids = good_match(n=30)
    ids = {i: ({ME} if i < 4 else set()) for i in range(30)}
    q = assess(rallies, plays, ids, audio_contacts=120, subject_touches=16)
    assert not q.usable
    failed = {c.name for c in q.failures}
    assert "subject_resolved" in failed
    assert "only 4 of 30 rallies" in str(q.as_dict())


@pytest.mark.parametrize("n,expect", [(1, "1 rally"), (4, "4 rallies")])
def test_messages_agree_with_their_own_numbers(n, expect):
    """Report copy is read by a person; "Only 1 rallies were found" undermines
    the sentence it appears in."""
    rallies, plays, ids = good_match(n=n)
    q = assess(rallies, plays, ids, audio_contacts=n * 4, subject_touches=n)
    text = " ".join(c["message"] for c in q.as_dict()["failures"])
    assert expect in text
    assert "1 rallies" not in text and "1 rally " not in text.replace(expect, "")


def test_unattributable_contacts_fail_loudly():
    rallies, plays, ids = good_match(n=30, touches_per_rally=1)
    q = assess(rallies, plays, ids, audio_contacts=300, subject_touches=30)
    assert not q.usable
    assert any(c.name == "attribution_rate" for c in q.failures)


def test_unknown_winners_fail_loudly():
    rallies, plays, ids = good_match(n=30)
    for r in rallies:
        r.winner = None
    q = assess(rallies, plays, ids, audio_contacts=120, subject_touches=120)
    assert not q.usable
    assert any(c.name == "winners_known" for c in q.failures)


def test_too_few_of_his_own_touches_fails_loudly():
    rallies, plays, ids = good_match(n=30)
    q = assess(rallies, plays, ids, audio_contacts=120, subject_touches=5)
    assert not q.usable
    assert any(c.name == "subject_touches" for c in q.failures)


def test_stats_are_reported_whether_or_not_the_checks_pass():
    rallies, plays, ids = good_match(n=12)
    q = assess(rallies, plays, ids, audio_contacts=60, subject_touches=48)
    assert q.stats["rallies"] == 12
    assert q.stats["rallies_with_subject"] == 12
    assert q.stats["audio_contacts"] == 60


def test_gate_removes_the_level_but_keeps_the_dimensions():
    """The number is what gets believed, so that is what goes. The dimensions
    stay, because they are how you see what went wrong."""
    rating = {"level": 3.1, "band": "A", "confidence": 0.7,
              "dimensions": {"attacking": {"level": 3.4}}, "note": "original"}
    rallies, plays, ids = good_match(n=2)
    q = assess(rallies, plays, ids, audio_contacts=8, subject_touches=8)
    gated = gate_rating(rating, q)
    assert gated["level"] is None and gated["band"] is None
    assert gated["confidence"] == 0.0
    assert gated["suppressed"] is True
    assert gated["dimensions"] == rating["dimensions"]
    assert "No level is shown" in gated["note"]


def test_gate_leaves_a_healthy_rating_untouched():
    rating = {"level": 3.1, "band": "A", "confidence": 0.7, "dimensions": {}}
    rallies, plays, ids = good_match()
    q = assess(rallies, plays, ids, audio_contacts=120, subject_touches=120)
    assert gate_rating(rating, q) == rating


def test_headline_names_the_worst_problem_and_counts_the_rest():
    rallies, plays, ids = good_match(n=2)
    q = assess(rallies, plays, ids, audio_contacts=400, subject_touches=2)
    head = q.headline()
    assert head
    assert "other problem" in head


@pytest.mark.parametrize("n_rallies", [0, 1])
def test_assess_survives_an_empty_or_single_rally_session(n_rallies):
    rallies, plays, ids = good_match(n=n_rallies)
    q = assess(rallies, plays, ids, audio_contacts=0, subject_touches=0)
    assert not q.usable
    assert q.as_dict()["stats"]["rallies"] == n_rallies
