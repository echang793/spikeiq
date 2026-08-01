import json

import pytest

from rallies import Rally
from review import Corrections, apply_actions, review_payload


def play(action="attack", confidence=0.8, track_id=1):
    return {"t": 1.0, "action": action, "track_id": track_id, "side": "near",
            "zone": 4, "x": 4.5, "y": 10.0, "airborne": 0.0,
            "confidence": confidence, "touch_index": 1}


def subject_doc(conf_by_rally, method="proximity"):
    return {
        "ids_by_rally": {str(i): [1] for i in conf_by_rally},
        "confidence_by_rally": {str(i): c for i, c in conf_by_rally.items()},
        "method_by_rally": {str(i): method for i in conf_by_rally},
    }


def test_corrections_round_trip(tmp_path):
    c = Corrections(subject={2: [7]}, actions={3: {0: "set"}}, deleted={5})
    c.save(tmp_path)
    back = Corrections.load(tmp_path)
    assert back.subject == {2: [7]}
    assert back.actions == {3: {0: "set"}}
    assert back.deleted == {5}


def test_corrections_load_on_missing_or_broken_file(tmp_path):
    assert Corrections.load(tmp_path).is_empty()
    (tmp_path / "corrections.json").write_text("{not json")
    assert Corrections.load(tmp_path).is_empty()


def test_merge_records_a_subject_fix():
    c = Corrections().merge({"subject": {"4": [9, 12]}})
    assert c.subject == {4: [9, 12]}


def test_merge_with_null_clears_a_correction():
    """A misclick has to be undoable, or the first wrong tap is permanent."""
    c = Corrections(subject={4: [9]}).merge({"subject": {"4": None}})
    assert 4 not in c.subject


def test_merge_records_and_clears_action_labels():
    c = Corrections().merge({"actions": {"2": {"0": "set", "1": "dig"}}})
    assert c.actions == {2: {0: "set", 1: "dig"}}
    c.merge({"actions": {"2": {"0": None}}})
    assert c.actions == {2: {1: "dig"}}
    c.merge({"actions": {"2": {"1": None}}})
    assert 2 not in c.actions


def test_merge_toggles_a_deleted_rally_both_ways():
    c = Corrections().merge({"deleted": {"3": True}})
    assert c.deleted == {3}
    c.merge({"deleted": {"3": False}})
    assert c.deleted == set()


def test_apply_actions_overrides_the_decoder():
    plays = [play("attack"), play("dig")]
    out = apply_actions(plays, Corrections(actions={0: {1: "block"}}), 0)
    assert out[0]["action"] == "attack"
    assert out[1]["action"] == "block"
    assert out[1]["confidence"] == 1.0
    assert out[1]["corrected"] is True


def test_apply_actions_is_a_no_op_without_corrections():
    plays = [play("attack")]
    assert apply_actions(plays, Corrections(), 0) == plays


def test_review_orders_the_worst_rallies_first():
    """The whole point of the queue: attention goes where it changes answers."""
    rallies = [Rally(0, 0.0, 8.0, contacts=[1.0], winner="near"),
               Rally(1, 30.0, 38.0, contacts=[31.0], winner="near"),
               Rally(2, 60.0, 68.0, contacts=[61.0], winner="near")]
    plays = {0: [play(confidence=0.9)], 1: [play(confidence=0.9)],
             2: [play(confidence=0.9)]}
    subject = subject_doc({0: 0.9, 1: 0.1, 2: 0.5})
    out = review_payload(rallies, plays, subject, None, 30.0, Corrections())
    assert [r["index"] for r in out["rallies"]] == [1, 2, 0]


def test_a_rally_with_no_subject_rises_to_the_top():
    rallies = [Rally(0, 0.0, 8.0, contacts=[1.0], winner="near"),
               Rally(1, 30.0, 38.0, contacts=[31.0], winner="near")]
    subject = subject_doc({0: 0.95, 1: 0.0})
    subject["ids_by_rally"]["1"] = []
    out = review_payload(rallies, {0: [play()], 1: []}, subject, None, 30.0,
                         Corrections())
    assert out["rallies"][0]["index"] == 1
    assert "not identified" in out["rallies"][0]["why"]


def test_an_unscored_rally_is_prioritised():
    rallies = [Rally(0, 0.0, 8.0, contacts=[1.0], winner="near"),
               Rally(1, 30.0, 38.0, contacts=[31.0], winner=None)]
    subject = subject_doc({0: 0.9, 1: 0.9}, method="bridged")
    out = review_payload(rallies, {0: [play()], 1: [play()]}, subject, None,
                         30.0, Corrections())
    assert out["rallies"][0]["index"] == 1
    assert "No winner" in out["rallies"][0]["why"]


def test_a_confident_proximity_match_does_not_cry_wolf():
    """Flagging every proximity rally as suspect would bury the ones where
    something is genuinely missing."""
    rallies = [Rally(0, 0.0, 8.0, contacts=[1.0], winner="near")]
    out = review_payload(rallies, {0: [play(confidence=0.9)]},
                         subject_doc({0: 0.44}), None, 30.0, Corrections())
    assert "guessed from position" in out["rallies"][0]["why"]

    out2 = review_payload(rallies, {0: [play(confidence=0.9)]},
                          subject_doc({0: 0.8}), None, 30.0, Corrections())
    assert out2["rallies"][0]["why"] == ""


def test_already_corrected_rallies_sink_to_the_bottom():
    """Once you have answered, you should not be asked again."""
    rallies = [Rally(0, 0.0, 8.0, contacts=[1.0], winner="near"),
               Rally(1, 30.0, 38.0, contacts=[31.0], winner="near")]
    subject = subject_doc({0: 0.1, 1: 0.9})
    out = review_payload(rallies, {0: [play()], 1: [play()]}, subject, None,
                         30.0, Corrections(subject={0: [1]}))
    assert out["rallies"][-1]["index"] == 0
    assert out["rallies"][-1]["priority"] == 0.0
    assert out["rallies"][-1]["subject"]["corrected"] is True


def test_thumbnail_lands_just_after_the_serve():
    """Players are spread and unoccluded right after contact — the easiest
    moment to recognise yourself."""
    rallies = [Rally(0, 10.0, 18.0, contacts=[11.5, 13.0])]
    out = review_payload(rallies, {0: []}, subject_doc({0: 0.5}), None, 30.0,
                         Corrections())
    assert out["rallies"][0]["thumb_t"] == pytest.approx(11.8)


def test_thumbnail_falls_back_when_a_rally_has_no_contacts():
    rallies = [Rally(0, 10.0, 18.0, contacts=[])]
    out = review_payload(rallies, {0: []}, subject_doc({0: 0.5}), None, 30.0,
                         Corrections())
    assert out["rallies"][0]["thumb_t"] == pytest.approx(10.5)


def test_candidate_boxes_let_a_click_become_a_track_id(make_tracks):
    from conftest import tracks_frame
    rows = []
    for f in range(300, 361, 2):
        rows.append(tracks_frame(f, 1, 400.0, 500.0))
        rows.append(tracks_frame(f, 2, 900.0, 520.0))
    rallies = [Rally(0, 10.0, 18.0, contacts=[10.2])]
    out = review_payload(rallies, {0: []}, subject_doc({0: 0.4}),
                         make_tracks(rows), 30.0, Corrections())
    boxes = out["rallies"][0]["candidates"]
    assert {b["track_id"] for b in boxes} == {1, 2}
    assert all(len(b["box"]) == 4 for b in boxes)


def test_deleted_rallies_are_marked_but_still_listed():
    """Deleting is reversible, so the rally has to stay visible."""
    rallies = [Rally(0, 0.0, 8.0, contacts=[1.0])]
    out = review_payload(rallies, {0: []}, subject_doc({0: 0.5}), None, 30.0,
                         Corrections(deleted={0}))
    assert out["rallies"][0]["deleted"] is True


def test_review_payload_survives_a_session_with_no_subject_doc():
    rallies = [Rally(0, 0.0, 8.0, contacts=[1.0])]
    out = review_payload(rallies, {}, {}, None, 30.0, Corrections())
    assert out["rallies"][0]["subject"]["ids"] == []
    assert out["rallies"][0]["priority"] > 0


def test_corrections_file_is_valid_json_for_the_accuracy_script(tmp_path):
    Corrections(subject={1: [3]}, actions={1: {0: "serve"}}).save(tmp_path)
    raw = json.loads((tmp_path / "corrections.json").read_text())
    assert raw["subject"] == {"1": [3]}
    assert raw["actions"] == {"1": {"0": "serve"}}
