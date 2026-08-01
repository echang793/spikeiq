"""The label-export and accuracy harness — the path from review clicks to a
measured number. This is what will run the project's 80%-on-50-labels gate the
moment there is volleyball footage, so it needs to work before there is.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from accuracy import confusion, per_action, report  # noqa: E402
from export_labels import session_labels  # noqa: E402


def make_session(tmp_path, plays, corrections) -> Path:
    sdir = tmp_path / "sess1"
    sdir.mkdir()
    (sdir / "plays.json").write_text(json.dumps(plays))
    (sdir / "meta.json").write_text(json.dumps({"label": "friday league"}))
    (sdir / "corrections.json").write_text(json.dumps(corrections))
    return sdir


def play(action, predicted=None, t=1.0):
    p = {"t": t, "action": action, "side": "near", "zone": 4, "x": 4.5,
         "y": 10.0, "airborne": 0.1, "touch_index": 1, "confidence": 0.7}
    if predicted:
        p["predicted"] = predicted
        p["corrected"] = True
        p["confidence"] = 1.0
    return p


def test_unreviewed_rallies_are_not_exported(tmp_path):
    """An uncorrected contact nobody looked at is not a correct prediction, and
    counting it as one would inflate every accuracy number computed here."""
    sdir = make_session(tmp_path, {"0": [play("attack")]},
                        {"subject": {}, "actions": {}, "deleted": [],
                         "confirmed": []})
    assert session_labels(sdir) == []


def test_a_confirmed_rally_exports_its_contacts_as_correct(tmp_path):
    sdir = make_session(tmp_path, {"0": [play("serve"), play("pass")]},
                        {"confirmed": [0]})
    rows = session_labels(sdir)
    assert len(rows) == 2
    assert all(r["correct"] for r in rows)
    assert [r["truth"] for r in rows] == ["serve", "pass"]


def test_a_corrected_contact_records_both_answers(tmp_path):
    """Once the label is overwritten there is no other record of what the model
    actually said, so `predicted` has to be preserved at correction time."""
    sdir = make_session(
        tmp_path,
        {"0": [play("serve"), play("set", predicted="attack")]},
        {"confirmed": [0], "actions": {"0": {"1": "set"}}})
    rows = session_labels(sdir)
    wrong = [r for r in rows if not r["correct"]]
    assert len(wrong) == 1
    assert wrong[0]["predicted"] == "attack"
    assert wrong[0]["truth"] == "set"


def test_a_rally_with_only_corrections_still_counts_as_reviewed(tmp_path):
    sdir = make_session(tmp_path, {"0": [play("dig", predicted="pass")]},
                        {"actions": {"0": {"0": "dig"}}})
    assert len(session_labels(sdir)) == 1


def test_deleted_rallies_are_excluded(tmp_path):
    sdir = make_session(tmp_path, {"0": [play("attack")]},
                        {"confirmed": [0], "deleted": [0]})
    assert session_labels(sdir) == []


def test_export_carries_features_for_tuning(tmp_path):
    sdir = make_session(tmp_path, {"0": [play("attack")]}, {"confirmed": [0]})
    row = session_labels(sdir)[0]
    assert row["features"]["zone"] == 4
    assert row["label"] == "friday league"
    assert row["session"] == "sess1"


def test_session_with_no_plays_file(tmp_path):
    sdir = tmp_path / "empty"
    sdir.mkdir()
    assert session_labels(sdir) == []


def rows_for(pairs):
    return [{"session": "s", "truth": t, "predicted": p, "correct": t == p}
            for t, p in pairs]


def test_confusion_matrix_counts_truth_against_prediction():
    m = confusion(rows_for([("pass", "dig"), ("pass", "pass"), ("dig", "dig")]))
    assert m["pass"]["dig"] == 1
    assert m["pass"]["pass"] == 1
    assert m["dig"]["dig"] == 1


def test_per_action_separates_recall_from_precision():
    """The distinction matters: a decoder that calls everything a pass has
    perfect recall on passes and terrible precision."""
    stats = per_action(rows_for([("pass", "pass"), ("dig", "pass"),
                                 ("attack", "pass")]))
    assert stats["pass"]["recall"] == 1.0
    assert stats["pass"]["precision"] == pytest.approx(1 / 3)
    assert stats["dig"]["recall"] == 0.0


def test_per_action_reports_none_rather_than_zero_without_support():
    stats = per_action(rows_for([("pass", "pass")]))
    assert stats["block"]["support"] == 0
    assert stats["block"]["recall"] is None


def test_gate_fails_on_too_few_labels(capsys):
    assert report(rows_for([("pass", "pass")] * 10)) is False
    assert "GATE NOT MET" in capsys.readouterr().out


def test_gate_fails_on_poor_accuracy(capsys):
    rows = rows_for([("pass", "pass")] * 30 + [("dig", "pass")] * 30)
    assert report(rows) is False
    out = capsys.readouterr().out
    assert "GATE FAILED" in out
    assert "grammar.TRANSITIONS" in out


def test_gate_passes_on_a_good_sample(capsys):
    rows = rows_for([("pass", "pass")] * 45 + [("dig", "pass")] * 5
                    + [("attack", "attack")] * 10)
    assert report(rows) is True
    assert "GATE PASSED" in capsys.readouterr().out


def test_report_handles_no_labels(capsys):
    assert report([]) is False
    assert "No labels" in capsys.readouterr().out


def test_scripts_run_end_to_end(tmp_path):
    """The two scripts have to actually work from the command line, since that
    is the only way they will ever be used."""
    data = tmp_path / "sessions"
    data.mkdir()
    sdir = data / "s1"
    sdir.mkdir()
    (sdir / "plays.json").write_text(json.dumps(
        {"0": [play("serve"), play("set", predicted="attack")]}))
    (sdir / "meta.json").write_text(json.dumps({"label": "x"}))
    (sdir / "corrections.json").write_text(json.dumps(
        {"confirmed": [0], "actions": {"0": {"1": "set"}}}))

    labels = tmp_path / "labels.jsonl"
    subprocess.run(
        [sys.executable, str(SCRIPTS / "export_labels.py"),
         "--data", str(data), "--out", str(labels)],
        check=True, capture_output=True)
    assert labels.exists()
    assert len(labels.read_text().strip().splitlines()) == 2

    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "accuracy.py"), str(labels)],
        capture_output=True, text=True)
    assert "overall accuracy: 50.0%" in done.stdout
    assert done.returncode == 1        # correctly refuses on a 2-label sample
