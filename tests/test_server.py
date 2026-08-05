import json
import subprocess

import pytest
from fastapi.testclient import TestClient

import pipeline
import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / "sessions"
    data.mkdir()
    monkeypatch.setattr(pipeline, "DATA", data)
    monkeypatch.setattr(server, "DATA", data)
    # keep the background worker out of unit tests; jobs are exercised directly
    monkeypatch.setattr(pipeline, "enqueue", lambda sdir, job: None)
    return TestClient(server.app)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], capture_output=True, check=True)
    return path


def upload(client, clip, **form):
    with clip.open("rb") as fh:
        return client.post("/api/upload", files={"file": ("clip.mp4", fh, "video/mp4")},
                           data={"label": "test match", **form})


def test_index_serves_the_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SpikeIQ" in r.text


def test_mount_prefix_is_injected_for_a_subpath_mount(client):
    """Without this the page loads behind a reverse proxy but every API call
    404s, which looks like the app silently doing nothing."""
    r = client.get("/", headers={"x-forwarded-prefix": "/app/spikeiq"})
    assert "window.__BASE__=" in r.text
    assert "/app/spikeiq" in r.text


def test_no_prefix_header_leaves_base_unset(client):
    assert "window.__BASE__=" not in client.get("/").text


def test_upload_creates_a_session_and_stores_height(client, clip):
    r = upload(client, clip, height_m="1.93")
    assert r.status_code == 200
    sid = r.json()["session"]
    meta = json.loads((pipeline.DATA / sid / "meta.json").read_text())
    assert meta["label"] == "test match"
    assert meta["height_m"] == 1.93


def test_sessions_lists_the_upload(client, clip):
    sid = upload(client, clip).json()["session"]
    rows = client.get("/api/sessions").json()["sessions"]
    assert any(s["session"] == sid for s in rows)


def test_session_detail_reports_uncalibrated_state(client, clip):
    sid = upload(client, clip).json()["session"]
    body = client.get(f"/api/session/{sid}").json()
    assert body["session"] == sid
    assert body["calibrated"] is False


def test_unknown_session_is_a_404(client):
    assert client.get("/api/session/deadbeef").status_code == 404


def test_calibrate_stores_the_homography_and_the_click(client, clip):
    sid = upload(client, clip).json()["session"]
    r = client.post(f"/api/session/{sid}/calibrate", json={
        "corners": [[520, 300], [1180, 300], [1600, 940], [180, 940]],
        "attack": [[540, 420], [1150, 420], [1450, 760], [300, 760]],
        "subject_click": [800, 700],
        "height_m": 1.85,
    })
    assert r.status_code == 200
    sdir = pipeline.DATA / sid
    assert (sdir / "calibration.json").exists()
    meta = json.loads((sdir / "meta.json").read_text())
    assert meta["subject_click"] == [800, 700]
    assert meta["height_m"] == 1.85


def test_calibrate_accepts_landmarks(client, clip, calib):
    """The new shape: any four or more named points."""
    from conftest import px_for
    from court import LANDMARKS

    sid = upload(client, clip).json()["session"]
    marks = {n: [float(v) for v in px_for(calib, *LANDMARKS[n])]
             for n in ["corner_near_left", "corner_near_right",
                       "attack_near_left", "attack_near_right",
                       "centre_left", "centre_right"]}
    r = client.post(f"/api/session/{sid}/calibrate", json={
        "landmarks": marks, "subject_click": [800, 700]})
    assert r.status_code == 200
    saved = json.loads(
        (pipeline.DATA / sid / "calibration.json").read_text())
    assert set(saved["landmarks"]) == set(marks)
    assert saved["quality"]["n_landmarks"] == 6


def test_calibrate_still_accepts_the_old_corner_shape(client, clip):
    """Older clients and stored calibrations must keep working."""
    sid = upload(client, clip).json()["session"]
    r = client.post(f"/api/session/{sid}/calibrate", json={
        "corners": [[520, 300], [1180, 300], [1600, 940], [180, 940]],
        "subject_click": [800, 700]})
    assert r.status_code == 200


def test_calibrate_refuses_too_few_landmarks(client, clip, calib):
    from conftest import px_for
    from court import LANDMARKS
    sid = upload(client, clip).json()["session"]
    marks = {n: [float(v) for v in px_for(calib, *LANDMARKS[n])]
             for n in ["corner_near_left", "corner_near_right"]}
    r = client.post(f"/api/session/{sid}/calibrate", json={
        "landmarks": marks, "subject_click": [800, 700]})
    assert r.status_code == 400
    assert "at least" in r.json()["detail"]


def test_calibrate_refuses_a_request_with_no_court_at_all(client, clip):
    sid = upload(client, clip).json()["session"]
    r = client.post(f"/api/session/{sid}/calibrate",
                    json={"subject_click": [800, 700]})
    assert r.status_code == 400


def test_landmark_menu_is_served_for_the_picker(client):
    body = client.get("/api/court/landmarks").json()
    assert body["minimum"] == 4
    assert len(body["landmarks"]) == 10
    one = body["landmarks"][0]
    assert set(one) == {"name", "court", "label"}
    assert all(l["label"] for l in body["landmarks"])


def test_detect_court_returns_nothing_without_a_frame(client, clip):
    """Auto-detection failing is an ordinary outcome, not a 500."""
    sid = upload(client, clip).json()["session"]
    r = client.post(f"/api/session/{sid}/detect-court", json={})
    assert r.status_code == 200
    assert r.json()["proposal"] is None


def test_session_payload_carries_the_proposal_and_camera(client, clip):
    sid = upload(client, clip).json()["session"]
    body = client.get(f"/api/session/{sid}").json()
    assert "court_proposal" in body
    assert "camera" in body


def test_calibrate_rejects_degenerate_points(client, clip):
    sid = upload(client, clip).json()["session"]
    r = client.post(f"/api/session/{sid}/calibrate", json={
        "corners": [[0, 0], [1, 1]], "subject_click": [10, 10]})
    assert r.status_code == 400


def test_recalibrating_drops_derived_data_but_keeps_tracking(client, clip):
    """Tracking is the hour-long stage and is pure pixel space, so a second
    calibration must not throw it away."""
    sid = upload(client, clip).json()["session"]
    sdir = pipeline.DATA / sid
    (sdir / "tracks.parquet").write_bytes(b"x")
    (sdir / "metrics.json").write_text("{}")
    client.post(f"/api/session/{sid}/calibrate", json={
        "corners": [[520, 300], [1180, 300], [1600, 940], [180, 940]],
        "subject_click": [800, 700]})
    assert (sdir / "tracks.parquet").exists()
    assert not (sdir / "metrics.json").exists()


def test_frame_is_404_until_prepare_has_run(client, clip):
    sid = upload(client, clip).json()["session"]
    assert client.get(f"/api/session/{sid}/frame").status_code == 404


def test_progress_lists_only_active_sessions(client, clip):
    sid = upload(client, clip).json()["session"]
    assert client.get("/api/progress").json()["running"] == []
    pipeline.set_status(pipeline.DATA / sid, "tracking", "running", progress=0.3)
    running = client.get("/api/progress").json()["running"]
    assert running and running[0]["session"] == sid


def test_delete_removes_the_session(client, clip):
    sid = upload(client, clip).json()["session"]
    r = client.post("/api/sessions/delete", json={"sessions": [sid]})
    assert r.json()["deleted"] == [sid]
    assert not (pipeline.DATA / sid).exists()


def test_delete_cannot_escape_the_sessions_directory(client):
    """A crafted id must not be able to delete anything outside data/."""
    victim = pipeline.DATA.parent / "keepme"
    victim.mkdir()
    r = client.post("/api/sessions/delete", json={"sessions": ["../keepme"]})
    assert r.json()["deleted"] == []
    assert victim.exists()


def test_clip_404s_for_a_rally_that_does_not_exist(client, clip):
    sid = upload(client, clip).json()["session"]
    assert client.get(f"/api/session/{sid}/clip/42").status_code == 404


def test_review_lists_rallies_worst_first(client, clip):
    sid = upload(client, clip).json()["session"]
    sdir = pipeline.DATA / sid
    pipeline.write_json(sdir, "ingest.json", {"fps": 30.0})
    pipeline.write_json(sdir, "rallies.json", [
        {"index": 0, "start": 0.0, "end": 8.0, "contacts": [1.0],
         "winner": "near", "set_index": 0},
        {"index": 1, "start": 30.0, "end": 38.0, "contacts": [31.0],
         "winner": None, "set_index": 0},
    ])
    pipeline.write_json(sdir, "subject_by_rally.json", {
        "ids_by_rally": {"0": [1], "1": []},
        "confidence_by_rally": {"0": 0.9, "1": 0.0},
        "method_by_rally": {"0": "bridged", "1": "none"}})
    body = client.get(f"/api/session/{sid}/review").json()
    assert [r["index"] for r in body["rallies"]] == [1, 0]
    assert body["rallies"][0]["why"]


def test_review_submission_is_persisted_and_requeues_analysis(client, clip):
    sid = upload(client, clip).json()["session"]
    sdir = pipeline.DATA / sid
    (sdir / "calibration.json").write_text(
        json.dumps({"corners_px": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "attack_px": None}))
    r = client.post(f"/api/session/{sid}/review", json={
        "subject": {"3": [42]},
        "actions": {"3": {"1": "block"}},
        "deleted": {"5": True}})
    assert r.status_code == 200
    saved = json.loads((sdir / "corrections.json").read_text())
    assert saved["subject"] == {"3": [42]}
    assert saved["actions"] == {"3": {"1": "block"}}
    assert saved["deleted"] == [5]


def test_review_does_not_discard_the_expensive_tracking(client, clip):
    """A correction must cost seconds. Clearing derived data the way
    calibration does would throw away the hour-long stage."""
    sid = upload(client, clip).json()["session"]
    sdir = pipeline.DATA / sid
    (sdir / "tracks.parquet").write_bytes(b"x")
    client.post(f"/api/session/{sid}/review", json={"subject": {"1": [2]}})
    assert (sdir / "tracks.parquet").exists()


def test_review_corrections_accumulate_across_submissions(client, clip):
    sid = upload(client, clip).json()["session"]
    sdir = pipeline.DATA / sid
    client.post(f"/api/session/{sid}/review", json={"subject": {"1": [2]}})
    client.post(f"/api/session/{sid}/review", json={"subject": {"4": [9]}})
    saved = json.loads((sdir / "corrections.json").read_text())
    assert saved["subject"] == {"1": [2], "4": [9]}


def test_thumb_404s_for_an_unknown_rally(client, clip):
    sid = upload(client, clip).json()["session"]
    pipeline.write_json(pipeline.DATA / sid, "rallies.json", [])
    assert client.get(f"/api/session/{sid}/thumb/3").status_code == 404


def test_trend_skips_sessions_with_no_rating(client, clip):
    a = upload(client, clip).json()["session"]
    b = upload(client, clip).json()["session"]
    pipeline.write_json(pipeline.DATA / b, "rating.json", {
        "level": 3.1, "band": "A", "confidence": 0.6,
        "dimensions": {"attacking": {"level": 3.4}}})
    points = client.get("/api/trend").json()["points"]
    assert [p["session"] for p in points] == [b]
    assert points[0]["dimensions"]["attacking"] == 3.4
    assert a not in [p["session"] for p in points]
