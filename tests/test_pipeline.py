import json
import subprocess

import pytest

import pipeline
from pipeline import (clear_derived, get_status, load_rallies, probe_video,
                      read_json, set_status, write_json)


@pytest.fixture
def sdir(tmp_path, monkeypatch):
    d = tmp_path / "sessions" / "abc123"
    d.mkdir(parents=True)
    monkeypatch.setattr(pipeline, "DATA", tmp_path / "sessions")
    return d


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    """A short synthetic clip with a real audio track, for the ffmpeg stages."""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], capture_output=True, check=True)
    return path


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("silent") / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
    ], capture_output=True, check=True)
    return path


def test_status_round_trips(sdir):
    set_status(sdir, "tracking", "running", progress=0.5)
    got = get_status(sdir)
    assert got["stage"] == "tracking" and got["state"] == "running"
    assert got["progress"] == 0.5


def test_status_clears_a_stale_error_on_the_next_success(sdir):
    """A retried stage that succeeds must not keep showing the old failure."""
    set_status(sdir, "tracking", "error", error="boom")
    assert get_status(sdir)["error"] == "boom"
    set_status(sdir, "tracking", "running")
    assert "error" not in get_status(sdir)


def test_status_survives_a_truncated_file(sdir):
    (sdir / "status.json").write_text("{not json")
    assert get_status(sdir)["state"] == "pending"


def test_missing_status_reads_as_new(sdir):
    assert get_status(sdir)["stage"] == "new"


def test_clear_derived_keeps_the_expensive_tracking(sdir):
    """Re-calibrating must cost seconds, not another full tracking pass."""
    (sdir / "tracks.parquet").write_bytes(b"x")
    (sdir / "metrics.json").write_text("{}")
    (sdir / "subject.parquet").write_bytes(b"x")
    clear_derived(sdir)
    assert (sdir / "tracks.parquet").exists()
    assert not (sdir / "metrics.json").exists()
    assert not (sdir / "subject.parquet").exists()


def test_probe_video_reads_stream_properties(sample_video):
    info = probe_video(sample_video)
    assert info["width"] == 640 and info["height"] == 360
    assert info["fps"] == pytest.approx(30.0, abs=0.1)
    assert info["has_audio"] is True
    assert info["codec"] == "h264"


def test_probe_video_notices_a_missing_audio_track(silent_video):
    assert probe_video(silent_video)["has_audio"] is False


def test_ingest_refuses_silent_footage(sdir, silent_video):
    """Rallies are found from sound. Failing loudly here beats producing an
    empty analysis that looks like the player simply never touched the ball."""
    with pytest.raises(RuntimeError, match="no audio track"):
        pipeline.ingest(sdir, silent_video)


def test_ingest_remuxes_rather_than_transcodes_when_it_can(sdir, sample_video):
    info = pipeline.ingest(sdir, sample_video)
    assert info["remuxed"] is True
    assert (sdir / "video.mp4").exists()
    assert (sdir / "audio.wav").exists()
    assert read_json(sdir, "ingest.json")["duration"] > 2.0


def test_prepare_runs_end_to_end_on_a_real_file(sdir, sample_video):
    """The whole audio-first phase: ingest, detect, segment, pick a frame.

    A pure sine has no whistles or pops in it, so the honest outcome is zero
    rallies — and the calibration frame must still be produced so the user can
    calibrate and see for themselves."""
    (sdir / "raw.mp4").write_bytes(sample_video.read_bytes())
    out = pipeline.prepare(sdir)
    assert get_status(sdir)["stage"] == "prepared"
    assert (sdir / "frame0.jpg").exists()
    assert out["rallies"]["rally_count"] == 0
    assert json.loads((sdir / "whistles.json").read_text()) == []


def test_load_rallies_round_trips_through_json(sdir):
    write_json(sdir, "rallies.json", [
        {"index": 0, "start": 1.0, "end": 6.0, "set_index": 0,
         "contacts": [1.5, 2.5], "serving_side": "near", "winner": "far",
         "source": "whistle"},
    ])
    rl = load_rallies(sdir)
    assert len(rl) == 1
    assert rl[0].serving_side == "near" and rl[0].winner == "far"
    assert rl[0].serve_t == 1.5


def test_load_rallies_on_a_missing_file(sdir):
    assert load_rallies(sdir) == []


def test_analyze_refuses_without_a_subject_click(sdir, sample_video, calib):
    from court import CourtCalibration
    (sdir / "raw.mp4").write_bytes(sample_video.read_bytes())
    pipeline.prepare(sdir)
    CourtCalibration([[0, 0], [100, 0], [100, 100], [0, 100]]).save(
        sdir / "calibration.json")
    write_json(sdir, "meta.json", {})
    with pytest.raises(RuntimeError, match="no subject selected|no players"):
        pipeline.analyze(sdir)
