import numpy as np

from audio import (Whistle, _merge_whistles, _runs, detect_contacts,
                   detect_whistles)


def test_runs_finds_contiguous_true_blocks():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert _runs(mask) == [(1, 2), (5, 5), (7, 9)]


def test_runs_on_empty_mask():
    assert _runs(np.zeros(10, dtype=bool)) == []


def test_merge_whistles_bridges_a_single_frame_dropout():
    ws = [Whistle(1.00, 1.20, 3000.0), Whistle(1.24, 1.45, 3000.0)]
    merged = _merge_whistles(ws)
    assert len(merged) == 1
    assert merged[0].start == 1.00 and merged[0].end == 1.45


def test_merge_whistles_drops_squeaks_and_sirens():
    ws = [Whistle(1.0, 1.05, 3000.0),   # too short — shoe squeak
          Whistle(5.0, 5.4, 3000.0),    # a real blast
          Whistle(9.0, 12.0, 3000.0)]   # too long — not a whistle
    merged = _merge_whistles(ws)
    assert [round(w.duration, 2) for w in merged] == [0.4]


def test_detect_whistles_finds_the_synthetic_blasts(synth_audio):
    path, want, _ = synth_audio
    got = detect_whistles(path)
    assert len(got) == len(want)
    for w, expected in zip(got, want):
        assert abs(w.start - expected) < 0.15
        assert 2500 < w.peak_hz < 3500


def test_detect_contacts_finds_the_pops(synth_audio):
    path, _, want = synth_audio
    got = detect_contacts(path)
    times = [c.t for c in got]
    for expected in want:
        assert any(abs(t - expected) < 0.10 for t in times), f"missed {expected}"


def test_detect_contacts_ignores_the_whistles(synth_audio):
    """A whistle onset is loud and broadband at its attack — without the guard
    it would be scored as a ball contact and fake an extra touch per rally."""
    path, whistle_times, _ = synth_audio
    whistles = detect_whistles(path)
    got = detect_contacts(path, whistles)
    for w in whistle_times:
        assert not any(abs(c.t - w) < 0.12 for c in got)


def test_contact_strength_is_normalised(synth_audio):
    path, _, _ = synth_audio
    got = detect_contacts(path)
    assert got
    strengths = [c.strength for c in got]
    assert max(strengths) == 1.0
    assert all(0.0 < s <= 1.0 for s in strengths)


def test_detectors_are_safe_on_a_silent_clip(tmp_path):
    import soundfile as sf
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(22050 * 3), 22050)
    assert detect_whistles(path) == []
    assert detect_contacts(path) == []
