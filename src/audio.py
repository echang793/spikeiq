"""Audio event detection: referee whistles and ball contacts.

The two sounds are physically different and want different detectors, which is
what makes a volleyball match easier to segment than a pickleball one:

- A whistle is a SUSTAINED, NARROWBAND tone (roughly 1.8-4.5 kHz, 0.15-1.5 s).
  Energy sits in a couple of FFT bins and stays there.
- A ball contact is a TRANSIENT, BROADBAND pop. Energy is spread and gone in
  tens of milliseconds.

So whistles are found by tonality-over-time and contacts by attack sharpness
(the latter ported from dinkiq/src/events.py `detect_hits`). Detecting whistles
separately is what gives us a rally clock for free: the referee brackets every
rally for us, which beats inferring rally edges from silence gaps.

Thresholds here are adaptive (percentiles of this recording), not absolute —
gym reverb, mic gain and crowd noise vary far too much between venues for fixed
levels to survive.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

SR = 22050
HOP = 512

# --- whistle ---------------------------------------------------------------
WHISTLE_LO_HZ = 1800
WHISTLE_HI_HZ = 4500
MIN_WHISTLE_S = 0.12       # shorter than this is a squeak, not a whistle
MAX_WHISTLE_S = 2.0
WHISTLE_MERGE_S = 0.08     # bridge a one-frame dropout inside a single blast
TONALITY_MIN = 0.30        # peak-bin share of in-band energy for a pure tone
WHISTLE_ENERGY_PCTL = 97.0  # in-band energy percentile a blast must exceed

# --- contact ---------------------------------------------------------------
CONTACT_LO_HZ = 800
CONTACT_HI_HZ = 8000
MIN_CONTACT_GAP_S = 0.18   # two real touches cannot be closer than this
ATTACK_RATIO = 2.2         # post/pre RMS ratio a true transient must exceed
WHISTLE_GUARD_S = 0.15     # ignore "contacts" this close to a whistle edge


@dataclass
class Whistle:
    start: float
    end: float
    peak_hz: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Contact:
    t: float
    strength: float   # 0-1, relative to the loudest contact in this recording

    def as_dict(self) -> dict:
        return {"t": round(self.t, 3), "strength": round(self.strength, 3)}


def load_audio(path: Path, sr: int = SR) -> tuple[np.ndarray, int]:
    import librosa  # heavy import, and audio may be absent entirely
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y, int(sr_out)


def detect_whistles(path: Path, sr: int = SR) -> list[Whistle]:
    """Referee whistle blasts, as (start, end, peak frequency) in seconds/Hz."""
    import librosa

    y, sr = load_audio(path, sr)
    if len(y) < sr // 2:
        return []
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= WHISTLE_LO_HZ) & (freqs <= WHISTLE_HI_HZ)
    Sb = S[band]
    if Sb.size == 0:
        return []
    band_energy = Sb.sum(axis=0)
    peak_bin = Sb.argmax(axis=0)
    peak_energy = Sb.max(axis=0)
    # tonality: how much of the in-band energy the single loudest bin holds
    tonality = peak_energy / np.maximum(band_energy, 1e-9)
    loud = band_energy > np.percentile(band_energy, WHISTLE_ENERGY_PCTL)
    is_whistle = loud & (tonality >= TONALITY_MIN)

    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)
    band_freqs = freqs[band]
    out: list[Whistle] = []
    for start_i, end_i in _runs(is_whistle):
        t0, t1 = float(times[start_i]), float(times[min(end_i, len(times) - 1)])
        out.append(Whistle(t0, t1, float(np.median(band_freqs[peak_bin[start_i:end_i + 1]]))))
    return _merge_whistles(out)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of each True run in a boolean mask."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_whistles(ws: list[Whistle]) -> list[Whistle]:
    """Join blasts separated by a sub-frame dropout, then drop the ones whose
    duration does not look like a whistle at all."""
    merged: list[Whistle] = []
    for w in ws:
        if merged and w.start - merged[-1].end <= WHISTLE_MERGE_S:
            prev = merged[-1]
            merged[-1] = Whistle(prev.start, w.end, (prev.peak_hz + w.peak_hz) / 2)
        else:
            merged.append(w)
    return [w for w in merged if MIN_WHISTLE_S <= w.duration <= MAX_WHISTLE_S]


def detect_contacts(path: Path, whistles: list[Whistle] | None = None,
                    sr: int = SR) -> list[Contact]:
    """Ball-contact times with a relative loudness for each.

    Loudness matters downstream: a serve or spike is a sharp, loud crack while a
    set or an overhead pass is soft, so `strength` is one of the features that
    separates them in `contacts.py`.
    """
    import librosa
    from scipy.signal import butter, sosfiltfilt

    y, sr = load_audio(path, sr)
    if len(y) < sr:
        return []
    sos = butter(4, [CONTACT_LO_HZ, CONTACT_HI_HZ], btype="band", fs=sr, output="sos")
    yf = sosfiltfilt(sos, y)
    onset_env = librosa.onset.onset_strength(y=yf, sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, units="time",
        backtrack=False, delta=0.30, wait=int(MIN_CONTACT_GAP_S * sr / HOP),
    )

    guard = _guard_intervals(whistles or [])
    win = int(0.010 * sr)
    kept: list[tuple[float, float]] = []
    for t in np.asarray(onsets, dtype=float):
        if any(lo <= t <= hi for lo, hi in guard):
            continue
        i = int(t * sr)
        # onset timestamps lag the physical pop by up to a hop (~23 ms), so look
        # for the energy peak in a window straddling t
        lo_i, hi_i = max(0, i - int(0.06 * sr)), min(len(yf), i + int(0.03 * sr))
        peaks = [np.sqrt(np.mean(yf[j:j + win] ** 2) + 1e-12)
                 for j in range(lo_i, hi_i - win, max(1, win // 2))]
        peak = max(peaks) if peaks else 0.0
        base_seg = yf[max(0, i - int(0.15 * sr)):max(1, i - int(0.05 * sr))]
        base = np.sqrt(np.mean(base_seg ** 2) + 1e-12)
        if peak / base >= ATTACK_RATIO:
            kept.append((float(t), float(peak)))

    if not kept:
        return []
    loudest = max(p for _, p in kept)
    return [Contact(t, p / loudest) for t, p in kept]


def _guard_intervals(whistles: list[Whistle]) -> list[tuple[float, float]]:
    """Time ranges around whistles where an "onset" is really the whistle."""
    return [(w.start - WHISTLE_GUARD_S, w.end + WHISTLE_GUARD_S) for w in whistles]


def whistles_to_json(ws: list[Whistle]) -> list[dict]:
    return [asdict(w) | {"duration": round(w.duration, 3)} for w in ws]


def contacts_to_json(cs: list[Contact]) -> list[dict]:
    return [c.as_dict() for c in cs]
