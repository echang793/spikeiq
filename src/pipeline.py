"""Session storage and the two-phase analysis pipeline.

Split in two on purpose, which is what the audio-first design buys us:

    prepare()  ingest -> audio -> rallies -> calibration frame
    analyze()  tracking -> contacts -> grammar -> metrics -> rating -> feedback

`prepare` needs no input from the user and runs on audio alone in seconds, and
its output is what makes the calibration step good: instead of asking you to
click court corners on frame zero — which is often a wall, a warmup, or an empty
gym — it hands you the first frame of the first rally, where the court is framed
and twelve players are spread out and standing still.

Every stage is resumable and writes one artifact. Pixel-space artifacts
(`tracks.parquet`) survive re-calibration; everything court-derived is
disposable and gets cleared when the calibration changes.
"""

import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sessions"
MODELS = ROOT / "models"

PREPARE_STAGES = ["ingest", "audio", "rallies"]
ANALYZE_STAGES = ["tracking", "subject", "rotation", "contacts", "grammar",
                  "jump", "metrics", "rating", "feedback"]

# artifacts that depend on the court calibration and must be rebuilt when it
# changes; tracks.parquet is deliberately NOT here — it is pure pixel space
DERIVED = ["subject.parquet", "contacts.parquet", "rotation.json", "plays.json",
           "jumps.json", "metrics.json", "rating.json", "feedback.json"]


def session_dir(sid: str) -> Path:
    d = DATA / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_session() -> tuple[str, Path]:
    sid = uuid.uuid4().hex[:12]
    return sid, session_dir(sid)


def set_status(sdir: Path, stage: str, state: str, error: str | None = None,
               progress: float | None = None) -> None:
    status = get_status(sdir)
    status.update({"stage": stage, "state": state, "updated": time.time()})
    if error is not None:
        status["error"] = error
    elif state != "error":
        status.pop("error", None)
    if progress is not None:
        status["progress"] = round(progress, 3)
    (sdir / "status.json").write_text(json.dumps(status))


def get_status(sdir: Path) -> dict:
    path = sdir / "status.json"
    if not path.exists():
        return {"stage": "new", "state": "pending"}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"stage": "unknown", "state": "pending"}


def read_json(sdir: Path, name: str, default=None):
    path = sdir / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def write_json(sdir: Path, name: str, payload) -> None:
    (sdir / name).write_text(json.dumps(payload, indent=1))


def clear_derived(sdir: Path) -> None:
    """Drop everything that depends on the calibration, keeping the expensive
    pixel-space tracking so a re-calibration costs seconds, not an hour."""
    for name in DERIVED:
        (sdir / name).unlink(missing_ok=True)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-4:]
        raise RuntimeError(f"{cmd[0]} failed: {' '.join(tail)}")


def probe_video(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", str(path)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("ffprobe failed — is the file a video?")
    info = json.loads(out.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise RuntimeError("no video stream found")
    num, den = (video.get("avg_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return {
        "fps": round(fps, 4),
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "duration": float(info["format"].get("duration", 0.0)),
        "codec": video.get("codec_name", ""),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def ingest(sdir: Path, raw: Path) -> dict:
    """Normalise the upload and split out the audio track.

    A file that is already 1080p-or-smaller H.264 is remuxed rather than
    re-encoded — a full transcode of an hour of 60 fps footage costs more than
    the entire rest of the pipeline.
    """
    info = probe_video(raw)
    video = sdir / "video.mp4"
    fast_path = (info["codec"] == "h264" and info["height"] <= 1080)
    if fast_path:
        _run(["ffmpeg", "-y", "-i", str(raw), "-c", "copy",
              "-movflags", "+faststart", str(video)])
    else:
        _run(["ffmpeg", "-y", "-i", str(raw), "-vf", "scale=-2:1080",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
              "-c:a", "aac", str(video)])
    if not info["has_audio"]:
        raise RuntimeError(
            "this clip has no audio track — SpikeIQ finds rallies from the "
            "referee's whistle and the sound of the ball, so it cannot analyse "
            "silent footage")
    _run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "22050",
          str(sdir / "audio.wav")])
    info["remuxed"] = fast_path
    write_json(sdir, "ingest.json", info)
    return info


def extract_frame(sdir: Path, t: float, name: str = "frame0.jpg") -> Path:
    out = sdir / name
    _run(["ffmpeg", "-y", "-ss", f"{max(t, 0):.3f}", "-i", str(sdir / "video.mp4"),
          "-frames:v", "1", "-q:v", "3", str(out)])
    return out


def prepare(sdir: Path) -> dict:
    """Ingest, find the rallies from audio, and pick a calibration frame."""
    import audio as audio_mod
    import rallies as rallies_mod

    raw = next(sdir.glob("raw.*"))
    set_status(sdir, "ingest", "running")
    info = ingest(sdir, raw)

    set_status(sdir, "audio", "running")
    whistles = audio_mod.detect_whistles(sdir / "audio.wav")
    contacts = audio_mod.detect_contacts(sdir / "audio.wav", whistles)
    write_json(sdir, "whistles.json", audio_mod.whistles_to_json(whistles))
    write_json(sdir, "contacts_audio.json", audio_mod.contacts_to_json(contacts))

    set_status(sdir, "rallies", "running")
    rl = rallies_mod.segment_rallies(whistles, contacts, info["duration"])
    write_json(sdir, "rallies.json", [r.as_dict() for r in rl])
    summary = rallies_mod.summarise(rl)
    write_json(sdir, "rallies_summary.json", summary)

    # calibrate on the first rally's opening frame: court framed, players spread
    # and still. Falls back to one second in if no rally was found at all.
    t = rl[0].start if rl else 1.0
    extract_frame(sdir, t)
    write_json(sdir, "calibration_frame.json", {"t": round(t, 3)})

    set_status(sdir, "prepared", "done")
    return {"ingest": info, "rallies": summary}


def load_rallies(sdir: Path) -> list:
    from rallies import Rally
    rows = read_json(sdir, "rallies.json", []) or []
    out = []
    for r in rows:
        out.append(Rally(index=r["index"], start=r["start"], end=r["end"],
                         set_index=r.get("set_index", 0),
                         contacts=r.get("contacts", []),
                         serving_side=r.get("serving_side"),
                         winner=r.get("winner"),
                         source=r.get("source", "whistle")))
    return out


def analyze(sdir: Path) -> dict:
    """Everything downstream of calibration. Requires calibration.json and the
    subject click stored in meta.json."""
    import contacts as contacts_mod
    import grammar as grammar_mod
    import jump as jump_mod
    import metrics as metrics_mod
    import rating as rating_mod
    import feedback as feedback_mod
    import rotation as rotation_mod
    import rallies as rallies_mod
    from court import CourtCalibration
    from tracking import (run_tracking, stitch_chain_ids, stitch_subject,
                          subject_court_positions)

    calib = CourtCalibration.load(sdir / "calibration.json")
    meta = read_json(sdir, "meta.json", {}) or {}
    info = read_json(sdir, "ingest.json", {}) or {}
    fps = float(info.get("fps", 30.0))
    rl = load_rallies(sdir)

    # --- tracking (the expensive stage; cached in pixel space) --------------
    tracks_path = sdir / "tracks.parquet"
    if tracks_path.exists():
        tracks = pd.read_parquet(tracks_path)
    else:
        set_status(sdir, "tracking", "running", progress=0.0)
        windows = [(r.start, r.end) for r in rl]

        def on_progress(_frame, total, done):
            set_status(sdir, "tracking", "running",
                       progress=done / total if total else 0.0)

        tracks = run_tracking(sdir / "video.mp4", tracks_path, MODELS,
                              windows=windows, progress_cb=on_progress)
    if tracks.empty:
        raise RuntimeError("no players were detected — check the calibration "
                           "frame actually shows the court")

    # --- subject -----------------------------------------------------------
    set_status(sdir, "subject", "running")
    subject_id = _resolve_subject(tracks, meta, fps)
    subject_ids = stitch_chain_ids(tracks, subject_id)
    subject = stitch_subject(tracks, subject_id)
    subject.to_parquet(sdir / "subject.parquet", index=False)
    positions = subject_court_positions(tracks, subject_id, calib, fps)

    # --- rally bookkeeping: who served, and therefore who won ---------------
    set_status(sdir, "rotation", "running")
    for r in rl:
        r.serving_side = rallies_mod.detect_serving_side(r, tracks, calib, fps)
    rallies_mod.assign_winners(rl)
    write_json(sdir, "rallies.json", [r.as_dict() for r in rl])

    roles = rotation_mod.rally_roles(rl, subject, calib, fps)
    write_json(sdir, "rotation.json", {
        "per_rally": [r.as_dict() for r in roles],
        "summary": rotation_mod.summarise(roles),
        "coverage": rotation_mod.rotation_coverage(roles),
    })

    # --- contacts and the rally grammar ------------------------------------
    set_status(sdir, "contacts", "running")
    strengths = {round(c["t"], 3): c["strength"]
                 for c in (read_json(sdir, "contacts_audio.json", []) or [])}
    plays_by_rally: dict[int, list[dict]] = {}
    all_features = []
    for i, r in enumerate(rl):
        set_status(sdir, "contacts", "running", progress=(i + 1) / max(len(rl), 1))
        feats = []
        for t in r.contacts:
            f = contacts_mod.contact_features(
                t, strengths.get(round(t, 3), 0.5), tracks, calib, fps)
            if f is not None:
                feats.append(f)
        all_features.extend(feats)
        plays_by_rally[r.index] = grammar_mod.decode_rally(feats)

    set_status(sdir, "grammar", "running")
    if all_features:
        contacts_mod.to_frame(all_features).to_parquet(
            sdir / "contacts.parquet", index=False)
    write_json(sdir, "plays.json", {str(k): v for k, v in plays_by_rally.items()})

    # --- jumps -------------------------------------------------------------
    set_status(sdir, "jump", "running")
    height_m = float(meta.get("height_m") or jump_mod.DEFAULT_HEIGHT_M)
    jumps = jump_mod.detect_jumps(subject, fps, height_m)
    jump_summary = jump_mod.summarise(jumps)
    write_json(sdir, "jumps.json", {"jumps": jump_mod.jumps_to_json(jumps),
                                    "summary": jump_summary,
                                    "height_m_used": height_m,
                                    "height_provided": meta.get("height_m") is not None})

    # --- metrics, rating, feedback ------------------------------------------
    set_status(sdir, "metrics", "running")
    role_groups = rotation_mod.group_by_role(roles)
    m = metrics_mod.compute(rl, plays_by_rally, subject_ids, role_groups,
                            positions, jump_summary)
    write_json(sdir, "metrics.json", m)

    set_status(sdir, "rating", "running")
    r = rating_mod.estimate(m["overall"] | {"coverage": m["coverage"]},
                            jump_summary)
    write_json(sdir, "rating.json", r)

    set_status(sdir, "feedback", "running")
    fb = feedback_mod.build(r, m)
    write_json(sdir, "feedback.json", fb)

    set_status(sdir, "complete", "done", progress=1.0)
    return {"metrics": m, "rating": r, "feedback": fb}


def _resolve_subject(tracks: pd.DataFrame, meta: dict, fps: float) -> int:
    from tracking import pick_subject
    click = meta.get("subject_click")
    if not click:
        raise RuntimeError("no subject selected — click yourself on the "
                           "calibration frame first")
    frame = int(float(meta.get("subject_click_t", 0.0)) * fps)
    return pick_subject(tracks, (float(click[0]), float(click[1])),
                        near_frame=frame)


# --- single background worker ----------------------------------------------
# One at a time on purpose: parallel analyses contend for the same MPS device
# and end up slower than running them in sequence.
_QUEUE: "queue.Queue[tuple[Path, str]]" = queue.Queue()
_WORKER: threading.Thread | None = None
_JOBS = {"prepare": prepare, "analyze": analyze}


def _worker() -> None:
    while True:
        sdir, job = _QUEUE.get()
        try:
            _JOBS[job](sdir)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            set_status(sdir, job, "error", error=str(exc))
        finally:
            _QUEUE.task_done()


def enqueue(sdir: Path, job: str) -> None:
    global _WORKER
    if _WORKER is None or not _WORKER.is_alive():
        _WORKER = threading.Thread(target=_worker, daemon=True)
        _WORKER.start()
    set_status(sdir, job, "queued")
    _QUEUE.put((sdir, job))


def queue_position(sdir: Path) -> int | None:
    pending = [d for d, _ in list(_QUEUE.queue)]
    return pending.index(sdir) if sdir in pending else None
