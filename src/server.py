"""SpikeIQ HTTP server.

Flow: upload -> prepare (audio only, seconds) -> you calibrate the court and
click yourself on the first rally's frame -> analyze -> report.

Mount-prefix handling is carried over from dinkiq and is load-bearing even
though this runs at the root today: `_spa` injects `window.__BASE__` from an
`X-Forwarded-Prefix` header, and the frontend prefixes every API call with it.
Without that, mounting behind a reverse proxy subpath serves the page fine but
404s every API call, which looks like the app silently doing nothing.
"""

import hmac
import json
import os
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

import pipeline
from pipeline import (DATA, get_status, new_session, read_json, write_json)

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
PORT = int(os.environ.get("SPIKEIQ_PORT", "8101"))

app = FastAPI(title="SpikeIQ")


def _load_dotenv(path: Path) -> None:
    """Load .env process-wide so the pipeline thread sees it too."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")


_LOCALHOST = ("127.0.0.1", "::1", "localhost")


def _is_direct_local(client_host: str, headers) -> bool:
    """True only for an unproxied connection from localhost.

    Pulled out of the middleware as a pure function so the bypass decision is
    testable without fighting TestClient's ASGI scope internals (its default
    peer address isn't "127.0.0.1" at all) — and so the one thing that matters
    here, that any forwarding header defeats the bypass regardless of peer
    address, is a single obvious assertion rather than buried in middleware.
    """
    forwarded = bool(headers.get("x-forwarded-for")
                     or headers.get("x-forwarded-prefix"))
    return client_host in _LOCALHOST and not forwarded


@app.middleware("http")
async def _basic_auth(request: Request, call_next):
    """Password-gate everything except a direct localhost connection, when a
    password is set.

    The localhost bypass only applies to a DIRECT connection. Behind the
    reverse proxy `_mount_prefix`/`X-Forwarded-Prefix` is written to support,
    the proxy's own connection to uvicorn is itself on localhost, so every
    remote request would otherwise satisfy the bypass and SPIKEIQ_PASSWORD
    would protect nothing in exactly the deployment it exists for. A request
    carrying any forwarding header is therefore never treated as local,
    whatever its peer address says.
    """
    password = os.environ.get("SPIKEIQ_PASSWORD")
    client = request.client.host if request.client else ""
    if password and not _is_direct_local(client, request.headers):
        import base64
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Basic "):
            try:
                _, _, supplied = base64.b64decode(header[6:]).decode().partition(":")
                ok = hmac.compare_digest(supplied, password)
            except Exception:  # noqa: BLE001 - malformed header is just a failure
                ok = False
        if not ok:
            return JSONResponse({"detail": "auth required"}, status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="spikeiq"'})
    return await call_next(request)


def _mount_prefix(request: Request) -> str:
    return (request.headers.get("x-forwarded-prefix") or "").rstrip("/")


def _spa(path: Path, request: Request) -> HTMLResponse:
    html = path.read_text()
    prefix = _mount_prefix(request)
    if prefix:
        html = html.replace("<!--BASE-->",
                            f"<script>window.__BASE__={json.dumps(prefix)}</script>")
    return HTMLResponse(html)


@app.get("/")
def index(request: Request):
    return _spa(STATIC / "dashboard.html", request)


def _is_within_sessions(d: Path) -> bool:
    """Whether `d` is a directory that genuinely lives inside DATA — the guard
    against a crafted session id ('../../etc') escaping the sessions directory
    onto anywhere else on disk. Every `sid`-based endpoint reads from, and
    several write to, whatever `_sdir` returns, so this has to hold everywhere
    a `sid` is turned into a path, not only on the one endpoint that deletes.
    """
    return d.is_dir() and d.resolve().parent == DATA.resolve()


def _sdir(sid: str) -> Path:
    d = DATA / sid
    if not _is_within_sessions(d):
        raise HTTPException(404, "no such session")
    return d


# --- upload ----------------------------------------------------------------

@app.post("/api/upload")
async def upload(file: UploadFile, label: str = Form(""),
                 height_m: float = Form(0.0)):
    sid, sdir = new_session()
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    raw = sdir / f"raw{suffix}"
    with raw.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    write_json(sdir, "meta.json", {
        "label": label or Path(file.filename or "clip").stem,
        "filename": file.filename,
        "height_m": height_m or None,
        "created": os.path.getmtime(raw),
    })
    pipeline.enqueue(sdir, "prepare")
    return {"session": sid}


# --- calibration -----------------------------------------------------------

def _bad_point(name: str, point) -> str | None:
    """None when `point` is a valid [x, y] pixel pair, else an error string.

    Every point in this API is a pixel pair. A truncated one (from a buggy
    client, say) must fail synchronously here as a clean 400 — matching every
    other malformed-input case this endpoint already raises — rather than as
    an uncaught IndexError: immediately, inside `CourtCalibration.__init__`
    for `landmarks` (a bare 500), or worse, only later for `subject_click`,
    inside the background analyze worker, where it would surface minutes
    afterward as a raw exception string in the progress panel instead of a
    validation error at submission time.
    """
    if not isinstance(point, list) or len(point) != 2:
        return f"{name} must be an [x, y] pixel pair, got {point!r}"
    return None


class Calibration(BaseModel):
    # the landmark form; `corners`/`attack` are the previous fixed-order shape and
    # are still accepted so stored calibrations and older clients keep working
    landmarks: dict[str, list[float]] | None = None
    corners: list[list[float]] | None = None
    attack: list[list[float]] | None = None
    subject_click: list[float]
    subject_click_t: float | None = None
    height_m: float | None = None


@app.post("/api/session/{sid}/calibrate")
def calibrate(sid: str, body: Calibration):
    from court import CourtCalibration

    sdir = _sdir(sid)

    bad = _bad_point("subject_click", body.subject_click)
    if body.landmarks:
        for name, point in body.landmarks.items():
            bad = bad or _bad_point(f"landmarks[{name!r}]", point)
    if body.corners:
        for i, point in enumerate(body.corners):
            bad = bad or _bad_point(f"corners[{i}]", point)
    if body.attack:
        for i, point in enumerate(body.attack):
            bad = bad or _bad_point(f"attack[{i}]", point)
    if bad:
        raise HTTPException(400, bad)

    try:
        if body.landmarks:
            calib = CourtCalibration(body.landmarks)
        elif body.corners:
            calib = CourtCalibration(corners_px=body.corners,
                                     attack_px=body.attack)
        else:
            raise ValueError("no court landmarks were given")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    calib.save(sdir / "calibration.json")

    frame_t = read_json(sdir, "calibration_frame.json", {}) or {}
    meta = read_json(sdir, "meta.json", {}) or {}
    meta["subject_click"] = body.subject_click
    meta["subject_click_t"] = (body.subject_click_t
                               if body.subject_click_t is not None
                               else frame_t.get("t", 0.0))
    if body.height_m:
        meta["height_m"] = body.height_m
    write_json(sdir, "meta.json", meta)

    # the tracking parquet is pixel space and survives; only court-derived
    # artifacts are rebuilt, so re-calibrating costs seconds
    pipeline.clear_derived(sdir)
    pipeline.enqueue(sdir, "analyze")
    return {"ok": True, "queued": True}


# --- reading a session -----------------------------------------------------

class DetectRequest(BaseModel):
    inside_hint: list[float] | None = None


@app.post("/api/session/{sid}/detect-court")
def detect_court(sid: str, body: DetectRequest | None = None):
    """Re-run auto court detection, optionally with a point inside the court.

    One click inside the court is a cheap disambiguator and much stronger than
    anything else available when several rectangles fit the floor about equally
    well — worth offering before falling back to placing landmarks by hand.
    """
    sdir = _sdir(sid)
    hint = tuple(body.inside_hint) if body and body.inside_hint else None
    proposal = pipeline.propose_court(sdir, inside_hint=hint)
    write_json(sdir, "court_proposal.json", proposal or {})
    return {"proposal": proposal}


@app.get("/api/court/landmarks")
def court_landmarks():
    """The landmark menu the calibration UI offers, with court coordinates so it
    can draw the little diagram."""
    from court import LANDMARK_LABELS, LANDMARKS, MIN_LANDMARKS

    return {
        "minimum": MIN_LANDMARKS,
        "landmarks": [{"name": name, "court": list(LANDMARKS[name]),
                       "label": LANDMARK_LABELS[name]}
                      for name in LANDMARKS],
    }


@app.get("/api/sessions")
def sessions():
    out = []
    for d in sorted(DATA.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta = read_json(d, "meta.json", {}) or {}
        rating = read_json(d, "rating.json", {}) or {}
        out.append({
            "session": d.name,
            "label": meta.get("label", d.name),
            "status": get_status(d),
            "band": rating.get("band"),
            "level": rating.get("level"),
        })
    return {"sessions": out}


@app.get("/api/session/{sid}")
def session(sid: str):
    sdir = _sdir(sid)
    return {
        "session": sid,
        "status": get_status(sdir),
        "meta": read_json(sdir, "meta.json", {}),
        "ingest": read_json(sdir, "ingest.json", {}),
        "rallies_summary": read_json(sdir, "rallies_summary.json", {}),
        "rallies": read_json(sdir, "rallies.json", []),
        "calibration_frame": read_json(sdir, "calibration_frame.json", {}),
        "court_proposal": read_json(sdir, "court_proposal.json", {}),
        "camera": read_json(sdir, "camera.json", {}),
        "calibrated": (sdir / "calibration.json").exists(),
        "quality": read_json(sdir, "quality.json", {}),
        "subject_by_rally": read_json(sdir, "subject_by_rally.json", {}),
        "rotation": read_json(sdir, "rotation.json", {}),
        "jumps": read_json(sdir, "jumps.json", {}),
        "metrics": read_json(sdir, "metrics.json", {}),
        "rating": read_json(sdir, "rating.json", {}),
        "feedback": read_json(sdir, "feedback.json", {}),
    }


@app.get("/api/session/{sid}/frame")
def frame(sid: str):
    path = _sdir(sid) / "frame0.jpg"
    if not path.exists():
        raise HTTPException(404, "calibration frame not ready yet")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/session/{sid}/plays")
def plays(sid: str):
    return read_json(_sdir(sid), "plays.json", {})


# --- rally review ----------------------------------------------------------

@app.get("/api/session/{sid}/review")
def review(sid: str):
    """Rallies to check, worst first, so 20 minutes lands where it matters."""
    import pandas as pd
    import review as review_mod

    sdir = _sdir(sid)
    info = read_json(sdir, "ingest.json", {}) or {}
    plays_raw = read_json(sdir, "plays.json", {}) or {}
    plays = {int(k): v for k, v in plays_raw.items()}
    tracks_path = sdir / "tracks.parquet"
    tracks = pd.read_parquet(tracks_path) if tracks_path.exists() else None
    return review_mod.review_payload(
        rallies=pipeline.load_rallies(sdir),
        plays_by_rally=plays,
        subject=read_json(sdir, "subject_by_rally.json", {}) or {},
        tracks=tracks,
        fps=float(info.get("fps", 30.0)),
        corrections=review_mod.Corrections.load(sdir),
    )


class ReviewPatch(BaseModel):
    subject: dict[str, list[int] | None] | None = None
    actions: dict[str, dict[str, str | None]] | None = None
    deleted: dict[str, bool] | None = None
    confirmed: dict[str, bool] | None = None
    reanalyze: bool = True


@app.post("/api/session/{sid}/review")
def submit_review(sid: str, body: ReviewPatch):
    """Record corrections and re-run everything below tracking.

    Deliberately does NOT clear derived artifacts the way calibration does:
    `analyze` rewrites them all anyway, and tracks.parquet is what makes this
    a seconds-long job rather than an hour-long one.
    """
    import review as review_mod

    sdir = _sdir(sid)
    corrections = review_mod.Corrections.load(sdir).merge(body.model_dump())
    corrections.save(sdir)
    if body.reanalyze and (sdir / "calibration.json").exists():
        pipeline.enqueue(sdir, "analyze")
    return {"ok": True, "queued": body.reanalyze}


@app.get("/api/session/{sid}/thumb/{index}")
def thumb(sid: str, index: int):
    """One frame per rally, cut lazily and cached — a match has a hundred."""
    sdir = _sdir(sid)
    rallies = read_json(sdir, "rallies.json", []) or []
    match = next((r for r in rallies if r["index"] == index), None)
    if match is None:
        raise HTTPException(404, "no such rally")
    thumbs = sdir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    out = thumbs / f"rally{index:03d}.jpg"
    if not out.exists():
        contacts = match.get("contacts") or []
        t = (contacts[0] + 0.3) if contacts else match["start"] + 0.5
        try:
            pipeline.extract_frame(sdir, t, name=f"thumbs/rally{index:03d}.jpg")
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
    return FileResponse(out, media_type="image/jpeg")


@app.get("/api/progress")
def progress():
    running = []
    for d in DATA.glob("*"):
        if not d.is_dir():
            continue
        status = get_status(d)
        if status.get("state") in ("running", "queued"):
            meta = read_json(d, "meta.json", {}) or {}
            running.append({"session": d.name, "label": meta.get("label", d.name),
                            **status})
    return {"running": running}


# --- clips -----------------------------------------------------------------

@app.get("/api/session/{sid}/clip/{index}")
def clip(sid: str, index: int):
    """Cut one rally out of the source video, on demand and cached.

    Generated lazily rather than up front: a match has a hundred rallies and
    almost none of them get watched, so pre-cutting them all would be minutes
    of ffmpeg for nothing.
    """
    sdir = _sdir(sid)
    rallies = read_json(sdir, "rallies.json", []) or []
    match = next((r for r in rallies if r["index"] == index), None)
    if match is None:
        raise HTTPException(404, "no such rally")
    clips = sdir / "clips"
    clips.mkdir(exist_ok=True)
    out = clips / f"rally{index:03d}.mp4"
    if not out.exists():
        pad = 1.5
        start = max(0.0, match["start"] - pad)
        duration = match["duration"] + 2 * pad
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(sdir / "video.mp4"),
             "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "26", "-c:a", "aac", str(out)],
            capture_output=True, check=False)
    if not out.exists():
        raise HTTPException(500, "could not cut that rally")
    return FileResponse(out, media_type="video/mp4")


class BulkDelete(BaseModel):
    sessions: list[str]


@app.post("/api/sessions/delete")
def delete_sessions(body: BulkDelete):
    removed = []
    for sid in body.sessions:
        d = DATA / sid
        if _is_within_sessions(d):
            shutil.rmtree(d)
            removed.append(sid)
    return {"deleted": removed}


# --- cross-session trend ---------------------------------------------------

@app.get("/api/trend")
def trend():
    """Per-skill levels over time — one match is a snapshot, the trajectory is
    the actual question."""
    points = []
    for d in sorted(DATA.glob("*"), key=lambda p: p.stat().st_mtime):
        rating = read_json(d, "rating.json", {}) or {}
        if not rating.get("level"):
            continue
        meta = read_json(d, "meta.json", {}) or {}
        points.append({
            "session": d.name,
            "label": meta.get("label", d.name),
            "created": meta.get("created"),
            "level": rating["level"],
            "band": rating.get("band"),
            "confidence": rating.get("confidence"),
            "dimensions": {k: v["level"]
                           for k, v in (rating.get("dimensions") or {}).items()},
        })
    return {"points": points}


if __name__ == "__main__":
    import uvicorn
    DATA.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
