"""SpikeIQ HTTP server.

Flow: upload -> prepare (audio only, seconds) -> you calibrate the court and
click yourself on the first rally's frame -> analyze -> report.

Mount-prefix handling is carried over from dinkiq and is load-bearing even
though this runs at the root today: `_spa` injects `window.__BASE__` from an
`X-Forwarded-Prefix` header, and the frontend prefixes every API call with it.
Without that, mounting behind a reverse proxy subpath serves the page fine but
404s every API call, which looks like the app silently doing nothing.
"""

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


@app.middleware("http")
async def _basic_auth(request: Request, call_next):
    """Password-gate everything except localhost, when a password is set."""
    password = os.environ.get("SPIKEIQ_PASSWORD")
    client = request.client.host if request.client else ""
    if password and client not in ("127.0.0.1", "::1", "localhost"):
        import base64
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Basic "):
            try:
                _, _, supplied = base64.b64decode(header[6:]).decode().partition(":")
                ok = supplied == password
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


def _sdir(sid: str) -> Path:
    d = DATA / sid
    if not d.exists():
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

class Calibration(BaseModel):
    corners: list[list[float]]
    attack: list[list[float]] | None = None
    subject_click: list[float]
    subject_click_t: float | None = None
    height_m: float | None = None


@app.post("/api/session/{sid}/calibrate")
def calibrate(sid: str, body: Calibration):
    from court import CourtCalibration

    sdir = _sdir(sid)
    try:
        calib = CourtCalibration(body.corners, body.attack)
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
        # guard against a crafted id escaping the sessions directory
        if d.is_dir() and d.resolve().parent == DATA.resolve():
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
