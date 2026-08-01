# spikeiq

## Purpose

Indoor 6v6 volleyball film analysis: upload match video, get per-skill and
per-position strengths and weaknesses for one player (the subject).

## Stack

Python 3.13, FastAPI, ultralytics YOLOv8-pose + ByteTrack, librosa, OpenCV,
pandas/Parquet, ffmpeg. Torch runs on MPS.

## Commands

```bash
.venv/bin/python src/server.py            # http://127.0.0.1:8101
.venv/bin/python -m pytest tests/ -q      # full suite
```

Venv is Python 3.13 (torch has no 3.14 build); ffmpeg via brew.

## Architecture

Two-phase pipeline, split because the audio pass is cheap and the video pass is
not:

- `pipeline.prepare` — ingest, whistle/contact detection, rally segmentation,
  calibration frame. Audio only, seconds.
- `pipeline.analyze` — tracking, subject, rotation, contacts, grammar, jump,
  metrics, rating, feedback. Needs calibration + subject click.

Module map: `court` (geometry/homography) → `tracking` (YOLO + stitch) →
`audio` → `rallies` → `contacts` (pose features) → `grammar` (Viterbi) →
`rotation` / `jump` → `metrics` → `rating` → `feedback`.

## Invariants

- **Tracking runs at full stride only inside rally windows**, plus a sparse
  ~2 fps trickle between them (`tracking.BRIDGE_FPS`). The trickle is not
  optional: without it the subject's identity cannot survive the dead ball, and
  every stat collapses to the first rally. Costs ~20% more tracking time.
- **The subject is resolved per rally** (`tracking.resolve_subject`), never once
  per match. `metrics`/`rotation`/`jump` all take `dict[rally_index, set[ids]]`.
- **Bridging is proven by temporal continuity, not shared track ids.** A track id
  recurring after a gap proves nothing about whether it is the same person.
- **Proximity re-anchoring is capped at low confidence** (`PROXIMITY_MAX_CONF`).
  Two teammates can swap places during a dead ball, and when they do, "nearest
  to where he was" is confidently wrong with no signal able to detect it.
- **Contact attribution is constrained by ball flight** (`BALL_MAX_SPEED`). One
  bad attribution corrupts a whole rally, because `grammar.decode` treats the
  attributed side as observed fact.
- **`quality.assess` gates the report.** Below threshold the level estimate is
  suppressed entirely. With no footage to validate against, being wrong quietly
  is the failure that matters.
- **`rating.RUBRIC` is the single tuning point.** Its anchors are informed
  guesses, NOT fitted to data — every output must keep saying so.
- **`tracks.parquet` is pixel space** and survives re-calibration. Court-derived
  artifacts are disposable and listed in `pipeline.DERIVED`.
- **Frame numbers in parquets are real video frames**; time = frame / fps.
- **Rally winners come from the side-out rule** (`winner(N) = serving_side(N+1)`),
  never guessed. Unknown stays `None` — it propagates into the kill/error split,
  so a guess there would silently corrupt hitting percentage.
- **Never chain winners across a set boundary** — teams switch ends.
- **Subject side is computed per rally** from his own tracked position, which is
  what makes the between-set end switch a non-issue.
- **Blocks do not consume one of a side's three touches** (`grammar._next_touches`).
- **Illegal action sequences are penalised, never forbidden.** One missed soft
  contact must not derail the rest of the rally's decode.
- **Honest degradation**: every rate carries its denominator and a `low_sample`
  flag; unscored rallies are excluded from rates, not counted as errors.
- **Single analysis worker** (`pipeline.enqueue`) — parallel analyses contend
  for MPS and end up slower.
- Never pass `half=True` to the YOLO tracker — the MPS fp16 fallback is ~27x
  slower (measured in the sibling dinkiq project).

## Performance (measured, `scripts/bench_tracking.py`, M1)

| | ms/frame | projected hour-long 60 fps match |
|---|---|---|
| yolov8s-pose (default) | 66 | 39 min |
| yolov8n-pose | 34 | 20 min |

- **Decode is free** (0.4 ms/frame). Inference is the entire cost.
- **`stream=True` is NOT faster** than per-frame `model.track` (66.0 vs 66.1).
  The per-call overhead hypothesis was measured and disproved — don't re-litigate.
- Nano is ~1.9x faster and lost nothing on the pickleball test clip (slightly
  more detections, identical keypoint completeness). It is still not the default,
  because that clip has large near-court players and the volleyball risk is the
  opposite: a far-endline player is a third the pixel height, and missing them
  breaks subject resolution and attribution. Switch with `SPIKEIQ_MODEL=yolov8n-pose.pt`
  and confirm with `scripts/accuracy.py` once there is footage.

## Gotchas

- pandas 3 returns **read-only** arrays from `.to_numpy()`; `.copy()` before
  masking in place.
- A PostToolUse hook runs `ruff check --fix` and DELETES unused imports — add
  imports in the same edit as the code using them, and verify by importing the
  module (`py_compile` passes with missing imports).
- Jump height uses the player's **standing height in pixels** as the ruler, not
  the homography: the homography maps the ground plane and overstates vertical
  distance by a tilt-dependent factor.
- Keypoints at `(0,0)` mean "not detected", not "at the origin" — blind
  averaging drags midpoints toward the top-left corner.
- Keep frontend URLs `BASE`-prefixed; without it a subpath mount loads the page
  but 404s every API call.
- UI copy: levels "resemble" a club band. Never present them as a placement.

## Related

Sibling project `~/Desktop/dinkiq` is the pickleball equivalent and the source
of `court`, `tracking`, `pipeline` and the server scaffolding. Fixes to shared
logic are worth checking against it.
