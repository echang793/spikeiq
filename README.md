# SpikeIQ

Upload indoor 6v6 volleyball film, get an analysis of **your** play: hitting
percentage, serve-receive rating, blocking, defence, jump height and movement —
broken out by the position you were playing in each rally.

## Setup

```bash
brew install ffmpeg
/opt/homebrew/bin/python3.13 -m venv .venv    # torch has no 3.14 build
.venv/bin/pip install -r requirements.txt
```

YOLO weights download to `models/` on first run.

## Run

```bash
.venv/bin/python src/server.py    # http://127.0.0.1:8101
```

Upload a match, wait a few seconds for the rally scan, click the eight court
reference points and then yourself, and press Analyse.

## How to record

Recording well still helps most, but only one of these is actually required.

**Required:** audio on and unobstructed. Rallies are found from the referee's
whistle and the sound of the ball, so silent footage cannot be analysed at all.

**Strongly preferred, with the cost of each compromise:**

| | Preferred | If you can't |
|---|---|---|
| Camera | Fixed tripod, elevated corner behind the endline, 3–4 m up | Handheld is supported. Drift and shake around one framing are tracked and removed; a camera swung between ends is not, and the report says so. Expect a few percent of frames to be dropped. |
| Framing | Whole court, both endlines and the free zone | A cropped court is supported — click any four reference points you *can* see and the rest is extrapolated. But **an endline out of frame costs you hitting percentage**: the server standing behind it is how rally winners are worked out, so kills and errors go unscored. |
| Resolution | 1080p, 60 fps | 30 fps works, blurrier at spike speed. |
| Position | Same spot every match | Different spots still analyse; cross-match trends compare less cleanly. |

## Calibrating

After the rally scan, SpikeIQ tries to find the court itself and draws it over
the frame. If it looks right, one click accepts it.

When it can't — a shallow angle, a worn floor, players standing on the lines —
you place points by hand: pick a reference point you can see from the list
(corners, attack-line ends, centre-line ends) and click it on the frame. **Any
four is enough**, and they do not have to be corners, which is what makes a
partly visible court workable. The report tells you which parts of the court were
measured and which were extrapolated.

If several rectangles fit the floor markings, one click inside the court
disambiguates it.

## How it works

Audio comes first, because a volleyball match is only about a quarter
ball-in-play and the referee brackets every rally for free:

1. **Whistles vs contacts.** A whistle is a sustained narrowband 1.8-4.5 kHz
   tone; a ball contact is a broadband transient. Different detectors, cleanly
   separated.
2. **Rallies** are the whistle-bracketed intervals that actually contain
   contacts — timeouts and substitutions fall out on their own. Footage with no
   referee falls back to silence-gap segmentation.
3. **Tracking** (YOLOv8-pose + ByteTrack) runs *only inside rally windows*.
4. **Rally winners come free from the side-out rule**: whoever wins a rally
   serves the next one, so `winner(N) = serving_side(N+1)`. No ball tracking, no
   scoreboard reading, no manual tagging. This is what makes kills, errors and
   hitting percentage computable.
5. **The rally grammar** decodes each rally with Viterbi over the sequence of
   contacts. A forearm pass and a dig look *identical* — what separates them is
   that you pass a serve and dig an attack, so the whole rally is decoded at
   once rather than each touch independently. The touch count is carried in the
   decoder state, including that a block is not one of the three touches.
6. **Position awareness.** The rotational slot at the serve and the zone he
   actually plays are tracked separately, so every stat is reported per
   position — a player who rotates through all six is not described by one
   blended number.

| Stage | Artifact |
|---|---|
| ingest | `video.mp4`, `audio.wav`, `frame0.jpg`, `ingest.json` |
| audio | `whistles.json`, `contacts_audio.json` |
| rallies | `rallies.json` |
| tracking | `tracks.parquet` (pixel space, survives re-calibration) |
| subject | `subject.parquet` |
| rotation | `rotation.json` |
| contacts | `contacts.parquet` |
| grammar | `plays.json` |
| jump | `jumps.json` |
| metrics | `metrics.json` |
| rating | `rating.json` |
| feedback | `feedback.json` |

## Reviewing a match

The analysis proposes; you confirm. After it finishes, **Review rallies** lists
them worst-first — the ones where it could not tell which player was you, could
not work out who won, or could not read the touches. Click yourself on a
thumbnail to fix the subject, change any action label from the dropdowns, mark a
rally "looks right" or "not a rally", then save. Re-running skips tracking
entirely, so corrections cost seconds.

Fixing the top few rallies is usually enough. Your corrections also become a
labelled dataset:

```bash
.venv/bin/python scripts/export_labels.py --out labels.jsonl
.venv/bin/python scripts/accuracy.py labels.jsonl
```

That prints per-action recall, precision and a confusion matrix, and enforces
the standing gate — 80% on 50 labelled contacts — before the skill numbers mean
anything.

## Speed

An hour of 60 fps film takes about 39 minutes to track on an M1. Decode is free;
the pose model is the entire cost. `SPIKEIQ_MODEL=yolov8n-pose.pt` roughly halves
it — measure with `scripts/bench_tracking.py`, and check accuracy before trusting
the faster model on far-court players.

## What it does not claim

- **The rubric is uncalibrated.** Level bands (B/BB/A/AA/Open) come from
  informed anchors in `rating.RUBRIC`, not from film of players whose level is
  known. They are a consistent yardstick for tracking your own progress, not a
  placement against the field.
- **Rates carry their denominators** and a `low_sample` flag. Two swings at
  100% is reported as two swings at 100%.
- **Unknown stays unknown.** The last rally of a set has no next serve, so it
  has no winner and is excluded from kill/error rates rather than guessed.
- **No ball tracking yet.** Shot direction (line vs cross) is therefore not
  reported. Everything above works without it.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```
