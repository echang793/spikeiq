#!/usr/bin/env python
"""Where the tracking time actually goes.

Tracking is the only expensive stage, so it sets how long a match takes and
how slow every experiment is. Before changing anything it is worth knowing
which part is slow: decoding frames, running the pose model, or the per-call
overhead of driving ultralytics one frame at a time instead of letting it
stream.

    .venv/bin/python scripts/bench_tracking.py path/to/clip.mp4

Reports ms/frame for each, and projects an hour-long 60 fps match at the
pipeline's real duty cycle.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

MODELS = ROOT / "models"
# a match is roughly this fraction ball-in-play, which is the whole reason
# tracking is restricted to rally windows
DUTY_CYCLE = 0.28
MATCH_MINUTES = 60


def timed(fn, n: int) -> float:
    """ms per iteration."""
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0 / max(n, 1)


def bench_decode(video: Path, stride: int, limit: int) -> tuple[float, float]:
    """Cost of pulling frames, with and without the colour convert + copy."""
    def grab_only():
        cap = cv2.VideoCapture(str(video))
        for _ in range(limit):
            if not cap.grab():
                break
        cap.release()

    def full_read():
        cap = cv2.VideoCapture(str(video))
        for i in range(limit):
            if i % stride:
                if not cap.grab():
                    break
            elif not cap.read()[0]:
                break
        cap.release()

    return timed(grab_only, limit), timed(full_read, limit)


def bench_model(video: Path, model_name: str, mode: str, stride: int,
                limit: int) -> tuple[float, int]:
    """ms per PROCESSED frame for a given model and driving style."""
    from ultralytics import YOLO
    model = YOLO(str(MODELS / model_name))

    if mode == "stream":
        start = time.perf_counter()
        n = 0
        for _ in model.track(source=str(video), stream=True, persist=True,
                             classes=[0], tracker="bytetrack.yaml",
                             verbose=False, vid_stride=stride):
            n += 1
            if n * stride >= limit:
                break
        return (time.perf_counter() - start) * 1000.0 / max(n, 1), n

    cap = cv2.VideoCapture(str(video))
    n = 0
    start = time.perf_counter()
    for i in range(limit):
        ok, frame = (cap.read() if i % stride == 0 else (cap.grab(), None))
        if i % stride:
            continue
        if not ok:
            break
        if mode == "track":
            model.track(frame, persist=True, classes=[0],
                        tracker="bytetrack.yaml", verbose=False)
        else:
            model.predict(frame, classes=[0], verbose=False)
        n += 1
    cap.release()
    return (time.perf_counter() - start) * 1000.0 / max(n, 1), n


def project(ms_per_frame: float, stride: int, bridge: bool) -> float:
    """Minutes to track an hour-long 60 fps match at the real duty cycle."""
    from tracking import BRIDGE_FPS
    source_fps = 60.0
    rally_frames = MATCH_MINUTES * 60 * source_fps * DUTY_CYCLE / stride
    bridge_frames = (MATCH_MINUTES * 60 * (1 - DUTY_CYCLE) * BRIDGE_FPS
                     if bridge else 0.0)
    return (rally_frames + bridge_frames) * ms_per_frame / 1000.0 / 60.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--frames", type=int, default=240,
                    help="source frames to walk")
    ap.add_argument("--models", default="yolov8s-pose.pt,yolov8n-pose.pt")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"{video} not found", file=sys.stderr)
        return 2

    grab_ms, read_ms = bench_decode(video, args.stride, args.frames)
    print(f"\ndecode        grab-only {grab_ms:6.2f} ms/frame"
          f"   read+convert {read_ms:6.2f} ms/frame")
    print("(decode is per SOURCE frame; everything below is per PROCESSED "
          "frame)\n")

    print(f"{'model':<18}{'mode':<10}{'ms/frame':>10}{'match (min)':>14}")
    results = {}
    for name in args.models.split(","):
        name = name.strip()
        if not (MODELS / name).exists():
            print(f"{name:<18}{'-':<10}{'not downloaded':>10}")
            continue
        for mode in ("predict", "track", "stream"):
            try:
                ms, n = bench_model(video, name, mode, args.stride, args.frames)
            except Exception as exc:  # noqa: BLE001 - a mode may be unsupported
                print(f"{name:<18}{mode:<10}{'failed':>10}  {exc}")
                continue
            results[(name, mode)] = ms
            print(f"{name:<18}{mode:<10}{ms:>10.1f}"
                  f"{project(ms, args.stride, bridge=True):>14.1f}")

    base = results.get(("yolov8s-pose.pt", "track"))
    if base:
        print(f"\nbaseline (current pipeline): "
              f"{project(base, args.stride, bridge=True):.0f} min/match")
        for (name, mode), ms in sorted(results.items(), key=lambda kv: kv[1]):
            if ms < base:
                print(f"  {name} / {mode}: {base / ms:.2f}x faster "
                      f"-> {project(ms, args.stride, bridge=True):.0f} min")
    print("\nNote: the smaller model is only a win if its accuracy holds up. "
          "Check with scripts/accuracy.py before adopting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
