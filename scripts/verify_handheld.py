#!/usr/bin/env python
"""Measure camera-motion accuracy on a real clip, against known ground truth.

Warps a clip by a homography we chose, runs the solver, and reports how well it
recovered it. Unlike the unit tests, which use a synthetic gym image, this runs
on real footage with real compression, motion blur and moving players.

    .venv/bin/python scripts/verify_handheld.py path/to/clip.mp4

One subtlety worth keeping, learned the hard way: broadcast and phone footage
often has camera motion of its OWN. Comparing the solve against only the warp we
added then attributes the source's real movement to solver error — the first run
of this reported 47 px of "error" that turned out to be the solver correctly
tracking a genuine broadcast camera move. So the clip is solved twice, warped and
unwarped, and the source's own motion is composed out:

    W_warped  ==  A_0 · W_unwarped · A_t⁻¹

which isolates exactly the warp we introduced.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camera import CameraSolver  # noqa: E402

SAMPLE_EVERY = 30      # solve one frame a second; the solve is the expensive part
PROBE = np.float32([[300, 400], [900, 500], [640, 620], [200, 250]]
                   ).reshape(-1, 1, 2)


def wobble(i: int, fps: float, size, amount: float = 1.0) -> np.ndarray:
    """A plausible handheld path: slow sway plus a small breathing rotation."""
    w, h = size
    t = i / fps
    dx = amount * (14 * math.sin(2 * math.pi * t / 6.0)
                   + 5 * math.sin(2 * math.pi * t / 1.7))
    dy = amount * (-9 * math.sin(2 * math.pi * t / 4.3))
    rot = amount * 0.8 * math.sin(2 * math.pi * t / 5.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return np.vstack([M, [0, 0, 1]])


def make_handheld(src: Path, dst: Path, amount: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    truth, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H = wobble(i, fps, (w, h), amount)
        truth.append(H)
        out.write(cv2.warpPerspective(frame, H, (w, h)))
        i += 1
    cap.release()
    out.release()
    return truth


def solve(path: Path, detect_cuts: bool = False):
    cap = cv2.VideoCapture(str(path))
    ok, ref = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {path}")
    solver = CameraSolver(ref, detect_cuts=detect_cuts)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % SAMPLE_EVERY == 0:
            solver.update(i, frame)
    cap.release()
    return solver.track


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip")
    ap.add_argument("--amount", type=float, default=1.0,
                    help="scale the synthetic wobble (2.0 = twice as shaky)")
    ap.add_argument("--tolerance", type=float, default=6.0,
                    help="max acceptable recovery error, pixels")
    args = ap.parse_args()

    src = Path(args.clip)
    if not src.exists():
        print(f"{src} not found", file=sys.stderr)
        return 2
    warped = src.with_name(src.stem + "_handheld.mp4")

    print(f"warping {src.name} ...")
    truth = make_handheld(src, warped, args.amount)

    print("solving the source clip (to measure its own motion) ...")
    unwarped_track = solve(src, detect_cuts=True)
    own = unwarped_track.summary()
    print(f"  source regime: {own['regime']}, median motion "
          f"{own['median_motion_px']}px, max {own['max_motion_px']}px, "
          f"cuts {own['cuts']}")
    if own["max_motion_px"] > 20:
        print("  NOTE: this clip moves on its own; composing that out below.")

    print("solving the warped clip ...")
    warped_track = solve(warped)

    errs = []
    for idx in sorted(warped_track.warps):
        base = unwarped_track.warp_for(idx)
        if base is None:
            continue
        expect = truth[0] @ base @ np.linalg.inv(truth[idx])
        a = cv2.perspectiveTransform(PROBE, expect).reshape(-1, 2)
        b = cv2.perspectiveTransform(PROBE, warped_track.warp_for(idx)).reshape(-1, 2)
        errs.append(float(np.max(np.hypot(*(a - b).T))))

    if not errs:
        print("\nno frames could be compared — the solver found nothing")
        return 1

    e = np.array(errs)
    s = warped_track.summary()
    print(f"\nrecovered the added warp over {len(e)} sampled frames")
    print(f"  median {np.median(e):.2f}px   mean {e.mean():.2f}px   "
          f"p90 {np.percentile(e, 90):.2f}px   max {e.max():.2f}px")
    print(f"  regime {s['regime']}, solved {s['solved_fraction']:.0%} of frames")

    ok = float(np.median(e)) <= args.tolerance
    print(f"\n{'PASS' if ok else 'FAIL'}: median error "
          f"{np.median(e):.2f}px against a {args.tolerance:.1f}px tolerance")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
