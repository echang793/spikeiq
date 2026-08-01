"""Player detection + tracking (YOLOv8-pose + ByteTrack) and subject selection.

Ported from dinkiq/src/tracking.py, with two volleyball-driven changes:

1. Tracking runs only inside rally windows. A volleyball match is ~25-30 %
   ball-in-play; running pose inference over the dead time would triple the
   cost and can only manufacture phantom events. We still decode the video
   sequentially (cv2 `grab()` skips a frame far more cheaply than seeking, and
   avoids keyframe-seek inaccuracy) but run inference only on wanted frames.
2. `pick_opponents` is gone. It assumed one or two opponents; with twelve
   players on court, team membership comes from which side of the net a track
   spends its time on (`assign_sides`).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from court import NET_Y, CourtCalibration, on_court

MODEL_NAME = "yolov8s-pose.pt"

# COCO keypoint indices we keep. Wrists drive contact classification, ankles
# drive jump height, shoulders/hips give the body reference heights that tell a
# set (hands above head) from a pass (platform below the waist).
KEYPOINTS = {
    "nose": 0, "lsh": 5, "rsh": 6, "lw": 9, "rw": 10,
    "lhip": 11, "rhip": 12, "lank": 15, "rank": 16,
}

COLUMNS = (["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]
           + [f"{name}{axis}" for name in KEYPOINTS for axis in ("x", "y")])

TRACK_STRIDE = 2  # process every 2nd frame; 1 = accuracy mode


def wanted_frames(n_frames: int, fps: float, stride: int,
                  windows: list[tuple[float, float]] | None) -> np.ndarray:
    """Frame indices to run inference on: every `stride`-th frame that falls
    inside one of the (start, end) second-windows. `windows=None` means the
    whole video (used before rallies are known, and by the tests).
    """
    idx = np.arange(0, n_frames, stride)
    if not windows:
        return idx
    t = idx / fps
    keep = np.zeros(len(idx), dtype=bool)
    for start, end in windows:
        keep |= (t >= start) & (t <= end)
    return idx[keep]


def run_tracking(video: Path, out_parquet: Path, models_dir: Path,
                 windows: list[tuple[float, float]] | None = None,
                 progress_cb=None, frame_cb=None,
                 stride: int | None = None) -> pd.DataFrame:
    """Track every person inside the rally windows; write the tracks parquet.

    Frame numbers written to the parquet are REAL video frame indices, so time
    is always `frame / fps` regardless of stride or windowing.

    Single-decode integration is preserved from dinkiq: `frame_cb(frame_idx,
    image, boxes_xyxy)` fires for each processed frame so a later ball detector
    can ride the same decode instead of reading the video a second time.
    """
    if stride is None:
        stride = TRACK_STRIDE
    model = YOLO(str(models_dir / MODEL_NAME))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    targets = set(wanted_frames(n_frames, fps, stride, windows).tolist())

    rows: list[tuple] = []
    done = 0
    try:
        idx = 0
        while True:
            if idx not in targets:
                # grab() advances the decoder without the colour-convert and
                # copy that retrieve() does — the cheap way to skip.
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            # NOTE: do NOT pass half=True — on MPS it hits a deprecated fp16
            # fallback that is ~27x SLOWER (measured in dinkiq: 1430 ms/frame
            # vs 53 ms/frame).
            r = model.track(frame, persist=True, classes=[0],
                            tracker="bytetrack.yaml", verbose=False)[0]
            xyxy = None
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                if r.keypoints is not None and r.keypoints.xy is not None:
                    kps = r.keypoints.xy.cpu().numpy()  # (n,17,2); (0,0)=missing
                else:
                    kps = np.zeros((len(ids), 17, 2))
                for tid, box, conf, kp in zip(ids, xyxy, confs, kps):
                    flat = [v for i in KEYPOINTS.values() for v in kp[i].tolist()]
                    rows.append((idx, int(tid), *box.tolist(), float(conf), *flat))
            if frame_cb is not None:
                frame_cb(idx, frame, xyxy)
            done += 1
            if progress_cb and done % 30 == 0:
                progress_cb(idx, len(targets), done)
            idx += 1
    finally:
        cap.release()

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_parquet(out_parquet, index=False)
    return df


def feet_px(df: pd.DataFrame) -> np.ndarray:
    """Bottom-centre of each bbox: the feet contact point used for homography."""
    return np.stack([(df["x1"] + df["x2"]) / 2.0, df["y2"]], axis=1)


def pick_subject(df: pd.DataFrame, click_xy: tuple[float, float],
                 near_frame: int | None = None, window: int = 90) -> int:
    """Track id whose bbox centre is nearest the user's 'this is me' click.

    `near_frame` scopes the search to the frame the user was actually looking
    at when they clicked — with twelve players who all move, matching against
    the whole video would let a completely different player win on a later
    frame that happens to pass under the click point.
    """
    if df.empty:
        raise ValueError("no tracks found in video")
    if near_frame is None:
        near_frame = int(df["frame"].min())
    near = df[(df["frame"] >= near_frame - window) & (df["frame"] <= near_frame + window)]
    if near.empty:
        near = df
    cx = (near["x1"] + near["x2"]) / 2.0
    cy = (near["y1"] + near["y2"]) / 2.0
    d2 = (cx - click_xy[0]) ** 2 + (cy - click_xy[1]) ** 2
    return int(near.assign(d2=d2).groupby("track_id")["d2"].min().idxmin())


MAX_STITCH_GAP = 45      # frames the subject may vanish before we stop following
STITCH_OVERLAP = 20      # a continuation may START this many frames BEFORE the
                         # current fragment ends (tracker fragments overlap)
STITCH_JUMP_PX = 12.0    # allowed bbox-centre drift per missing frame
STITCH_BASE_PX = 60.0    # base allowance: an id switch mid-motion jumps a bit


def stitch_subject(df: pd.DataFrame, first_id: int,
                   max_gap: int = MAX_STITCH_GAP) -> pd.DataFrame:
    """Follow the subject across track-id breaks by position continuity.

    Trackers fragment ids on occlusion, and with twelve bodies crossing in
    front of each other that happens constantly. Starting from the clicked
    track, whenever the current fragment ends we adopt the unused track whose
    position at the handover point is nearest, searching a window that includes
    the overlap where the replacement id starts before the old one dies.
    """
    df = df.sort_values("frame")
    df = df.assign(cx=(df["x1"] + df["x2"]) / 2.0, cy=(df["y1"] + df["y2"]) / 2.0)
    by_id = {tid: g.reset_index(drop=True) for tid, g in df.groupby("track_id")}
    if first_id not in by_id:
        raise ValueError(f"track id {first_id} not present")
    starts = {tid: int(g["frame"].iloc[0]) for tid, g in by_id.items()}

    chain = [first_id]
    used = {first_id}
    cur = by_id[first_id]
    while True:
        last = cur.iloc[-1]
        last_f = int(last["frame"])
        best_id, best_score = None, np.inf
        for tid, f0 in starts.items():
            if tid in used or not (last_f - STITCH_OVERLAP < f0 <= last_f + max_gap):
                continue
            g = by_id[tid]
            at = g[g["frame"] >= last_f]
            head = at.iloc[0] if len(at) else g.iloc[-1]
            gap = max(0, f0 - last_f)
            d = float(np.hypot(head["cx"] - last["cx"], head["cy"] - last["cy"]))
            if d <= STITCH_BASE_PX + STITCH_JUMP_PX * gap and d + 2.0 * gap < best_score:
                best_id, best_score = tid, d + 2.0 * gap
        if best_id is None:
            break
        used.add(best_id)
        chain.append(best_id)
        cur = by_id[best_id]

    out = df[df["track_id"].isin(chain)].drop(columns=["cx", "cy"])
    # a frame can appear in overlapping fragments — keep the first occurrence
    return out.sort_values("frame").drop_duplicates("frame").reset_index(drop=True)


def stitch_chain_ids(df: pd.DataFrame, first_id: int) -> set[int]:
    """Track ids in the subject's stitched chain (see stitch_subject)."""
    return set(stitch_subject(df, first_id)["track_id"].unique().tolist())


MIN_TRACK_FRAMES = 15


def assign_sides(df: pd.DataFrame, calib: CourtCalibration) -> dict[int, str]:
    """Map each track id to the side of the net it plays on ('far' / 'near').

    Replaces dinkiq's `pick_opponents`, which assumed at most two opponents. A
    track is assigned by majority vote over its on-court frames, so a defender
    who chases a ball across the endline still counts for their own side. Track
    ids that are never on court (spectators, the referee stand) are omitted
    rather than forced onto a side.
    """
    sides: dict[int, str] = {}
    for tid, g in df.groupby("track_id"):
        if len(g) < MIN_TRACK_FRAMES:
            continue
        pts = calib.to_court(feet_px(g))
        keep = np.array([on_court(x, y) for x, y in pts])
        if not keep.any():
            continue
        ys = pts[keep, 1]
        sides[int(tid)] = "far" if float(np.mean(ys < NET_Y)) >= 0.5 else "near"
    return sides


def subject_court_positions(df: pd.DataFrame, subject_id: int,
                            calib: CourtCalibration, fps: float) -> pd.DataFrame:
    """Subject's feet in court metres per frame, off-court points dropped."""
    sub = stitch_subject(df, subject_id)
    pts = calib.to_court(feet_px(sub))
    out = pd.DataFrame({
        "frame": sub["frame"].to_numpy(),
        "t": sub["frame"].to_numpy() / fps,
        "x": pts[:, 0],
        "y": pts[:, 1],
    })
    mask = [on_court(x, y) for x, y in zip(out["x"], out["y"])]
    return out[mask].reset_index(drop=True)
