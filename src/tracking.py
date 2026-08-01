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

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from court import NET_Y, CourtCalibration, on_court

# Pose model. Benchmarked on an M1 (scripts/bench_tracking.py): the nano model
# is ~1.9x faster — about 20 minutes per match against 39 — and on the test clip
# it lost nothing, detecting marginally MORE people per frame with identical
# keypoint completeness.
#
# The default stays on the small model anyway, because that clip is broadcast
# footage of large, near-court players and the case that matters here is the
# opposite one: a player at the far endline of a volleyball court is a third the
# pixel height, and under-detecting them breaks subject resolution and contact
# attribution, which cost far more than the wait. Switch with SPIKEIQ_MODEL and
# confirm with scripts/accuracy.py once there is footage to confirm against.
DEFAULT_MODEL = "yolov8s-pose.pt"


def model_name() -> str:
    return os.environ.get("SPIKEIQ_MODEL", DEFAULT_MODEL)



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

# Sparse frames sampled BETWEEN rallies, purely to keep player identity alive
# across the dead ball. Skipping dead time entirely is what made the subject
# unresolvable after rally one: twenty seconds later, six identically-dressed
# teammates have all moved, and position alone cannot say which one is him.
# At this rate players move a metre or two between samples, which ordinary
# proximity tracking handles, so identity survives the gap for about a fifth
# more tracking time.
BRIDGE_FPS = 2.0


def wanted_frames(n_frames: int, fps: float, stride: int,
                  windows: list[tuple[float, float]] | None,
                  bridge_fps: float | None = BRIDGE_FPS) -> np.ndarray:
    """Frame indices to run inference on.

    Every `stride`-th frame inside a rally window, plus a sparse trickle of
    frames outside them (`bridge_fps`) so tracks survive the dead ball.
    `windows=None` means the whole video. `bridge_fps=None` disables bridging,
    which is the old behaviour and is kept for benchmarking the cost.
    """
    idx = np.arange(0, n_frames, stride)
    if not windows:
        return idx
    t = idx / fps
    keep = np.zeros(len(idx), dtype=bool)
    for start, end in windows:
        keep |= (t >= start) & (t <= end)
    if bridge_fps:
        # snap the bridge interval to a whole number of strides, so the samples
        # land on the stride grid and come out evenly spaced
        every = max(1, int(round(fps / bridge_fps / stride)))
        bridge = np.zeros(len(idx), dtype=bool)
        bridge[::every] = True
        keep |= bridge
    return idx[keep]


def bridge_gap_frames(fps: float, bridge_fps: float = BRIDGE_FPS) -> int:
    """Frame gap the subject may vanish for once bridging is on.

    Derived from the bridge rate rather than a constant: with samples every
    `fps / bridge_fps` frames, a stitching allowance smaller than that spacing
    guarantees the chain breaks at the first dead ball, which is the bug this
    replaces.
    """
    return int(round(fps / max(bridge_fps, 1e-6) * 2.5))


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
    model = YOLO(str(models_dir / model_name()))

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


# --- subject identity across a whole match ---------------------------------

MAX_REANCHOR_M = 9.0      # a player cannot be anywhere but their own half
MIN_MARGIN_M = 1.5        # runner-up this close means we genuinely do not know
MIN_RALLY_FRAMES = 4
# Proximity across the dead ball is capped low on purpose. Twenty seconds is
# long enough for two teammates to swap places, and when they do, "nearest to
# where he was" is not merely uncertain — it is confidently wrong, and there is
# no signal in position alone that can tell the two cases apart. Bridging is the
# real answer; this is a hint for the review UI to confirm, never an assertion.
PROXIMITY_MAX_CONF = 0.45


@dataclass
class SubjectResolution:
    """Per-rally identity, with an explicit confidence for each rally.

    Confidence matters more here than anywhere else in the pipeline. Bridging
    (see BRIDGE_FPS) makes identity reliable when tracking held through the dead
    ball, but when it did not, the fallback is proximity among six teammates in
    matching jerseys — which is genuinely weak, and says so rather than
    pretending. Low-confidence rallies are what the review UI puts in front of
    the user first.
    """
    ids_by_rally: dict[int, set[int]]
    confidence_by_rally: dict[int, float]
    method_by_rally: dict[int, str]      # 'bridged' | 'proximity' | 'none'

    @property
    def resolved_fraction(self) -> float:
        if not self.ids_by_rally:
            return 0.0
        found = sum(1 for v in self.ids_by_rally.values() if v)
        return found / len(self.ids_by_rally)

    def as_dict(self) -> dict:
        return {
            "ids_by_rally": {str(k): sorted(v)
                             for k, v in self.ids_by_rally.items()},
            "confidence_by_rally": {str(k): round(v, 3)
                                    for k, v in self.confidence_by_rally.items()},
            "method_by_rally": {str(k): v for k, v in self.method_by_rally.items()},
            "resolved_fraction": round(self.resolved_fraction, 3),
        }


def rally_segment(tracks: pd.DataFrame, rally, fps: float) -> pd.DataFrame:
    f0, f1 = int(rally.start * fps), int(rally.end * fps)
    return tracks[(tracks["frame"] >= f0) & (tracks["frame"] <= f1)]


def _track_court_centre(g: pd.DataFrame, calib: CourtCalibration):
    pts = calib.to_court(feet_px(g))
    keep = np.array([on_court(x, y) for x, y in pts])
    if not keep.any():
        return None
    return np.median(pts[keep], axis=0)


def _side_of(point) -> str:
    return "far" if point[1] < NET_Y else "near"


def _candidates(seg: pd.DataFrame, calib: CourtCalibration,
                anchor_side: str | None) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for tid, g in seg.groupby("track_id"):
        if len(g) < MIN_RALLY_FRAMES:
            continue
        centre = _track_court_centre(g, calib)
        if centre is None:
            continue
        # players do not swap ends mid-set, so an opponent is never him
        if anchor_side is not None and _side_of(centre) != anchor_side:
            continue
        out[int(tid)] = centre
    return out


def resolve_subject(tracks: pd.DataFrame, rallies, seed_id: int, seed_rally: int,
                    calib: CourtCalibration, fps: float) -> SubjectResolution:
    """Follow the subject across every rally of the match.

    Two mechanisms, in order of trustworthiness:

    1. **Bridged.** If tracking sampled the dead ball (`BRIDGE_FPS`), the ordinary
       within-match stitch already reaches into the next rally and identity is
       simply carried. This is the reliable path and the reason bridging exists.
    2. **Proximity.** Otherwise, pick the nearest track on the subject's own half
       to where he was last seen. With six teammates in the same jersey this is
       weak by construction; it reports low confidence, and reports *nothing*
       when the runner-up is nearly as close.

    Walks outward from the rally the user actually clicked on, in both
    directions, and keeps going after a rally it cannot resolve — losing him for
    one rally must not end the match.
    """
    by_index = {r.index: r for r in rallies}
    order = sorted(by_index)
    if seed_rally not in by_index and order:
        seed_rally = order[0]

    ids: dict[int, set[int]] = {i: set() for i in order}
    conf: dict[int, float] = {i: 0.0 for i in order}
    method: dict[int, str] = {i: "none" for i in order}
    if not order:
        return SubjectResolution(ids, conf, method)

    max_gap = bridge_gap_frames(fps)
    # one whole-match stitch: when bridge frames exist this alone spans the
    # match, and each rally reads its slice out of the chain. What counts as
    # evidence is the chain having FRAMES inside a rally, not merely sharing a
    # track id with it — an id that happens to recur after the dead ball proves
    # nothing about whether it is still the same person.
    chain = None
    if seed_id in set(tracks["track_id"].unique().tolist()):
        try:
            chain = stitch_subject(tracks, seed_id, max_gap=max_gap)
        except ValueError:
            chain = None

    anchor = None
    anchor_side = None
    seed_pos = order.index(seed_rally)
    for direction in (1, -1):
        cursor = anchor
        cursor_side = anchor_side
        steps = order[seed_pos:] if direction == 1 else order[seed_pos::-1]
        for idx in steps:
            seg = rally_segment(tracks, by_index[idx], fps)
            found = _bridged_ids(chain, by_index[idx], by_index[seed_rally],
                                 fps, max_gap)
            if idx == seed_rally and seed_id in set(seg["track_id"]):
                ids[idx] = _expand_within_rally(seg, {seed_id})
                conf[idx] = 1.0
                method[idx] = "clicked"
            elif found:
                ids[idx] = _expand_within_rally(seg, found)
                conf[idx] = 0.9
                method[idx] = "bridged"
            else:
                picked, c = _nearest_candidate(seg, calib, cursor, cursor_side)
                if picked is not None:
                    ids[idx] = _expand_within_rally(seg, {picked})
                    conf[idx] = c
                    method[idx] = "proximity"
            if ids[idx]:
                centre = _track_court_centre(
                    seg[seg["track_id"].isin(ids[idx])], calib)
                if centre is not None:
                    cursor = centre
                    cursor_side = _side_of(centre)
            if idx == seed_rally:
                anchor, anchor_side = cursor, cursor_side
    return SubjectResolution(ids, conf, method)


def continuous_blocks(frames: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Split a frame sequence wherever it jumps by more than `max_gap`.

    A track id reappearing after the dead ball is not evidence that it is still
    the same person — trackers recycle ids, and a chain filtered by id alone
    looks continuous across a gap it never actually crossed. Only an unbroken
    run of frames proves identity survived, so bridging is decided on these
    blocks rather than on id membership.
    """
    if len(frames) == 0:
        return []
    f = np.sort(np.asarray(frames))
    breaks = np.flatnonzero(np.diff(f) > max_gap)
    starts = np.concatenate([[f[0]], f[breaks + 1]])
    ends = np.concatenate([f[breaks], [f[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _bridged_ids(chain, rally, seed_rally, fps: float,
                 max_gap: int) -> set[int]:
    """Ids from the chain that reached this rally WITHOUT crossing a break.

    Empty when tracking did not survive the dead ball between here and the
    rally the user clicked on, which is the honest answer and hands the rally
    to the proximity fallback (and then to the review UI).
    """
    if chain is None or chain.empty:
        return set()
    blocks = continuous_blocks(chain["frame"].to_numpy(), max_gap)
    seed_f0, seed_f1 = int(seed_rally.start * fps), int(seed_rally.end * fps)
    f0, f1 = int(rally.start * fps), int(rally.end * fps)

    def overlaps(block, a, b):
        return block[0] <= b and block[1] >= a

    live = [b for b in blocks if overlaps(b, seed_f0, seed_f1)]
    if not any(overlaps(b, f0, f1) for b in live):
        return set()
    inside = rally_segment(chain, rally, fps)
    return set(inside["track_id"].unique().tolist())


def _expand_within_rally(seg: pd.DataFrame, seed_ids: set[int]) -> set[int]:
    """Grow a set of ids into the full chain WITHIN one rally.

    Fragmentation inside a rally is a solved problem — `stitch_subject` handles
    it — and re-anchoring across rallies must not replace that.
    """
    out: set[int] = set()
    for sid in seed_ids:
        try:
            out |= stitch_chain_ids(seg, sid)
        except ValueError:
            out.add(sid)
    return out


def _nearest_candidate(seg: pd.DataFrame, calib: CourtCalibration,
                       anchor, anchor_side) -> tuple[int | None, float]:
    if anchor is None or seg.empty:
        return None, 0.0
    cands = _candidates(seg, calib, anchor_side)
    if not cands:
        return None, 0.0
    ranked = sorted(((float(np.hypot(*(c - anchor))), tid)
                     for tid, c in cands.items()))
    best_d, best_id = ranked[0]
    if best_d > MAX_REANCHOR_M:
        return None, 0.0
    if len(ranked) > 1:
        gap = ranked[1][0] - best_d
        # two players equally close: any pick would be a coin flip, and a coin
        # flip here credits someone else's touches to him
        if gap < MIN_MARGIN_M:
            return None, 0.0
        margin = min(1.0, gap / (2 * MIN_MARGIN_M))
    else:
        margin = 1.0
    closeness = 1.0 - min(best_d / MAX_REANCHOR_M, 1.0)
    # more plausible teammates on his half means less to go on, whatever the
    # nearest one happens to be
    crowding = 1.0 / (1.0 + 0.35 * (len(ranked) - 1))
    score = PROXIMITY_MAX_CONF * crowding * (0.5 * closeness + 0.5 * margin)
    return best_id, round(score, 3)


def resolve_subject_by_rally(tracks: pd.DataFrame, rallies, seed_id: int,
                             seed_rally: int, calib: CourtCalibration,
                             fps: float) -> dict[int, set[int]]:
    """Just the ids, for callers that do not need the confidence breakdown."""
    return resolve_subject(tracks, rallies, seed_id, seed_rally,
                           calib, fps).ids_by_rally


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
