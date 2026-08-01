"""The rally-by-rally review payload, and the corrections that come back.

The accuracy plan for this project was always "assisted": the pipeline proposes,
the user confirms. What makes that worth 15-20 minutes rather than an hour is
ordering — rallies are served worst-first, so attention lands where it changes
the answer instead of being spent nodding at rallies that were already right.

Corrections are facts, not proposals. Once a rally's subject or an action label
has been set by hand it is never re-derived, it survives re-analysis, and it
becomes a labelled example for `scripts/accuracy.py`.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

CORRECTIONS = "corrections.json"

# how much each unknown drags a rally up the review queue
WEIGHT_SUBJECT = 0.5
WEIGHT_ACTIONS = 0.3
WEIGHT_WINNER = 0.2


@dataclass
class Corrections:
    """User-supplied ground truth, keyed by rally index."""
    subject: dict[int, list[int]] = field(default_factory=dict)
    actions: dict[int, dict[int, str]] = field(default_factory=dict)
    deleted: set[int] = field(default_factory=set)
    # rallies the user actually looked at and accepted. Without this the
    # accuracy harness has no denominator: an uncorrected contact is either a
    # correct prediction or one nobody has checked, and those are very
    # different things.
    confirmed: set[int] = field(default_factory=set)

    @classmethod
    def load(cls, sdir: Path) -> "Corrections":
        path = sdir / CORRECTIONS
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError:
            return cls()
        return cls(
            subject={int(k): list(v) for k, v in (raw.get("subject") or {}).items()},
            actions={int(k): {int(i): a for i, a in v.items()}
                     for k, v in (raw.get("actions") or {}).items()},
            deleted={int(i) for i in (raw.get("deleted") or [])},
            confirmed={int(i) for i in (raw.get("confirmed") or [])},
        )

    def save(self, sdir: Path) -> None:
        (sdir / CORRECTIONS).write_text(json.dumps({
            "subject": {str(k): v for k, v in self.subject.items()},
            "actions": {str(k): {str(i): a for i, a in v.items()}
                        for k, v in self.actions.items()},
            "deleted": sorted(self.deleted),
            "confirmed": sorted(self.confirmed),
        }, indent=1))

    def merge(self, patch: dict) -> "Corrections":
        """Apply one review submission. Explicit nulls clear a correction, so a
        misclick can be undone rather than being permanent."""
        for key, ids in (patch.get("subject") or {}).items():
            idx = int(key)
            if ids is None:
                self.subject.pop(idx, None)
            else:
                self.subject[idx] = [int(i) for i in ids]
        for key, actions in (patch.get("actions") or {}).items():
            idx = int(key)
            slot = self.actions.setdefault(idx, {})
            for i, action in actions.items():
                if action is None:
                    slot.pop(int(i), None)
                else:
                    slot[int(i)] = action
            if not slot:
                self.actions.pop(idx, None)
        for idx, gone in (patch.get("deleted") or {}).items():
            (self.deleted.add if gone else self.deleted.discard)(int(idx))
        for idx, ok in (patch.get("confirmed") or {}).items():
            (self.confirmed.add if ok else self.confirmed.discard)(int(idx))
        return self

    def is_empty(self) -> bool:
        return not (self.subject or self.actions or self.deleted
                    or self.confirmed)


def apply_actions(plays: list[dict], corrections: Corrections,
                  rally_index: int) -> list[dict]:
    """Overlay hand-set action labels on a decoded rally.

    The decoder's original answer is kept in `predicted`, which is the whole
    basis of the accuracy harness: once the label is overwritten there is no
    other record of what the model actually said.
    """
    fixes = corrections.actions.get(rally_index)
    if not fixes:
        return plays
    out = []
    for i, p in enumerate(plays):
        if i in fixes:
            p = dict(p, action=fixes[i], predicted=p["action"],
                     confidence=1.0, corrected=True)
        out.append(p)
    return out


def review_payload(rallies, plays_by_rally: dict, subject: dict, tracks,
                   fps: float, corrections: Corrections) -> dict:
    """Everything the review screen needs, worst rally first."""
    items = []
    for rally in rallies:
        idx = rally.index
        ids = set(subject.get("ids_by_rally", {}).get(str(idx), []))
        subject_conf = float(subject.get("confidence_by_rally", {})
                             .get(str(idx), 0.0))
        method = subject.get("method_by_rally", {}).get(str(idx), "none")
        plays = plays_by_rally.get(idx, [])
        action_conf = min((p.get("confidence", 0.0) for p in plays), default=0.0)

        items.append({
            "index": idx,
            "start": round(rally.start, 2),
            "end": round(rally.end, 2),
            "set_index": rally.set_index,
            "thumb_t": round(_thumb_time(rally), 2),
            "serving_side": rally.serving_side,
            "winner": rally.winner,
            "deleted": idx in corrections.deleted,
            "confirmed": idx in corrections.confirmed,
            "subject": {
                "ids": sorted(ids),
                "confidence": round(subject_conf, 3),
                "method": method,
                "corrected": idx in corrections.subject,
            },
            "plays": [dict(p, corrected=bool(
                corrections.actions.get(idx, {}).get(i) is not None))
                for i, p in enumerate(plays)],
            "candidates": _candidate_boxes(tracks, rally, fps),
            "priority": round(_priority(
                subject_conf, action_conf, rally,
                idx in corrections.subject or idx in corrections.confirmed), 4),
            "why": _why(subject_conf, action_conf, rally, method),
        })
    items.sort(key=lambda i: i["priority"], reverse=True)
    return {"rallies": items, "corrections": not corrections.is_empty()}


def _thumb_time(rally) -> float:
    """A frame just after the serve: players are spread and mostly unoccluded,
    which is the easiest moment to recognise yourself."""
    return rally.contacts[0] + 0.3 if rally.contacts else rally.start + 0.5


def _priority(subject_conf: float, action_conf: float, rally,
              already_reviewed: bool) -> float:
    """How badly this rally needs a human.

    Rallies already dealt with — corrected or confirmed as correct — sink to
    the bottom. Being asked again about something you have answered is the
    fastest way to make a review queue feel endless.
    """
    if already_reviewed:
        return 0.0
    score = WEIGHT_SUBJECT * (1.0 - subject_conf)
    score += WEIGHT_ACTIONS * (1.0 - action_conf)
    if not rally.winner:
        score += WEIGHT_WINNER
    return score


UNSURE_SUBJECT = 0.5


def _why(subject_conf: float, action_conf: float, rally, method: str) -> str:
    """The single most useful reason to look at this rally.

    Ordered by what the user can actually fix. A subject guessed by proximity
    only leads if the guess was shaky — flagging every proximity rally as
    suspect when the pipeline was fairly sure would bury the rallies where
    something is genuinely missing.
    """
    if subject_conf <= 0.0:
        return "You were not identified in this rally."
    if method == "proximity" and subject_conf < UNSURE_SUBJECT:
        return "You were guessed from position — worth a look."
    if not rally.winner:
        return "No winner could be worked out for this rally."
    if action_conf < 0.3:
        return "The touches here were hard to read."
    return ""


def _candidate_boxes(tracks, rally, fps: float, limit: int = 14) -> list[dict]:
    """Every tracked player at the thumbnail moment, in image pixels.

    The review screen needs these so that clicking the right player on the
    thumbnail can be turned back into a track id.
    """
    if tracks is None or len(tracks) == 0:
        return []
    t = _thumb_time(rally)
    frame = int(t * fps)
    near = tracks[(tracks["frame"] >= frame - int(0.5 * fps))
                  & (tracks["frame"] <= frame + int(0.5 * fps))]
    if near.empty:
        return []
    out = []
    for tid, g in near.groupby("track_id"):
        row = g.iloc[(g["frame"] - frame).abs().to_numpy().argmin()]
        out.append({"track_id": int(tid),
                    "box": [round(float(row[c]), 1)
                            for c in ("x1", "y1", "x2", "y2")]})
    return out[:limit]
