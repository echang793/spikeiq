#!/usr/bin/env python
"""Turn review corrections into a labelled dataset.

Every rally the user confirmed or corrected is ground truth. This walks the
sessions on disk and emits one JSON line per labelled contact, carrying both
what the decoder said and what the truth is, plus the pose features behind the
call so the labels are useful for tuning and not just for scoring.

    .venv/bin/python scripts/export_labels.py > labels.jsonl

Only reviewed rallies are exported. An uncorrected contact in a rally nobody
looked at is not a correct prediction, it is an unchecked one, and mixing the
two would quietly inflate every accuracy number computed from this file.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review import Corrections  # noqa: E402


def session_labels(sdir: Path) -> list[dict]:
    plays_path = sdir / "plays.json"
    if not plays_path.exists():
        return []
    corrections = Corrections.load(sdir)
    reviewed = corrections.confirmed | set(corrections.actions)
    if not reviewed:
        return []

    plays = json.loads(plays_path.read_text())
    meta = _read(sdir / "meta.json")
    out = []
    for key, rally_plays in plays.items():
        idx = int(key)
        if idx not in reviewed or idx in corrections.deleted:
            continue
        for slot, p in enumerate(rally_plays):
            truth = p["action"]
            # `predicted` only exists where a correction overwrote the decoder;
            # everywhere else in a reviewed rally the decoder was right
            predicted = p.get("predicted", truth)
            out.append({
                "session": sdir.name,
                "label": meta.get("label", sdir.name),
                "rally": idx,
                "slot": slot,
                "t": p.get("t"),
                "predicted": predicted,
                "truth": truth,
                "correct": predicted == truth,
                "confidence": p.get("confidence"),
                "features": {k: p.get(k) for k in
                             ("side", "zone", "x", "y", "airborne",
                              "touch_index")},
            })
    return out


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data" / "sessions"),
                    help="sessions directory")
    ap.add_argument("--out", help="write here instead of stdout")
    args = ap.parse_args()

    rows = []
    for sdir in sorted(Path(args.data).glob("*")):
        if sdir.is_dir():
            rows.extend(session_labels(sdir))

    text = "\n".join(json.dumps(r) for r in rows)
    if args.out:
        Path(args.out).write_text(text + ("\n" if text else ""))
        print(f"{len(rows)} labelled contacts -> {args.out}", file=sys.stderr)
    else:
        print(text)
    if not rows:
        print("No reviewed rallies found. Confirm or correct some rallies in "
              "the review screen first.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
