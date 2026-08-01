#!/usr/bin/env python
"""Score the action decoder against reviewed labels.

This is the harness for the project's standing accuracy gate: at least 80 %
on 50 hand-labelled contacts before the per-skill numbers downstream mean
anything.

    .venv/bin/python scripts/export_labels.py --out labels.jsonl
    .venv/bin/python scripts/accuracy.py labels.jsonl

Prints overall accuracy, a per-action breakdown, and a confusion matrix. The
per-action view is the one that matters: an overall figure carried by the two
common actions can hide a decoder that never once gets a block right.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ACTIONS = ["serve", "pass", "set", "attack", "block", "dig"]
GATE_ACCURACY = 0.80
GATE_SAMPLE = 50


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def confusion(rows: list[dict]) -> dict:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        matrix[r["truth"]][r["predicted"]] += 1
    return matrix


def per_action(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for action in ACTIONS:
        truth_rows = [r for r in rows if r["truth"] == action]
        pred_rows = [r for r in rows if r["predicted"] == action]
        hits = sum(1 for r in truth_rows if r["correct"])
        out[action] = {
            "support": len(truth_rows),
            "recall": hits / len(truth_rows) if truth_rows else None,
            "precision": hits / len(pred_rows) if pred_rows else None,
        }
    return out


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v * 100:5.1f}%"


def report(rows: list[dict]) -> bool:
    if not rows:
        print("No labels. Run export_labels.py after reviewing some rallies.")
        return False

    correct = sum(1 for r in rows if r["correct"])
    accuracy = correct / len(rows)
    print(f"\n{len(rows)} labelled contacts from "
          f"{len({r['session'] for r in rows})} session(s)")
    print(f"overall accuracy: {accuracy * 100:.1f}%  ({correct}/{len(rows)})\n")

    print(f"{'action':<10}{'support':>8}{'recall':>9}{'precision':>11}")
    for action, stats in per_action(rows).items():
        print(f"{action:<10}{stats['support']:>8}"
              f"{_fmt(stats['recall']):>9}{_fmt(stats['precision']):>11}")

    print("\nconfusion (rows = truth, columns = predicted)")
    matrix = confusion(rows)
    print(f"{'':<10}" + "".join(f"{a[:5]:>7}" for a in ACTIONS))
    for truth in ACTIONS:
        row = matrix.get(truth, Counter())
        print(f"{truth:<10}" + "".join(f"{row.get(p, 0):>7}" for p in ACTIONS))

    print()
    if len(rows) < GATE_SAMPLE:
        print(f"GATE NOT MET: {len(rows)} labels, need {GATE_SAMPLE}. "
              "Review more rallies before drawing conclusions.")
        return False
    if accuracy < GATE_ACCURACY:
        print(f"GATE FAILED: {accuracy * 100:.1f}% is below the "
              f"{GATE_ACCURACY * 100:.0f}% needed. Tune the transition priors "
              "in grammar.TRANSITIONS or the feature ramps in "
              "grammar.emission_scores before trusting the skill numbers.")
        return False
    print(f"GATE PASSED: {accuracy * 100:.1f}% on {len(rows)} labels.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", nargs="?", default="labels.jsonl")
    args = ap.parse_args()
    path = Path(args.labels)
    if not path.exists():
        print(f"{path} not found — run scripts/export_labels.py first",
              file=sys.stderr)
        return 2
    return 0 if report(load(path)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
