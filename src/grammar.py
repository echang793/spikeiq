"""Decode a rally's contact sequence into volleyball actions.

Classifying each contact on its own from pose is mediocre, because the two most
common touches — a forearm pass and a dig — look *identical*. The thing that
tells them apart is not the body, it is the context: you pass a serve and you
dig an attack. So the whole rally is decoded at once with Viterbi over a
transition model that encodes the rules and rhythms of the sport.

The hidden state is (action, touches used by the possessing side). Carrying the
touch count in the state lets the decoder apply the three-touch rule, including
the wrinkle that a block does not count as one of the three. Which side each
contact happened on is *observed*, not hidden, so possession changes are known
exactly and only the action label has to be inferred.

Illegal paths are penalised, never forbidden: the audio layer will sometimes
miss a soft touch, and a decoder that treated a four-touch sequence as
impossible would respond to one missed contact by mislabelling the whole rest of
the rally.
"""

import math
from dataclasses import dataclass

from contacts import ContactFeatures, behind_endline

ACTIONS = ("serve", "pass", "set", "attack", "block", "dig")

# Weights, not probabilities — normalised to log-space on load.
TRANSITIONS: dict[tuple[str, str], dict[str, float]] = {
    # opening touch of the rally
    ("start", "same"): {"serve": 0.95, "pass": 0.02, "set": 0.01,
                        "attack": 0.01, "block": 0.005, "dig": 0.005},
    # a serve crosses the net and is received
    ("serve", "cross"): {"pass": 0.74, "dig": 0.20, "attack": 0.02,
                         "set": 0.02, "block": 0.01, "serve": 0.01},
    ("serve", "same"): {"pass": 0.35, "dig": 0.35, "set": 0.20,
                        "attack": 0.05, "block": 0.03, "serve": 0.02},
    # first touch of a possession is almost always followed by a set
    ("pass", "same"): {"set": 0.78, "attack": 0.10, "pass": 0.08,
                       "dig": 0.02, "block": 0.01, "serve": 0.01},
    ("pass", "cross"): {"pass": 0.28, "dig": 0.28, "block": 0.22,
                        "attack": 0.18, "set": 0.03, "serve": 0.01},
    ("set", "same"): {"attack": 0.88, "set": 0.05, "pass": 0.04,
                      "dig": 0.02, "block": 0.005, "serve": 0.005},
    ("set", "cross"): {"block": 0.38, "dig": 0.30, "pass": 0.28,
                       "attack": 0.02, "set": 0.01, "serve": 0.01},
    # an attack is dug or blocked by the other side
    ("attack", "cross"): {"dig": 0.44, "block": 0.39, "pass": 0.14,
                          "set": 0.02, "attack": 0.005, "serve": 0.005},
    # ...or covered by the attacker's own side after a block deflection
    ("attack", "same"): {"dig": 0.40, "set": 0.28, "pass": 0.25,
                         "attack": 0.05, "block": 0.01, "serve": 0.01},
    ("block", "cross"): {"dig": 0.50, "pass": 0.20, "set": 0.19,
                         "attack": 0.10, "block": 0.005, "serve": 0.005},
    ("block", "same"): {"set": 0.40, "dig": 0.30, "attack": 0.20,
                        "pass": 0.09, "block": 0.005, "serve": 0.005},
    ("dig", "same"): {"set": 0.74, "attack": 0.11, "dig": 0.09,
                      "pass": 0.05, "block": 0.005, "serve": 0.005},
    ("dig", "cross"): {"dig": 0.30, "pass": 0.30, "block": 0.20,
                       "attack": 0.18, "set": 0.01, "serve": 0.01},
}

OVER_TOUCH_PENALTY = -4.0   # log-penalty per touch beyond the legal three
FLOOR = 1e-6


def _ramp(v: float, lo: float, hi: float) -> float:
    """Linear 0->1 ramp, clamped. Deliberately not a sigmoid: these thresholds
    are hand-set from how the sport looks, and a linear ramp keeps them
    readable and tunable."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def emission_scores(f: ContactFeatures) -> dict[str, float]:
    """Per-action plausibility in [0,1] from this contact's pose features.

    `pass` and `dig` intentionally score almost identically — the body does the
    same thing. Separating them is the transition model's job.
    """
    high_hands = _ramp(f.hand_height, 0.2, 1.0)
    low_hands = 1.0 - _ramp(f.hand_height, -0.4, 0.4)
    hands_together = 1.0 - _ramp(f.hand_spread, 0.4, 1.2)
    hands_apart = _ramp(f.hand_spread, 0.6, 1.5)
    airborne = _ramp(f.airborne, 0.2, 0.8)
    grounded = 1.0 - _ramp(f.airborne, 0.1, 0.5)
    fast = _ramp(f.hand_speed, 2.0, 8.0)
    loud = _ramp(f.strength, 0.4, 0.95)
    quiet = 1.0 - _ramp(f.strength, 0.3, 0.8)
    at_net = 1.0 - _ramp(f.net_dist, 0.6, 2.5)
    back_court = _ramp(f.net_dist, 2.5, 7.0)

    return {
        # the serve is defined by where it happens more than how it looks
        "serve": (0.60 * float(behind_endline(f)) + 0.20 * high_hands
                  + 0.20 * loud),
        "pass": (0.45 * low_hands + 0.20 * hands_together
                 + 0.20 * grounded + 0.15 * quiet),
        "set": (0.45 * high_hands + 0.25 * hands_together
                + 0.20 * grounded + 0.10 * quiet),
        "attack": (0.35 * airborne + 0.25 * fast + 0.25 * loud
                   + 0.15 * hands_apart),
        "block": (0.35 * airborne + 0.30 * at_net + 0.20 * high_hands
                  + 0.15 * quiet),
        "dig": (0.40 * low_hands + 0.25 * back_court
                + 0.20 * grounded + 0.15 * loud),
    }


@dataclass
class Play:
    t: float
    action: str
    track_id: int
    side: str
    zone: int | None
    x: float
    y: float
    airborne: float
    confidence: float
    touch_index: int          # touch number within the possession (blocks are 0)

    def as_dict(self) -> dict:
        return {
            "t": round(self.t, 3), "action": self.action,
            "track_id": self.track_id, "side": self.side, "zone": self.zone,
            "x": round(self.x, 2), "y": round(self.y, 2),
            "airborne": round(self.airborne, 3),
            "confidence": round(self.confidence, 3),
            "touch_index": self.touch_index,
        }


def _next_touches(prev_touches: int, action: str, same_side: bool) -> int:
    """Touch bookkeeping. A block is not one of a side's three touches, which
    is why it gets its own branch here rather than being folded into the count."""
    if action == "block":
        return 0 if not same_side else prev_touches
    return prev_touches + 1 if same_side else 1


def decode(features: list[ContactFeatures]) -> list[Play]:
    """Viterbi-decode one rally's contacts into labelled actions."""
    if not features:
        return []

    log_trans = {
        key: {a: math.log(max(w, FLOOR)) for a, w in row.items()}
        for key, row in TRANSITIONS.items()
    }
    log_em = [
        {a: math.log(max(s, FLOOR)) for a, s in emission_scores(f).items()}
        for f in features
    ]

    # state -> (score, backpointer). state = (action, touches)
    first_side = features[0].side
    paths: dict[tuple[str, int], float] = {}
    back: list[dict[tuple[str, int], tuple[str, int] | None]] = [{}]
    for a in ACTIONS:
        touches = _next_touches(0, a, same_side=True)
        state = (a, touches)
        paths[state] = log_trans[("start", "same")][a] + log_em[0][a]
        back[0][state] = None

    prev_side = first_side
    for i in range(1, len(features)):
        side = features[i].side
        same = side == prev_side
        rel = "same" if same else "cross"
        nxt: dict[tuple[str, int], float] = {}
        bp: dict[tuple[str, int], tuple[str, int] | None] = {}
        for (pa, pt), pscore in paths.items():
            row = log_trans.get((pa, rel))
            if row is None:
                continue
            for a in ACTIONS:
                touches = _next_touches(pt, a, same)
                score = pscore + row[a] + log_em[i][a]
                if touches > 3:
                    score += OVER_TOUCH_PENALTY * (touches - 3)
                state = (a, touches)
                if state not in nxt or score > nxt[state]:
                    nxt[state] = score
                    bp[state] = (pa, pt)
        if not nxt:  # no legal continuation; restart the chain from this contact
            for a in ACTIONS:
                state = (a, _next_touches(0, a, same_side=True))
                nxt[state] = log_em[i][a]
                bp[state] = None
        paths, _ = nxt, back.append(bp)
        prev_side = side

    state = max(paths, key=paths.get)
    chain: list[tuple[str, int]] = [state]
    for i in range(len(features) - 1, 0, -1):
        prev = back[i].get(chain[-1])
        if prev is None:
            break
        chain.append(prev)
    chain.reverse()
    while len(chain) < len(features):     # a restart truncated the backtrace
        chain.insert(0, ("pass", 1))

    plays: list[Play] = []
    for f, (action, touches), em in zip(features, chain, log_em):
        total = sum(math.exp(v) for v in em.values())
        share = math.exp(em[action]) / total if total > 0 else 0.0
        plays.append(Play(
            t=f.t, action=action, track_id=f.track_id, side=f.side, zone=f.zone,
            x=f.x, y=f.y, airborne=f.airborne,
            # the pose only gets a vote in proportion to how much of it was
            # actually visible — an occluded contact should not look certain
            confidence=round(share * f.confidence, 4),
            touch_index=0 if action == "block" else touches,
        ))
    return plays


def decode_rally(features: list[ContactFeatures]) -> list[dict]:
    return [p.as_dict() for p in decode(features)]
