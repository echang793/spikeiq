"""Whether this analysis is worth believing, and saying so loudly if not.

Every other module degrades gracefully: rates carry denominators, unknowns stay
unknown, thin samples get flagged. That is right, but it adds up to a report
that can look completely normal while resting on almost nothing — a level band,
a confidence bar and a list of coaching tips, all computed from four resolved
rallies out of ninety.

There is no volleyball footage to validate the pipeline against yet, so the
failure mode that matters most is not being wrong, it is being wrong *quietly*.
This module makes the pipeline state plainly when it did not work, and suppress
the level estimate entirely rather than let a number stand in for evidence.

Thresholds are judgement calls, not measurements. They are set where a coach
would stop trusting the numbers, and they are all in one place so they can be
revised once there is film to check them against.
"""

from dataclasses import dataclass, field

from textutil import rallies as rallies_word

MIN_RALLIES = 10               # below this it is a highlight clip, not a match
MIN_SUBJECT_RESOLVED = 0.60    # rallies where we know which player is him
MIN_ATTRIBUTION_RATE = 0.50    # audio contacts we could pin on a player
MIN_WINNERS_KNOWN = 0.50       # rallies with a side-out-derived result
MIN_TOUCHES = 15               # his own touches, the base of every skill line


@dataclass
class Check:
    name: str
    value: float
    threshold: float
    ok: bool
    message: str


@dataclass
class Quality:
    checks: list[Check] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def usable(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {
            "usable": self.usable,
            "stats": self.stats,
            "failures": [{"name": c.name, "value": round(c.value, 3),
                          "threshold": c.threshold, "message": c.message}
                         for c in self.failures],
            "checks": [{"name": c.name, "value": round(c.value, 3),
                        "threshold": c.threshold, "ok": c.ok}
                       for c in self.checks],
            "headline": self.headline(),
        }

    def headline(self) -> str:
        if self.usable:
            return ""
        worst = self.failures[0]
        extra = (f" (and {len(self.failures) - 1} other problem"
                 f"{'s' if len(self.failures) > 2 else ''})"
                 if len(self.failures) > 1 else "")
        return f"{worst.message}{extra}"


def _fraction(num: int, den: int) -> float:
    return num / den if den else 0.0


def assess(rallies, plays_by_rally: dict, subject_ids, audio_contacts: int,
           subject_touches: int) -> Quality:
    """Measure whether the pipeline actually got a grip on this match."""
    from metrics import ids_for

    n_rallies = len(rallies)
    resolved = sum(1 for r in rallies if ids_for(subject_ids, r.index))
    winners = sum(1 for r in rallies if r.winner)
    attributed = sum(len(plays_by_rally.get(r.index, [])) for r in rallies)

    stats = {
        "rallies": n_rallies,
        "rallies_with_subject": resolved,
        "rallies_with_winner": winners,
        "audio_contacts": audio_contacts,
        "attributed_contacts": attributed,
        "subject_touches": subject_touches,
    }

    checks = [
        Check("rally_count", n_rallies, MIN_RALLIES, n_rallies >= MIN_RALLIES,
              f"This recording yielded only {rallies_word(n_rallies)} — too "
              "short a sample to describe how you play. Check the audio picked "
              "up the whistle and the ball."),
        Check("subject_resolved", _fraction(resolved, n_rallies),
              MIN_SUBJECT_RESOLVED,
              _fraction(resolved, n_rallies) >= MIN_SUBJECT_RESOLVED,
              f"You were identified in only {resolved} of "
              f"{rallies_word(n_rallies)}, so most of this analysis is not "
              "about you. Confirm yourself in the rallies flagged for review."),
        Check("attribution_rate", _fraction(attributed, audio_contacts),
              MIN_ATTRIBUTION_RATE,
              _fraction(attributed, audio_contacts) >= MIN_ATTRIBUTION_RATE,
              f"Only {attributed} of {audio_contacts} ball contacts could be "
              "pinned on a player. Usually the camera is too far away or too "
              "much of the court is out of frame."),
        Check("winners_known", _fraction(winners, n_rallies), MIN_WINNERS_KNOWN,
              _fraction(winners, n_rallies) >= MIN_WINNERS_KNOWN,
              f"The winner could only be worked out for {winners} of "
              f"{rallies_word(n_rallies)}, so kills and errors are mostly "
              "unscored. This needs the server visible behind the endline."),
        Check("subject_touches", subject_touches, MIN_TOUCHES,
              subject_touches >= MIN_TOUCHES,
              f"Only {subject_touches} of your own touches were found. There is "
              "not enough here to grade any skill."),
    ]
    return Quality(checks=checks, stats=stats)


def gate_rating(rating: dict, quality: Quality) -> dict:
    """Strip the level estimate when the analysis cannot support one.

    The dimensions stay — they are still the best view of what was measured, and
    hiding them would make it harder to see what went wrong. What goes is the
    single number people actually read, because that is the part that would be
    believed.
    """
    if quality.usable:
        return rating
    gated = dict(rating)
    gated["level"] = None
    gated["band"] = None
    gated["confidence"] = 0.0
    gated["suppressed"] = True
    gated["note"] = ("No level is shown: " + quality.headline())
    return gated
