"""Subject identity across a whole match.

These tests use `make_windowed_tracks`, which produces the frame layout the real
pipeline writes: rows only inside rally windows, with hundreds of dead-ball
frames between them. The first version of this project resolved the subject once
for the whole match with a 45-frame stitching gap, which silently truncated him
to rally one and computed the entire report from it.
"""

import pytest

from tracking import resolve_subject_by_rally


def moving_subject(rallies, me=1, opponent=9):
    """The subject drifts around his own half between rallies; an opponent sits
    on the far side throughout."""
    spots = [(3.0, 13.0), (6.0, 12.0), (2.5, 15.0), (7.0, 11.0)]
    return {r.index: {me: spots[i % len(spots)], opponent: (4.5, 4.0)}
            for i, r in enumerate(rallies)}


def test_subject_is_resolved_in_every_rally(calib, make_windowed_tracks,
                                            match_rallies):
    """The regression test for the bug that invalidated every real match: with
    tracking restricted to rally windows, the subject must still be found in
    rally four, not just rally one."""
    tracks = make_windowed_tracks(match_rallies, moving_subject(match_rallies))
    got = resolve_subject_by_rally(tracks, match_rallies, seed_id=1,
                                   seed_rally=0, calib=calib, fps=30.0)
    assert set(got) == {r.index for r in match_rallies}
    assert all(ids for ids in got.values()), f"unresolved rallies: {got}"


def test_subject_follows_tracker_reid_between_rallies(calib, make_windowed_tracks,
                                                      match_rallies):
    """Resetting the tracker per rally means the subject gets a brand new track
    id every rally. Re-anchoring has to bridge that."""
    spots = [(3.0, 13.0), (3.4, 12.6), (3.1, 13.3), (3.6, 12.9)]
    ids = [1, 20, 41, 62]
    positions = {r.index: {ids[i]: spots[i], 90 + i: (4.5, 4.0)}
                 for i, r in enumerate(match_rallies)}
    tracks = make_windowed_tracks(match_rallies, positions)
    got = resolve_subject_by_rally(tracks, match_rallies, seed_id=1,
                                   seed_rally=0, calib=calib, fps=30.0)
    for i, r in enumerate(match_rallies):
        assert ids[i] in got[r.index], f"rally {r.index}: got {got[r.index]}"


def test_subject_is_never_matched_to_the_other_side_of_the_net(
        calib, make_windowed_tracks, match_rallies):
    """Players do not change ends mid-set. An opponent must never be adopted,
    however close the score."""
    positions = {r.index: {1: (4.5, 13.0)} for r in match_rallies}
    positions[2] = {77: (4.5, 5.0)}          # only an opponent visible
    tracks = make_windowed_tracks(match_rallies, positions)
    got = resolve_subject_by_rally(tracks, match_rallies, seed_id=1,
                                   seed_rally=0, calib=calib, fps=30.0)
    assert got[2] == set(), "adopted a player from the opposite half"


def test_unresolvable_rally_returns_empty_not_a_guess(calib, make_windowed_tracks,
                                                      match_rallies):
    """When he is subbed out or simply not tracked, the honest answer is
    nothing — the review UI asks about it. A guess would quietly credit another
    player's touches to him."""
    positions = moving_subject(match_rallies)
    positions[1] = {}                         # nobody tracked at all
    tracks = make_windowed_tracks(match_rallies, positions)
    got = resolve_subject_by_rally(tracks, match_rallies, seed_id=1,
                                   seed_rally=0, calib=calib, fps=30.0)
    assert got[1] == set()
    assert got[2], "a single missing rally must not end the chain"


def test_a_gap_does_not_terminate_the_chain(calib, make_windowed_tracks,
                                            match_rallies):
    """Re-anchoring must keep working from the last rally he WAS seen in,
    rather than giving up the first time it loses him."""
    positions = moving_subject(match_rallies)
    positions[1] = {}
    positions[2] = {}
    tracks = make_windowed_tracks(match_rallies, positions)
    got = resolve_subject_by_rally(tracks, match_rallies, seed_id=1,
                                   seed_rally=0, calib=calib, fps=30.0)
    assert got[3], "chain died at the first gap instead of recovering"


def test_seed_rally_can_be_anywhere_in_the_match(calib, make_windowed_tracks,
                                                 match_rallies):
    """The user clicks himself on one frame; that frame may be mid-match, so
    re-anchoring has to run backwards as well as forwards."""
    positions = moving_subject(match_rallies)
    tracks = make_windowed_tracks(match_rallies, positions)
    got = resolve_subject_by_rally(tracks, match_rallies, seed_id=1,
                                   seed_rally=2, calib=calib, fps=30.0)
    assert all(got[r.index] for r in match_rallies)


def test_within_a_rally_the_existing_stitching_still_applies(
        calib, make_windowed_tracks, match_rallies):
    """Tracker fragmentation inside a rally is a different problem with a
    working solution; re-anchoring must not replace it."""
    import pandas as pd
    from conftest import px_for, tracks_frame
    from tracking import COLUMNS

    rally = match_rallies[0]
    px, py = px_for(calib, 3.0, 13.0)
    rows = []
    for i, frame in enumerate(range(300, 541, 2)):
        tid = 1 if i < 40 else 33          # re-id part way through the rally
        rows.append(tracks_frame(frame, tid, float(px), float(py) - 60.0, h=120.0))
    tracks = pd.DataFrame(rows, columns=COLUMNS)
    got = resolve_subject_by_rally(tracks, [rally], seed_id=1, seed_rally=0,
                                   calib=calib, fps=30.0)
    assert got[rally.index] == {1, 33}


def test_bridging_resolves_identity_that_proximity_cannot(calib, match_rallies):
    """Six teammates in the same jersey stand around during the dead ball. With
    only rally frames, proximity is a coin flip and correctly refuses to answer.
    With the sparse between-rally frames, the ordinary stitch carries identity
    straight through and every rally is resolved.
    """
    import pandas as pd
    from conftest import px_for, tracks_frame
    from tracking import COLUMNS, resolve_subject

    def rows_at(frame, positions):
        out = []
        for tid, (cx, cy) in positions.items():
            px, py = px_for(calib, cx, cy)
            out.append(tracks_frame(frame, tid, float(px), float(py) - 60.0,
                                    h=120.0))
        return out

    # a cluster of teammates who all drift; the subject is id 1
    def team_at(phase):
        return {1: (2.0 + phase, 12.0 + phase),
                2: (3.2 + phase, 12.4 + phase),
                3: (4.4 + phase, 12.8 + phase)}

    rally_rows, bridged_rows = [], []
    for r in match_rallies:
        phase = r.index * 1.2
        for frame in range(int(r.start * 30), int(r.end * 30) + 1, 2):
            rally_rows += rows_at(frame, team_at(phase))
    # the same match, plus 2 fps of dead-ball samples linking the rallies
    bridged_rows = list(rally_rows)
    for a, b in zip(match_rallies, match_rallies[1:]):
        for frame in range(int(a.end * 30), int(b.start * 30) + 1, 15):
            t = (frame / 30 - a.end) / max(b.start - a.end, 1e-6)
            bridged_rows += rows_at(frame,
                                    team_at(a.index * 1.2 + 1.2 * t))

    rally_only = resolve_subject(pd.DataFrame(rally_rows, columns=COLUMNS),
                                 match_rallies, 1, 0, calib, 30.0)
    bridged = resolve_subject(pd.DataFrame(bridged_rows, columns=COLUMNS),
                              match_rallies, 1, 0, calib, 30.0)

    assert bridged.resolved_fraction == 1.0
    assert all(m in ("bridged", "clicked")
               for m in bridged.method_by_rally.values())
    for r in match_rallies:
        assert 1 in bridged.ids_by_rally[r.index]
    # without the dead-ball samples the same match either guesses or gives up,
    # and in both cases says so rather than reporting bridged certainty
    assert all(m in ("proximity", "none")
               for i, m in rally_only.method_by_rally.items() if i != 0)
    later = [c for i, c in rally_only.confidence_by_rally.items() if i != 0]
    assert max(later) < min(bridged.confidence_by_rally[r.index]
                            for r in match_rallies[1:])
    assert rally_only.resolved_fraction < 1.0


def test_proximity_refuses_when_teammates_are_bunched(calib, match_rallies):
    """Two teammates standing shoulder to shoulder during the dead ball. Any
    pick is a coin flip, and a coin flip credits someone else's kills to him —
    so the honest output is nothing, and bridging is what actually solves it."""
    import pandas as pd
    from conftest import px_for, tracks_frame
    from tracking import COLUMNS, resolve_subject

    rows = []
    for r in match_rallies:
        for frame in range(int(r.start * 30), int(r.end * 30) + 1, 2):
            for tid, (cx, cy) in {1: (4.0, 13.0), 2: (4.3, 13.2)}.items():
                px, py = px_for(calib, cx, cy)
                rows.append(tracks_frame(frame, tid, float(px), float(py) - 60.0,
                                         h=120.0))
    res = resolve_subject(pd.DataFrame(rows, columns=COLUMNS), match_rallies,
                          1, 0, calib, 30.0)
    assert res.ids_by_rally[0] == {1}          # the rally he was clicked in
    assert all(res.ids_by_rally[r.index] == set() for r in match_rallies[1:])


def test_a_positional_swap_is_never_reported_confidently(calib, match_rallies):
    """Two teammates trade places during the dead ball. Position alone cannot
    tell "he stayed put" from "they swapped", so proximity here is not merely
    uncertain, it is wrong — and the one thing it must never be is confident."""
    import pandas as pd
    from conftest import px_for, tracks_frame
    from tracking import COLUMNS, resolve_subject

    layouts = [{1: (2.0, 12.0), 2: (7.0, 15.0)},
               {2: (2.2, 12.2), 1: (6.8, 14.8)}]   # swapped
    rows = []
    for r in match_rallies:
        for frame in range(int(r.start * 30), int(r.end * 30) + 1, 2):
            for tid, (cx, cy) in layouts[r.index % 2].items():
                px, py = px_for(calib, cx, cy)
                rows.append(tracks_frame(frame, tid, float(px), float(py) - 60.0,
                                         h=120.0))
    res = resolve_subject(pd.DataFrame(rows, columns=COLUMNS), match_rallies,
                          1, 0, calib, 30.0)
    guessed = [res.confidence_by_rally[r.index] for r in match_rallies[1:]]
    assert all(c < 0.5 for c in guessed), guessed


def test_confidence_marks_the_guessed_rallies(calib, make_windowed_tracks,
                                              match_rallies):
    """Proximity-resolved rallies must not look as trustworthy as bridged ones —
    the review UI sorts on exactly this."""
    from tracking import resolve_subject
    tracks = make_windowed_tracks(match_rallies, moving_subject(match_rallies))
    res = resolve_subject(tracks, match_rallies, 1, 0, calib, 30.0)
    assert res.confidence_by_rally[0] == 1.0          # the clicked rally
    guessed = [res.confidence_by_rally[r.index] for r in match_rallies[1:]]
    assert all(0.0 < c < 0.9 for c in guessed), guessed


@pytest.mark.parametrize("seed", [1, 20])
def test_resolution_is_stable_whichever_id_seeds_it(calib, make_windowed_tracks,
                                                    match_rallies, seed):
    spots = [(3.0, 13.0), (3.4, 12.6), (3.1, 13.3), (3.6, 12.9)]
    ids = [1, 20, 41, 62]
    positions = {r.index: {ids[i]: spots[i]} for i, r in enumerate(match_rallies)}
    tracks = make_windowed_tracks(match_rallies, positions)
    seed_rally = ids.index(seed)
    got = resolve_subject_by_rally(tracks, match_rallies, seed_id=seed,
                                   seed_rally=seed_rally, calib=calib, fps=30.0)
    assert all(got[r.index] for r in match_rallies)
