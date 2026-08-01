"""Small helpers for user-facing copy.

Shared rather than duplicated because every report string is read by a person,
and "Only 1 rallies were found" undermines the sentence it appears in.
"""


def plural(n: int, singular: str, plural_form: str | None = None) -> str:
    """'1 rally' / '4 rallies'. Irregular plurals must be given explicitly."""
    word = singular if n == 1 else (plural_form or singular + "s")
    return f"{n} {word}"


def rallies(n: int) -> str:
    return plural(n, "rally", "rallies")
