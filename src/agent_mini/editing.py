"""How ``edit_file`` finds ``old_text`` in a file and puts ``new_text`` in its place.

The normal case is one exact match, in the first few lines of ``plan_edit``.

The rest is a small-model fix. Models often copy ``old_text`` without its
indentation, or with stray whitespace. When there is no exact match,
``fuzzy_replace`` matches line by line ignoring the whitespace around each line,
and ``_fit_indentation`` gives ``new_text`` the file's indentation (or refuses,
when it can't tell what that should be). When there is no match at all,
``closest_match`` shows the most similar region so the model can copy it
correctly on its next try.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

# closest_match: how similar a region must be to be shown, and the sizes past
# which the search would be too slow to bother.
_MIN_SIMILARITY = 0.6
_MAX_FILE_LINES = 5000
_MAX_OLD_LINES = 200


@dataclass(frozen=True)
class EditPlan:
    """An edit worked out but not written yet."""

    content: str  # the whole file after the edit, with \n line ends
    first_line: int  # 0-based index of the first changed line
    new_line_count: int  # how many lines the new text has
    fuzzy: bool  # matched ignoring whitespace
    crlf: bool  # the file uses \r\n line ends; write it back that way


def plan_edit(raw: str, old_text: str, new_text: str) -> EditPlan | str:
    """Work out an edit of a file whose text is *raw*: an ``EditPlan``, or an ``Error:`` string."""
    crlf = "\r\n" in raw
    content, old, new = (s.replace("\r\n", "\n") for s in (raw, old_text, new_text))

    count = content.count(old)
    if count == 1:
        start = content.index(old)
        return EditPlan(
            content=content[:start] + new + content[start + len(old):],
            first_line=content.count("\n", 0, start),
            new_line_count=new.count("\n") + 1,
            fuzzy=False,
            crlf=crlf,
        )
    if count > 1:
        return _matches_many(count)

    # Small-model fix: no exact match, so try again ignoring whitespace around each line.
    match = fuzzy_replace(content, old, new)
    if isinstance(match, str):
        return match
    if isinstance(match, int):
        if match > 1:
            return _matches_many(match)
        return ("Error: old_text was not found. Copy it exactly from read_file output, "
                "including indentation." + closest_match(content, old))
    new_content, first_line, new_line_count = match
    return EditPlan(new_content, first_line, new_line_count, fuzzy=True, crlf=crlf)


def _matches_many(count: int) -> str:
    return f"Error: old_text matches {count} places. Include more surrounding lines so it matches exactly once."


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _base_indent(lines: list[str]) -> str:
    """The shortest indentation among the non-blank lines."""
    return min((_indent_of(ln) for ln in lines if ln.strip()), key=len, default="")


def fuzzy_replace(content: str, old: str, new: str) -> tuple[str, int, int] | int | str:
    """Replace *old*, matched line by line ignoring the whitespace around each line.

    Returns ``(new_content, first_line, new_line_count)`` on a unique match,
    otherwise the number of matches. *new* is re-indented to the file's
    indentation; when that can't be done safely, the result is an ``Error:``
    string showing the matched lines, so the model can copy them exactly.
    """
    lines = content.split("\n")
    old_lines = old.strip("\n").split("\n")
    key = [ln.strip() for ln in old_lines]
    if not any(key):
        return 0
    stripped = [ln.strip() for ln in lines]
    n = len(key)
    hits = [i for i in range(len(lines) - n + 1) if stripped[i : i + n] == key]
    if len(hits) != 1:
        return len(hits)

    i = hits[0]
    new_lines = new.strip("\n").split("\n") if new.strip("\n") else []
    fitted = _fit_indentation(lines[i : i + n], old_lines, new_lines)
    if fitted is None:
        region = "\n".join(lines[i : i + n])
        return (f"Error: old_text matches lines {i + 1}-{i + n} only when indentation is ignored, and the "
                "indentation of new_text can't be fixed safely. Copy old_text exactly, including indentation, "
                f"and indent new_text to match. The lines are:\n{region}")
    return "\n".join(lines[:i] + fitted + lines[i + n :]), i, len(fitted)


def _fit_indentation(file_lines: list[str], old_lines: list[str], new_lines: list[str]) -> list[str] | None:
    """*new_lines* indented to fit the file, given that *old_lines* matched *file_lines* ignoring
    whitespace. None when there is no safe way to tell what the indentation should be.
    """
    pairs = [(_indent_of(o), _indent_of(f)) for o, f in zip(old_lines, file_lines) if o.strip()]
    if all(o == f for o, f in pairs):
        return new_lines  # only whitespace at line ends differed

    # Line for line: new_text is indented exactly like old_text (the same mistake in
    # both), so each new line takes the indentation of the file line it replaces.
    if len(new_lines) == len(old_lines) and all(
        _indent_of(nl) == _indent_of(ol) for nl, ol in zip(new_lines, old_lines) if nl.strip()
    ):
        return [_indent_of(fl) + nl.lstrip() if nl.strip() else nl for nl, fl in zip(new_lines, file_lines)]

    # new_text is indented differently from old_text: the model chose its indentation,
    # so trust it as written (a stray tab in old_text must not strip the tabs a Makefile needs).
    old_base, file_base = _base_indent(old_lines), _base_indent(file_lines)
    if _base_indent(new_lines) != old_base:
        return new_lines

    # The whole block was copied shifted (usually without its indentation): shift new_text back.
    shifted = all(o.startswith(old_base) and f.startswith(file_base) and o[len(old_base):] == f[len(file_base):]
                  for o, f in pairs)
    if shifted and all(nl.startswith(old_base) for nl in new_lines if nl.strip()):
        return [file_base + nl[len(old_base):] if nl.strip() else nl for nl in new_lines]

    # Some lines are off by more than others, and new_text has lines old_text doesn't.
    # Guessing here silently breaks code (or a Makefile's tabs); asking costs one retry.
    return None


def closest_match(content: str, old: str) -> str:
    """The region of *content* most like *old*, as a hint for the model; '' if none is close."""
    lines = content.split("\n")
    old_lines = old.strip("\n").split("\n")
    n = len(old_lines)
    if len(lines) > _MAX_FILE_LINES or n > _MAX_OLD_LINES:
        return ""
    target = "\n".join(ln.strip() for ln in old_lines)

    best, best_i = 0.0, -1
    for i in range(max(1, len(lines) - n + 1)):
        window = "\n".join(ln.strip() for ln in lines[i : i + n])
        m = difflib.SequenceMatcher(None, window, target)
        # The quick ratios are cheap upper bounds: skip the slow ratio() when they can't beat the best.
        if m.real_quick_ratio() <= best or m.quick_ratio() <= best:
            continue
        ratio = m.ratio()
        if ratio > best:
            best, best_i = ratio, i

    if best < _MIN_SIMILARITY:
        return ""
    region = "\n".join(lines[best_i : best_i + n])
    return f"\nClosest match (line {best_i + 1}, {best:.0%} similar):\n{region}"
