"""Command resolver — precedence-based command lookup."""

from __future__ import annotations

from pathlib import Path

from flowcoder_flowchart import Command, load_command


class CommandNotFoundError(Exception):
    pass


def resolve_command(
    name: str,
    search_paths: list[str | Path] | None = None,
    priority_paths: list[str | Path] | None = None,
) -> Command:
    """Resolve a command by name using precedence-based search.

    Search order:
    1. Each path in priority_paths (path/<name>.json or path/commands/<name>.json)
    2. Current directory (commands/<name>.json)
    3. Each path in search_paths (path/<name>.json or path/commands/<name>.json)
    4. ~/.flowcoder/commands/<name>.json

    ``priority_paths`` is for a caller that means one *specific* flowchart, not
    "whichever <name>.json turns up first" -- a spawn block's ``search_path``
    aimed at one bundle's own flowchart, say.  It has to outrank the current
    directory as well as ``search_paths``, because a same-named file in either
    would otherwise win with no error raised: the wrong flowchart simply runs.
    It defaults to empty, so a caller that omits it sees the historical order.

    Raises CommandNotFoundError if not found.
    """
    candidates: list[Path] = []

    # 1. Priority paths -- ahead of cwd, see the docstring.
    if priority_paths:
        for pp in priority_paths:
            p = Path(pp)
            candidates.append(p / f"{name}.json")
            candidates.append(p / "commands" / f"{name}.json")

    # 2. Current directory
    candidates.append(Path.cwd() / "commands" / f"{name}.json")
    candidates.append(Path.cwd() / f"{name}.json")

    # 3. Search paths
    if search_paths:
        for sp in search_paths:
            p = Path(sp)
            candidates.append(p / f"{name}.json")
            candidates.append(p / "commands" / f"{name}.json")

    # 4. Home directory
    candidates.append(Path.home() / ".flowcoder" / "commands" / f"{name}.json")

    for candidate in candidates:
        if candidate.exists():
            return load_command(candidate)

    searched = [str(c.parent) for c in candidates]
    raise CommandNotFoundError(
        f"Command '{name}' not found. Searched: {', '.join(dict.fromkeys(searched))}"
    )
