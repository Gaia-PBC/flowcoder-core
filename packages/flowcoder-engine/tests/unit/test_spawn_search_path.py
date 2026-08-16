"""Per-spawn ``search_path`` aims one spawn at a specific flowchart.

A ``SpawnBlock`` may set ``search_path`` (templated); ``_exec_spawn`` resolves
it and hands it to ``resolve_command`` as a *priority* path — one that outranks
both the current working directory and the walker's inherited ``search_paths``.

The precedence is the whole point, not a detail.  ``resolve_command`` checks cwd
first, so a same-named ``commands/<name>.json`` under cwd, or an earlier entry
in the inherited ``search_paths``, would otherwise win *silently*: no error, the
wrong flowchart just runs.  The two ``TestSearchPathPrecedence`` cases below are
regressions for exactly those two collisions, on real directories.

The priority is *inherited* as a priority, not flattened into the child's
ordinary ``search_paths``: a spawned child carries it into its own ``command``
*and* ``spawn`` blocks, so a bundle's entry flowchart that delegates to a
sibling still gets that bundle's copy rather than a cwd shadow one level down.
Flattening would leave the child cwd-shadowable at depth 1 -- the bundle path
would sit in ``search_paths``, which ``resolve_command`` checks *after* cwd.

That is not hypothetical.  A live bundle's ``soul-lite.json`` holds a spawn of
``soul-lite-do`` with no ``search_path`` of its own, while a same-named
``soul-lite-do.json`` sits in the engine's working directory.  The two are
byte-identical until a mutation edits the bundle's copy -- at which point a
flattened child would evaluate the *unmutated* production file and score it as
if the edit had taken effect, silently.  Hence the deliberately differing
contents below: identical fixtures would pass under either design and prove
nothing.  ``test_priority_is_inherited_by_a_nested_spawn`` reproduces it.

Per-spawn ``cwd`` is not an alternative fix: ``resolve_command`` reads
``Path.cwd()`` (the engine process), ``_exec_spawn`` only hands ``cwd`` to
``clone()``, and the engine calls ``os.chdir`` nowhere -- so it moves the child
session without touching resolution at all.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import flowcoder_engine.walker as walker_mod
from flowcoder_engine.walker import GraphWalker
from flowcoder_flowchart import (
    BashBlock,
    Command,
    CommandBlock,
    Connection,
    EndBlock,
    Flowchart,
    SpawnBlock,
    StartBlock,
    WaitBlock,
    save_command,
)

from tests.conftest import MockProtocol, MockSession


def _leaf_flowchart(name: str) -> Flowchart:
    """A do-nothing child flowchart, identifiable by its ``name``."""
    return Flowchart(
        name=name,
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[Connection(source_id="cs", target_id="ce")],
    )


def _nesting_flowchart(name: str, nested: str) -> Flowchart:
    """A child flowchart that invokes ``nested`` through a ``command`` block."""
    return Flowchart(
        name=name,
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "call": CommandBlock(id="call", name="Call", command_name=nested),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[
            Connection(source_id="cs", target_id="call"),
            Connection(source_id="call", target_id="ce"),
        ],
    )


def _marking_flowchart(name: str, marker: Path) -> Flowchart:
    """A flowchart whose one effect is appending ``name`` to ``marker``.

    Asserting on a side effect rather than on which file got loaded is what
    makes "the bundle's copy ran" a claim about execution.  The two copies in
    the nested-spawn test differ only in what they write, mirroring a bundle
    whose flowchart has been mutated away from the production original.
    """
    return Flowchart(
        name=name,
        blocks={
            "cs": StartBlock(id="cs", name="ChildStart"),
            "mark": BashBlock(id="mark", name="Mark", command=f"echo {name} >> {marker}"),
            "ce": EndBlock(id="ce", name="ChildEnd"),
        },
        connections=[
            Connection(source_id="cs", target_id="mark"),
            Connection(source_id="mark", target_id="ce"),
        ],
    )


def _spawn_wait_flowchart(spawn: SpawnBlock, wait: WaitBlock) -> Flowchart:
    return Flowchart(
        blocks={
            "s": StartBlock(id="s", name="Start"),
            "spawn": spawn,
            "wait": wait,
            "e": EndBlock(id="e", name="End"),
        },
        connections=[
            Connection(source_id="s", target_id="spawn"),
            Connection(source_id="spawn", target_id="wait"),
            Connection(source_id="wait", target_id="e"),
        ],
    )


def _spawning(name: str, **kwargs) -> Flowchart:
    """The standard spawn-then-wait parent used by every test here."""
    spawn = SpawnBlock(
        id="spawn",
        name="Spawn",
        agent_name="cell-1",
        command_name=name,
        **kwargs,
    )
    wait = WaitBlock(id="wait", name="Wait", wait_for=["cell-1"])
    return _spawn_wait_flowchart(spawn, wait)


def _write_command(directory: Path, name: str, flowchart: Flowchart) -> Path:
    """Write ``<directory>/<name>.json`` so ``resolve_command`` can find it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    save_command(Command(name=name, flowchart=flowchart), path)
    return path


def _recording_walker() -> tuple[type[GraphWalker], list[GraphWalker]]:
    """A GraphWalker subclass that appends every instance it constructs.

    Patched over ``flowcoder_engine.walker.GraphWalker`` it captures the child
    walkers the engine builds for spawns and sub-commands, which is where the
    inherited priority paths and the resolved flowchart can be inspected.
    """
    made: list[GraphWalker] = []

    class _Recording(GraphWalker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    return _Recording, made


class TestSpawnSearchPathWiring:
    """Field -> templated -> resolve_command -> child walker."""

    def test_field_defaults_to_none(self):
        block = SpawnBlock(id="spawn", name="Spawn")
        assert block.search_path is None

    async def test_search_path_is_templated_and_reaches_resolve_command(
        self, tmp_path
    ):
        fc = _spawning("child-cmd", search_path="{{base}}/bundle-$1")
        walker = GraphWalker(
            fc,
            MockSession(),
            {"base": str(tmp_path), "$1": "7"},
            MockProtocol(),
        )

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _leaf_flowchart("child")
        with patch(
            "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
        ) as resolve:
            result = await walker.run()

        assert result.status == "completed", result
        # Both {{base}} and $1 resolved, and the result arrived as a priority
        # path -- ahead of cwd, which is what makes the spawn's choice stick.
        assert resolve.call_args.kwargs["priority_paths"] == [
            f"{tmp_path}/bundle-7"
        ]

    async def test_search_path_reaches_the_child_walker(self, tmp_path):
        fc = _spawning("child-cmd", search_path=str(tmp_path / "bundle"))
        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(), search_paths=["/inherited"]
        )

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _leaf_flowchart("child")
        recording, made = _recording_walker()
        with (
            patch("flowcoder_engine.walker.resolve_command", return_value=mock_cmd),
            patch.object(walker_mod, "GraphWalker", recording),
        ):
            result = await walker.run()

        assert result.status == "completed", result
        assert len(made) == 1
        child = made[0]
        assert child._priority_paths == [str(tmp_path / "bundle")]
        # The inherited search paths are untouched -- the spawn adds a
        # higher-precedence path, it does not replace the walker's own.
        assert child._search_paths == ["/inherited"]

    async def test_no_search_path_leaves_resolution_unchanged(self):
        fc = _spawning("child-cmd")
        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(), search_paths=["/inherited"]
        )

        mock_cmd = MagicMock()
        mock_cmd.flowchart = _leaf_flowchart("child")
        recording, made = _recording_walker()
        with (
            patch(
                "flowcoder_engine.walker.resolve_command", return_value=mock_cmd
            ) as resolve,
            patch.object(walker_mod, "GraphWalker", recording),
        ):
            result = await walker.run()

        assert result.status == "completed", result
        # No search_path -> no priority paths -> historical precedence.
        assert resolve.call_args.kwargs["priority_paths"] == []
        assert resolve.call_args.kwargs["search_paths"] == ["/inherited"]
        assert made[0]._priority_paths == []


class TestSearchPathPrecedence:
    """Real directories, real ``resolve_command`` -- the collisions that bite."""

    async def test_search_path_beats_a_colliding_command_under_cwd(
        self, tmp_path, monkeypatch
    ):
        # The exact shadowing case: cwd has commands/<name>.json, which
        # resolve_command checks before anything else.
        cwd = tmp_path / "cwd"
        _write_command(cwd / "commands", "child-cmd", _leaf_flowchart("from-cwd"))
        bundle = tmp_path / "bundle"
        _write_command(bundle, "child-cmd", _leaf_flowchart("from-bundle"))
        monkeypatch.chdir(cwd)

        fc = _spawning("child-cmd", search_path=str(bundle))
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())

        recording, made = _recording_walker()
        with patch.object(walker_mod, "GraphWalker", recording):
            result = await walker.run()

        assert result.status == "completed", result
        assert [w._flowchart.name for w in made] == ["from-bundle"]

    async def test_search_path_beats_an_earlier_inherited_search_path(
        self, tmp_path
    ):
        # The ordering case: the inherited path already holds a flowchart of the
        # same name, so appending rather than prepending would resolve it.
        seed = tmp_path / "seed"
        _write_command(seed, "orchestrator", _leaf_flowchart("from-seed"))
        bundle = tmp_path / "bundle"
        _write_command(bundle, "orchestrator", _leaf_flowchart("from-bundle"))

        fc = _spawning("orchestrator", search_path=str(bundle))
        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(), search_paths=[str(seed)]
        )

        recording, made = _recording_walker()
        with patch.object(walker_mod, "GraphWalker", recording):
            result = await walker.run()

        assert result.status == "completed", result
        assert [w._flowchart.name for w in made] == ["from-bundle"]

    async def test_priority_is_inherited_by_nested_commands(
        self, tmp_path, monkeypatch
    ):
        # The recorded decision: the child carries the priority into its own
        # command blocks, so a cwd shadow cannot capture a nested sub-command.
        cwd = tmp_path / "cwd"
        _write_command(cwd / "commands", "nested-cmd", _leaf_flowchart("from-cwd"))
        bundle = tmp_path / "bundle"
        _write_command(
            bundle, "orchestrator", _nesting_flowchart("bundle-orch", "nested-cmd")
        )
        _write_command(bundle, "nested-cmd", _leaf_flowchart("from-bundle"))
        monkeypatch.chdir(cwd)

        fc = _spawning("orchestrator", search_path=str(bundle))
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())

        recording, made = _recording_walker()
        with patch.object(walker_mod, "GraphWalker", recording):
            result = await walker.run()

        assert result.status == "completed", result
        assert sorted(w._flowchart.name for w in made) == [
            "bundle-orch",
            "from-bundle",
        ]

    async def test_priority_is_inherited_by_a_nested_spawn(
        self, tmp_path, monkeypatch
    ):
        # The live shape: the bundle's entry flowchart spawns a sibling without
        # naming a search path itself, so depth 1 resolves purely on what the
        # child inherited.  Flattening the bundle path into the child's
        # search_paths puts it *behind* cwd there, and the decoy runs.
        marker = tmp_path / "which-ran.txt"
        cwd = tmp_path / "cwd"
        _write_command(
            cwd / "commands", "entry", _marking_flowchart("DECOY-ENTRY", marker)
        )
        _write_command(
            cwd / "commands", "nested", _marking_flowchart("DECOY-NESTED", marker)
        )

        bundle = tmp_path / "bundle"
        # No search_path on the inner spawn -- exactly like the real spawn_do.
        _write_command(bundle, "entry", _spawning("nested"))
        _write_command(bundle, "nested", _marking_flowchart("BUNDLE-NESTED", marker))
        monkeypatch.chdir(cwd)

        fc = _spawning("entry", search_path=str(bundle))
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())

        result = await walker.run()

        assert result.status == "completed", result
        # What actually executed, two levels down -- the bundle's copy, not the
        # byte-for-byte decoy sitting in the process working directory.
        assert marker.read_text().split() == ["BUNDLE-NESTED"]

    async def test_a_nested_spawn_can_override_the_inherited_priority(
        self, tmp_path, monkeypatch
    ):
        # Inheriting a priority must not freeze it: a nested spawn that names
        # its own search_path prepends, so the innermost wins.
        marker = tmp_path / "which-ran.txt"
        cwd = tmp_path / "cwd"
        _write_command(
            cwd / "commands", "nested", _marking_flowchart("DECOY-NESTED", marker)
        )

        inner = tmp_path / "inner"
        _write_command(inner, "nested", _marking_flowchart("INNER-NESTED", marker))
        outer = tmp_path / "outer"
        _write_command(outer, "entry", _spawning("nested", search_path=str(inner)))
        _write_command(outer, "nested", _marking_flowchart("OUTER-NESTED", marker))
        monkeypatch.chdir(cwd)

        fc = _spawning("entry", search_path=str(outer))
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol())

        result = await walker.run()

        assert result.status == "completed", result
        assert marker.read_text().split() == ["INNER-NESTED"]

    async def test_without_a_search_path_cwd_still_wins(self, tmp_path, monkeypatch):
        # The invariant: absent search_path, precedence is exactly what it was,
        # cwd ahead of the inherited search paths.
        cwd = tmp_path / "cwd"
        _write_command(cwd / "commands", "child-cmd", _leaf_flowchart("from-cwd"))
        other = tmp_path / "other"
        _write_command(other, "child-cmd", _leaf_flowchart("from-other"))
        monkeypatch.chdir(cwd)

        fc = _spawning("child-cmd")
        walker = GraphWalker(
            fc, MockSession(), {}, MockProtocol(), search_paths=[str(other)]
        )

        recording, made = _recording_walker()
        with patch.object(walker_mod, "GraphWalker", recording):
            result = await walker.run()

        assert result.status == "completed", result
        assert [w._flowchart.name for w in made] == ["from-cwd"]
