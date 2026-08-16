"""``max_blocks <= 0`` disables the safety limit.

The limit exists to stop a buggy flowchart looping forever, and it stays on by
default at 1000.  But a deliberately unbounded orchestrator -- one that keeps
spawning epochs until something external stops it -- is not a bug, and it dies
at block 1000 with ``Safety limit: exceeded 1000 blocks``.  Passing
``--max-blocks 0`` opts out; every other value keeps the backstop.

Opt-out rather than opt-in on purpose: flipping the default would let any
runaway loop run forever, which is the case the limit was written for.
"""

from __future__ import annotations

import pytest
from flowcoder_engine.walker import ExecutionError, GraphWalker
from flowcoder_flowchart import (
    Connection,
    EndBlock,
    Flowchart,
    StartBlock,
    VariableBlock,
)

from tests.conftest import MockProtocol, MockSession

# Comfortably past the 1000-block default, so a run that reaches the end could
# not have done so under it.
CHAIN_LENGTH = 1200
LOOP_BLOCKS = 3000


def _chain_flowchart(length: int) -> Flowchart:
    """A straight run of ``length`` variable blocks that terminates.

    Executes ``length + 2`` blocks counting start and end -- long enough to
    prove the limit is gone, finite enough to assert "completed" rather than
    inferring it from a timeout.
    """
    blocks: dict = {"s": StartBlock(id="s")}
    connections = []
    previous = "s"
    for i in range(length):
        block_id = f"v{i}"
        blocks[block_id] = VariableBlock(
            id=block_id, variable_name="x", variable_value=str(i)
        )
        connections.append(Connection(source_id=previous, target_id=block_id))
        previous = block_id
    blocks["e"] = EndBlock(id="e")
    connections.append(Connection(source_id=previous, target_id="e"))
    return Flowchart(blocks=blocks, connections=connections)


def _endless_flowchart() -> Flowchart:
    """A block that loops back to itself -- it never reaches an end block."""
    return Flowchart(
        blocks={
            "s": StartBlock(id="s"),
            "v": VariableBlock(id="v", variable_name="x", variable_value="1"),
            "e": EndBlock(id="e"),
        },
        connections=[
            Connection(source_id="s", target_id="v"),
            Connection(source_id="v", target_id="v"),
        ],
    )


class _HaltingProtocol(MockProtocol):
    """Halts the walker after ``limit`` blocks, so an endless loop can be run.

    Stands in for whatever external signal stops a real unbounded orchestrator.
    Counting here rather than reading ``_blocks_executed`` keeps the assertion
    about how far the run actually got independent of the walker's own counter.
    """

    def __init__(self, walker_box: list[GraphWalker], limit: int):
        super().__init__()
        self._walker_box = walker_box
        self._limit = limit
        self.blocks_started = 0

    # Signature must track Protocol.emit_block_start exactly. It gained
    # `session` with session tagging; this override kept the old three-arg
    # shape and every run through it died on an unexpected kwarg, so the two
    # unlimited-limit cases never exercised the limit at all.
    def emit_block_start(
        self, block_id: str, block_name: str, block_type: str, session: str = ""
    ) -> None:
        self.blocks_started += 1
        if self.blocks_started >= self._limit:
            self._walker_box[0].halt()

    def log(self, message: str) -> None:  # keep the 3000-block run quiet
        pass


class TestUnlimitedMaxBlocks:
    async def test_zero_runs_a_flowchart_past_the_default_limit(self):
        fc = _chain_flowchart(CHAIN_LENGTH)
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol(), max_blocks=0)

        result = await walker.run()

        assert result.status == "completed", result
        assert walker._blocks_executed == CHAIN_LENGTH + 2

    async def test_the_same_flowchart_still_trips_a_positive_limit(self):
        # The other half of the pair: the backstop is disabled by 0, not gone.
        fc = _chain_flowchart(CHAIN_LENGTH)
        walker = GraphWalker(fc, MockSession(), {}, MockProtocol(), max_blocks=1000)

        with pytest.raises(ExecutionError, match="Safety limit"):
            await walker.run()

    async def test_zero_lets_an_endless_loop_run(self):
        # The motivating shape: a loop with no exit, stopped from outside rather
        # than by the limit.
        box: list[GraphWalker] = []
        protocol = _HaltingProtocol(box, LOOP_BLOCKS)
        walker = GraphWalker(
            _endless_flowchart(), MockSession(), {}, protocol, max_blocks=0
        )
        box.append(walker)

        result = await walker.run()

        assert result.status == "halted", result
        assert protocol.blocks_started >= LOOP_BLOCKS

    async def test_a_negative_limit_is_unlimited_too(self):
        box: list[GraphWalker] = []
        protocol = _HaltingProtocol(box, LOOP_BLOCKS)
        walker = GraphWalker(
            _endless_flowchart(), MockSession(), {}, protocol, max_blocks=-1
        )
        box.append(walker)

        result = await walker.run()

        assert result.status == "halted", result
        assert protocol.blocks_started >= LOOP_BLOCKS

    async def test_a_positive_limit_still_stops_an_endless_loop(self):
        walker = GraphWalker(
            _endless_flowchart(), MockSession(), {}, MockProtocol(), max_blocks=50
        )

        with pytest.raises(ExecutionError, match="Safety limit"):
            await walker.run()
        assert walker._blocks_executed == 50
