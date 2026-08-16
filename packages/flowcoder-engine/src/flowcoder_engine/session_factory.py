"""SessionFactory — creates sessions of any registered backend type."""

from __future__ import annotations

from typing import Any, Callable

from .session import BaseSession


# Type for a function that creates a session given a name, optional model,
# and optional env overrides.
SessionCreator = Callable[[str, str | None, dict[str, str] | None], BaseSession]


class SessionFactory:
    """Registry of backend session creators.

    Each backend (e.g. "claude") is registered with a creator function
    that produces a BaseSession given a name and optional model.
    """

    def __init__(self) -> None:
        self._creators: dict[str, SessionCreator] = {}

    def register(self, backend: str, creator: SessionCreator) -> None:
        """Register a creator function for a backend name."""
        self._creators[backend] = creator

    def create(
        self,
        backend: str,
        name: str,
        model: str | None = None,
        env: dict[str, str] | None = None,
    ) -> BaseSession:
        """Create a session for the given backend."""
        if backend not in self._creators:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available: {', '.join(self._creators) or 'none'}"
            )
        return self._creators[backend](name, model, env)

    @property
    def backends(self) -> list[str]:
        return list(self._creators)
