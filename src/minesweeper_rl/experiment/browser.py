"""Browser lifecycle helpers shared by training and evaluation.

Wraps :class:`BrowserController` in a context manager so the Playwright browser
is always cleaned up, even when an experiment stops early or raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..browser import BrowserController
from ..config import EnvironmentConfig
from ..environment import MinesweeperEnvironment


@contextmanager
def open_environment(
    url: str,
    profile: str | Path = "browser_profile",
    *,
    config: EnvironmentConfig | None = None,
) -> Iterator[MinesweeperEnvironment]:
    """Yield a live :class:`MinesweeperEnvironment` and always shut it down."""

    controller = BrowserController(url, profile, wait_for_board=False)
    try:
        page = controller.start()
        yield MinesweeperEnvironment(page, config or EnvironmentConfig())
    finally:
        controller.shutdown()
