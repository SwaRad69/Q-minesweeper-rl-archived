"""Long-lived Playwright setup and lifecycle, independent of game semantics."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import BrowserContext, Page, Playwright, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


class BrowserController:
    """Own one headed, persistent Chromium context for a Minesweeper session."""

    def __init__(
        self,
        url: str,
        profile_path: str | Path = "browser_profile",
        *,
        headless: bool = False,
        board_selector: str = ".cell[data-x][data-y]",
        board_timeout_ms: float = 10_000,
        wait_for_board: bool = True,
    ) -> None:
        self.url = url
        self.profile_path = Path(profile_path)
        self.headless = headless
        self.board_selector = board_selector
        self.board_timeout_ms = board_timeout_ms
        self.wait_for_board_on_start = wait_for_board
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

    def start(self) -> Page:
        """Launch the persistent browser and navigate.

        By default the call blocks until at least one board cell appears; pass
        ``wait_for_board=False`` when the landing page needs an extra click
        (for example the Minesweeper homepage) before a board exists.
        """

        if self._context is not None and self.page is not None:
            return self.page

        parsed_url = urlparse(self.url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(f"Invalid Minesweeper URL: {self.url!r}")
        if self.headless:
            raise ValueError("BrowserController requires headed mode (headless=False)")

        try:
            self.profile_path.mkdir(parents=True, exist_ok=True)
            self._playwright = sync_playwright().start()
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_path),
                headless=False,
            )
            self.page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self.page.goto(self.url, wait_until="domcontentloaded")
            if self.wait_for_board_on_start:
                self.wait_for_board()
            return self.page
        except Exception as exc:
            self.shutdown()
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError(f"Could not start Minesweeper browser: {exc}") from exc

    @property
    def context(self) -> BrowserContext | None:
        """Return the persistent context for future environment components."""

        return self._context

    def get_page(self) -> Page:
        """Return the active page, requiring a successful :meth:`start`."""

        if self.page is None:
            raise RuntimeError("BrowserController has not been started")
        return self.page

    def board_exists(self) -> bool:
        """Check whether at least one DOM element matches the board selector."""

        return self.get_page().locator(self.board_selector).count() > 0

    def wait_for_board(self) -> None:
        """Wait until at least one board cell is present in the DOM."""

        try:
            self.get_page().locator(self.board_selector).first.wait_for(
                state="attached", timeout=self.board_timeout_ms
            )
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"Minesweeper board selector {self.board_selector!r} did not appear "
                f"within {self.board_timeout_ms:g} ms"
            ) from exc

    def shutdown(self) -> None:
        """Close the persistent context and Playwright, safely and idempotently."""

        context, playwright = self._context, self._playwright
        self.page = None
        self._context = None
        self._playwright = None
        try:
            if context is not None:
                context.close()
        finally:
            if playwright is not None:
                playwright.stop()

    def close(self) -> None:
        """Backward-compatible alias for :meth:`shutdown`."""

        self.shutdown()

    def __enter__(self) -> BrowserController:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
