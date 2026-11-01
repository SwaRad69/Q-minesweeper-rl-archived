"""Browser interaction for a single reveal action."""

from __future__ import annotations

from playwright.sync_api import Locator, Page


class ActionError(ValueError):
    """Base class for invalid or impossible board actions."""


class InvalidActionError(ActionError):
    """Raised when an action targets an invalid or unavailable cell."""


class ActionTimeoutError(ActionError):
    """Raised when an action produced no observable board change."""


class ActionHandler:
    """Click one validated board coordinate through the browser page."""

    def __init__(self, page: Page, *, selector: str, width: int, height: int) -> None:
        self.page = page
        self.selector = selector
        self.width = width
        self.height = height

    def validate(self, x: int, y: int) -> None:
        """Reject coordinates that cannot name a cell on the board."""

        if (
            isinstance(x, bool) or isinstance(y, bool)
            or not isinstance(x, int) or not isinstance(y, int)
        ):
            raise TypeError("coordinates must be integers")
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"coordinate outside board: ({x}, {y})")

    def click(self, x: int, y: int) -> None:
        """Open or chord a cell using Playwright's trusted mouse interaction.

        Both OPEN and CHORD are performed with the primary (left) mouse button;
        the site resolves an OPEN on a hidden cell and a CHORD on an opened
        number itself.
        """

        cell = self._cell(x, y)
        cell.scroll_into_view_if_needed()
        cell.click()

    def right_click(self, x: int, y: int) -> None:
        """Toggle a flag using Playwright's trusted secondary mouse button."""

        cell = self._cell(x, y)
        cell.scroll_into_view_if_needed()
        cell.click(button="right")

    def left_click(self, x: int, y: int) -> None:
        """Reproduce the verified mousedown/mouseup/click DOM event sequence.

        Used as a fallback when Playwright's mouse interaction does not reach
        the website's game handlers.
        """

        cell = self._cell(x, y)
        cell.dispatch_event("mousedown", {"button": 0, "buttons": 1})
        cell.dispatch_event("mouseup", {"button": 0, "buttons": 0})
        cell.dispatch_event("click", {"button": 0, "buttons": 0})

    def right_click_events(self, x: int, y: int) -> None:
        """Reproduce the right-button DOM event sequence used to flag a cell.

        Mirrors the human recorder's verified observation that the site reacts
        to the secondary pointer button; ``contextmenu`` is included because the
        browser emits it for a right click and some site handlers listen for it.
        """

        cell = self._cell(x, y)
        cell.dispatch_event("mousedown", {"button": 2, "buttons": 2})
        cell.dispatch_event("mouseup", {"button": 2, "buttons": 0})
        cell.dispatch_event("contextmenu", {"button": 2, "buttons": 0})

    def _cell(self, x: int, y: int) -> Locator:
        self.validate(x, y)
        selector = f'{self.selector}[data-x="{x}"][data-y="{y}"]'
        cell = self.page.locator(selector)
        count = cell.count()
        if count != 1:
            raise LookupError(f"expected one cell at ({x}, {y}), found {count}")
        return cell
