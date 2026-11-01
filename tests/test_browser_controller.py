from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minesweeper_rl.browser import controller as controller_module
from minesweeper_rl.browser.controller import BrowserController


class FakeLocator:
    def __init__(self, count: int = 1) -> None:
        self.count_value = count
        self.first = self

    def count(self) -> int:
        return self.count_value

    def wait_for(self, **_: object) -> None:
        return None


class FakePage:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.locator_instance = FakeLocator()

    def goto(self, url: str, **_: object) -> None:
        self.urls.append(url)

    def locator(self, _: str) -> FakeLocator:
        return self.locator_instance


class FakeContext:
    def __init__(self) -> None:
        self.pages = [FakePage()]
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePlaywright:
    def __init__(self) -> None:
        self.context = FakeContext()
        self.stopped = False
        self.chromium = SimpleNamespace(
            launch_persistent_context=lambda **_: self.context
        )

    def stop(self) -> None:
        self.stopped = True


def test_start_uses_one_persistent_context_and_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePlaywright()
    monkeypatch.setattr(controller_module, "sync_playwright", lambda: SimpleNamespace(start=lambda: fake))
    controller = BrowserController("https://minesweeper.com/game", tmp_path / "profile")

    page = controller.start()
    assert page is fake.context.pages[0]
    assert page.urls == ["https://minesweeper.com/game"]
    assert controller.start() is page
    assert controller.board_exists() is True

    controller.shutdown()
    controller.shutdown()
    assert fake.context.closed is True
    assert fake.stopped is True


def test_invalid_url_does_not_start_playwright(tmp_path: Path) -> None:
    controller = BrowserController("not-a-url", tmp_path / "profile")

    with pytest.raises(ValueError, match="Invalid Minesweeper URL"):
        controller.start()


def test_page_requires_start(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="has not been started"):
        BrowserController("https://minesweeper.com", tmp_path / "profile").get_page()
