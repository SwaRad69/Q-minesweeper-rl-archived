"""Offline Playwright integration: real capture JS and DOM, no website access."""

from pathlib import Path

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

from minesweeper_rl.config import EnvironmentConfig
from minesweeper_rl.data import HumanDatasetStore, PersistentHumanRecorder, validate_dataset
from minesweeper_rl.data.record_human import HumanTrajectoryRecorder, _DRAIN_EVENTS
from minesweeper_rl.environment import MinesweeperEnvironment


HTML = """<div id="top_area_face" class="hd_top-area-face-unpressed"></div>
<div id="board"></div><script>
for (let y=0; y<2; y++) for (let x=0; x<4; x++) {
  const cell=document.createElement('button');
  cell.className='cell hd_closed'; cell.dataset.x=x; cell.dataset.y=y;
  cell.textContent=x+','+y; document.querySelector('#board').append(cell);
  cell.oncontextmenu=e=>e.preventDefault();
  cell.onpointerdown=e=> {
    if (e.button===2) {
      cell.classList.toggle('hd_flag');
    } else if (x===2) {
      document.querySelector('#top_area_face').className='hd_top-area-face-lose';
    } else if (cell.classList.contains('hd_opened')) {
      document.querySelector('[data-x="0"][data-y="1"]').className='cell hd_opened hd_type1';
    } else { cell.className='cell hd_opened hd_type1'; }
  };
}
</script>"""


@pytest.fixture
def browser_recorder(tmp_path):
    with sync_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).exists():
            pytest.skip("Chromium not installed")
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_content(HTML)
            config = EnvironmentConfig(width=4, height=2, observation_timeout_ms=500, observation_poll_ms=10)
            env = MinesweeperEnvironment(page, config)
            store = HumanDatasetStore(tmp_path / "human", width=4, height=2, mines=1)
            sink = PersistentHumanRecorder(store, width=4, height=2, mines=1)
            human = HumanTrajectoryRecorder(env, sink, action_timeout=0.1, quiescence_time=0, poll_interval=0.001, emit=lambda _: None)
            human.install_capture()
            human._handle_idle_board(env.observe())
            yield page, human, store
        finally:
            browser.close()


def test_real_browser_fast_batch_keeps_flag_and_chord_separate(browser_recorder):
    page, human, store = browser_recorder
    page.locator('[data-x="0"][data-y="0"]').click()
    page.locator('[data-x="1"][data-y="0"]').click(button="right")
    page.locator('[data-x="0"][data-y="0"]').click()
    page.locator('[data-x="2"][data-y="0"]').click()
    human.run(max_games=1)
    game = store.load_game("game_000001")
    assert game["result"] == "lost"
    assert [t["action"]["type"] for t in game["transitions"]] == ["open", "flag", "chord", "open"]
    assert game["transitions"][1]["newly_revealed"] == 0
    assert validate_dataset(store.root).ok


def test_evaluate_response_lost_after_js_executes_preserves_press(browser_recorder, monkeypatch):
    page, human, _store = browser_recorder
    page.locator('[data-x="0"][data-y="0"]').click()
    evaluate = page.evaluate
    failed = False

    def lose_response(expression, arg=None):
        nonlocal failed
        result = evaluate(expression, arg)
        if expression == _DRAIN_EVENTS and not failed:
            failed = True
            raise PlaywrightTimeoutError("reply lost after evaluation")
        return result

    monkeypatch.setattr(page, "evaluate", lose_response)
    assert human._safe_drain_events() is None
    events = human._safe_drain_events()
    assert len(events) == 1 and events[0]["isTrusted"]
    assert human.handle_event(events[0]) == "recorded"
    assert human._safe_drain_events() == []


def test_detached_board_restored_after_observation_timeout(browser_recorder):
    page, human, store = browser_recorder
    page.locator('[data-x="0"][data-y="0"]').click()
    event = human.drain_events()[0]
    page.evaluate("window.savedBoard=document.querySelector('#board'); savedBoard.remove()")
    assert human.handle_event(event) == "rejected"
    assert human._pending_action is not None
    page.evaluate("document.body.append(savedBoard)")
    human._handle_idle_board(human.environment.observe())
    assert human.recorder.steps_recorded == 1
    assert store.game_count == 0


def test_real_missing_face_recovers_without_abandonment(browser_recorder):
    page, human, store = browser_recorder
    page.locator('[data-x="0"][data-y="0"]').click()
    event = human.drain_events()[0]
    page.evaluate("window.savedFace=document.querySelector('#top_area_face'); savedFace.remove()")
    assert human.handle_event(event) == "rejected"
    for _ in range(3):
        human._handle_idle_board(human.environment.observe())
    assert human._pending_action is not None
    assert store.game_count == 0
    page.evaluate("document.body.append(savedFace)")
    human._handle_idle_board(human.environment.observe())
    assert human.recorder.steps_recorded == 1
