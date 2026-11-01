"""Configuration for the browser environment."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Stable board and DOM settings discovered for Minesweeper.com."""

    width: int = 30
    height: int = 16
    mines: int = 99
    difficulty: str = "expert"
    cell_selector: str = ".cell[data-x][data-y]"
    ready_timeout_ms: float = 10_000
    observation_timeout_ms: float = 2_000
    observation_poll_ms: float = 25
    action_timeout_ms: float = 5_000
    action_poll_ms: float = 25
    diagnostics: bool = False
    new_game_selector: str = "#top_area_face"
    new_game_timeout_ms: float = 30_000
    first_move: tuple[int, int] | None = None
    first_move_attempts: int = 3
