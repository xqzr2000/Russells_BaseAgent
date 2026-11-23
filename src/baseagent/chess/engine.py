"""Short, shared Stockfish searches for the chess demo."""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Literal

import chess
import chess.engine

from baseagent.chess.bot import choose_move

SEARCH_TIME = 0.06
OPPONENT_SKILL = 4
RETRY_DELAY = 5
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineMove:
    move: chess.Move
    source: Literal["stockfish", "practice"]
    reply: chess.Move | None = None


class Stockfish:
    """Reuse a UCI process, with one search at a time across chats."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._engine: chess.engine.SimpleEngine | None = None
        self._retry_at = 0.0
        self._users = 0

    def choose(self, board: chess.Board, *, beginner: bool = False) -> EngineMove:
        # Piece drills may omit kings. Stockfish requires a valid chess position.
        if board.is_valid():
            with self._lock:
                try:
                    result = self._play(board, beginner)
                    if result and result.move is not None and result.move in board.legal_moves:
                        return EngineMove(result.move, "stockfish", result.ponder)
                finally:
                    if not self._users:
                        self._close()
        return EngineMove(choose_move(board), "practice")

    def _play(self, board: chess.Board, beginner: bool) -> chess.engine.PlayResult | None:
        if monotonic() < self._retry_at:
            return None
        try:
            if self._engine is None:
                command = os.getenv("STOCKFISH_PATH") or shutil.which("stockfish")
                if not command and Path("/usr/games/stockfish").is_file():
                    command = "/usr/games/stockfish"
                if not command:
                    self._retry_at = monotonic() + RETRY_DELAY
                    _logger.info("Stockfish not installed; using the practice engine.")
                    return None
                self._engine = chess.engine.SimpleEngine.popen_uci(command, timeout=2)
            return self._engine.play(board, chess.engine.Limit(time=SEARCH_TIME),
                                     options={"Skill Level": OPPONENT_SKILL if beginner else 20})
        except (OSError, chess.engine.EngineError, TimeoutError) as exc:
            _logger.warning("Stockfish unavailable; using the practice engine: %s", exc)
            self._close()
            self._retry_at = monotonic() + RETRY_DELAY
            return None

    def _close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    @contextmanager
    def reuse(self):
        """Keep Stockfish warm for the life of the server."""
        with self._lock:
            self._users += 1
        try:
            yield
        finally:
            with self._lock:
                self._users -= 1
                if not self._users:
                    self._close()


engine = Stockfish()
