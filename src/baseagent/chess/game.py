"""One chat's chess board: the position, the game, and what the coach draws on it.

``ChessSession`` is the single source of truth for chess facts in a chat.
Coach tools and the move endpoint change it, and every change is published to
the UI as a ``board`` event built by ``view()``. The model never states a
move this object has not checked.

Two modes for now:

    lesson   the coach sets up positions (kingless ones are fine, e.g. a lone
             knight) and the user moves only when the coach unlocks the board
    game     a full game: the user is White, the bot replies as Black
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import chess

from baseagent.chess.bot import choose_move

Mode = Literal["lesson", "game"]
ArrowKind = Literal["best", "hint", "idea", "threat"]
HighlightColor = Literal["yellow", "green", "red", "blue"]

ARROW_COLORS: dict[str, str] = {"best": "green", "hint": "blue", "idea": "blue", "threat": "red"}
START_FEN = chess.STARTING_FEN


class ChessError(ValueError):
    """A move or position the board rejected; the message is shown to the model or user."""


@dataclass
class MoveOutcome:
    """What a user's move did, for the move endpoint."""

    user_san: str
    reply_san: str | None = None  # the bot's answer in a game
    game_over: bool = False
    needs_coach: bool = False  # should the coach speak about this move?

    def message(self) -> str:
        """The chat message that reports the move to the coach."""
        text = f"I played {self.user_san}."
        if self.reply_san:
            text += f" The bot answered {self.reply_san}."
        return text


@dataclass
class ChessSession:
    board: chess.Board = field(default_factory=chess.Board)
    mode: Mode = "lesson"
    user_can_move: bool = False
    caption: str = ""
    arrows: list[dict[str, str]] = field(default_factory=list)
    highlights: list[dict[str, str]] = field(default_factory=list)
    dots: list[str] = field(default_factory=list)
    last_move: str | None = None

    # ------------------------------------------------------------ positions

    def set_position(self, fen: str = "start", caption: str = "", user_moves_next: bool = False) -> None:
        """Show a lesson position. ``fen`` may omit kings (e.g. one knight on an empty board)."""

        board = self._parse_fen(fen)
        if user_moves_next and board.turn != chess.WHITE:
            raise ChessError("The user always plays White, so a position where the user moves "
                             "next must have White to move (' w ' in the FEN).")
        self.board = board
        self.mode = "lesson"
        self.user_can_move = user_moves_next
        self.caption = caption
        self.last_move = None
        self.clear_overlay()

    def new_game(self) -> None:
        """Start a full game from the opening position: the user is White, the bot Black."""

        self.board = chess.Board()
        self.mode = "game"
        self.user_can_move = True
        self.caption = "New game: you play White. Make your first move."
        self.last_move = None
        self.clear_overlay()

    @staticmethod
    def _parse_fen(fen: str) -> chess.Board:
        text = (fen or "").strip()
        if text.lower() in ("", "start", "startpos", "initial"):
            return chess.Board()
        parts = text.split()
        # Accept just the piece placement (and optionally the side to move) for lessons.
        defaults = ["", "w", "-", "-", "0", "1"]
        parts += defaults[len(parts):]
        try:
            board = chess.Board(" ".join(parts[:6]))
        except ValueError as exc:
            raise ChessError(f"Invalid FEN {fen!r}: {exc}") from exc
        # Lessons may leave out kings; anything else wrong with the position is a mistake.
        allowed = chess.STATUS_NO_WHITE_KING | chess.STATUS_NO_BLACK_KING
        problems = board.status() & ~allowed
        if problems:
            raise ChessError(f"Invalid position {fen!r}: {_status_text(problems)}.")
        return board

    # ------------------------------------------------------------ moves

    def parse_move(self, text: str, board: chess.Board | None = None) -> chess.Move:
        """Read a move in UCI (e2e4) or SAN (Nf3) and check it is legal on ``board``."""

        board = board or self.board
        text = (text or "").strip()
        if not text:
            raise ChessError("A move is required, e.g. e2e4 or Nf3.")
        try:
            move = chess.Move.from_uci(text.lower())
        except ValueError:
            try:
                return board.parse_san(text)
            except ValueError as exc:
                raise ChessError(f"{text} is not a legal move here ({_side(board)} to move).") from exc
        # A pawn dragged to the last rank becomes a queen unless a piece was named.
        piece = board.piece_at(move.from_square)
        if (piece and piece.piece_type == chess.PAWN and move.promotion is None
                and chess.square_rank(move.to_square) in (0, 7)):
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        if move not in board.legal_moves:
            raise ChessError(f"{text} is not a legal move here ({_side(board)} to move).")
        return move

    def find_move(self, text: str) -> tuple[chess.Move, chess.Board]:
        """A legal move for either side: the side to move first, then the other side (a threat)."""

        try:
            return self.parse_move(text), self.board
        except ChessError:
            other = self._other_side_board()
            try:
                return self.parse_move(text, other), other
            except ChessError:
                raise ChessError(f"{text} is not a legal move for either side in this position.") from None

    def user_move(self, text: str) -> MoveOutcome:
        """Play the user's (White's) move; in a game the bot answers at once."""

        if not self.user_can_move:
            raise ChessError("The board is locked right now. Wait for the coach.")
        if self.game_over:
            raise ChessError("The game is over. Ask the coach for a new game.")
        if self.board.turn != chess.WHITE:
            raise ChessError("It is not White's turn.")
        move = self.parse_move(text)
        outcome = MoveOutcome(user_san=self.board.san(move))
        self._push(move)

        if self.mode == "game":
            # A move that ends the game gets no reply (the CMU chess-app bug).
            if not self.game_over:
                reply = choose_move(self.board)
                outcome.reply_san = self.board.san(reply)
                self._push(reply)
            outcome.game_over = self.game_over
            outcome.needs_coach = outcome.game_over
            self.user_can_move = not outcome.game_over
            self.caption = self._status_line()
        else:
            # In a lesson the coach decides what happens next.
            outcome.game_over = self.game_over
            outcome.needs_coach = True
            self.user_can_move = False
            self.caption = ""
        return outcome

    def _push(self, move: chess.Move) -> None:
        self.board.push(move)
        self.last_move = move.uci()
        self.clear_overlay()

    def _other_side_board(self) -> chess.Board:
        other = self.board.copy(stack=False)
        other.turn = not other.turn
        other.ep_square = None
        return other

    # ------------------------------------------------------------ overlay

    def clear_overlay(self) -> None:
        self.arrows, self.highlights, self.dots = [], [], []

    def add_arrow(self, move: chess.Move, kind: ArrowKind) -> None:
        arrow = {"from": chess.square_name(move.from_square), "to": chess.square_name(move.to_square),
                 "color": ARROW_COLORS[kind]}
        self.arrows = [a for a in self.arrows if (a["from"], a["to"]) != (arrow["from"], arrow["to"])]
        self.arrows.append(arrow)

    def piece_targets(self, square_name: str) -> list[str]:
        """Squares the piece on ``square_name`` can move to (for either colour)."""

        square = _square(square_name)
        piece = self.board.piece_at(square)
        if piece is None:
            raise ChessError(f"There is no piece on {square_name}.")
        board = self.board if piece.color == self.board.turn else self._other_side_board()
        targets = sorted({chess.square_name(m.to_square) for m in board.legal_moves
                          if m.from_square == square}, key=_square)
        return targets

    def show_piece_moves(self, square_name: str) -> list[str]:
        targets = self.piece_targets(square_name)
        self.dots = targets
        self.highlights = [h for h in self.highlights if h["square"] != square_name.lower()]
        self.highlights.append({"square": square_name.lower(), "color": "yellow"})
        return targets

    def highlight(self, squares: list[str], color: HighlightColor) -> None:
        names = [chess.square_name(_square(s)) for s in squares]
        self.highlights = [h for h in self.highlights if h["square"] not in names]
        self.highlights += [{"square": name, "color": color} for name in names]

    # ------------------------------------------------------------ state

    @property
    def has_kings(self) -> bool:
        return self.board.king(chess.WHITE) is not None and self.board.king(chess.BLACK) is not None

    @property
    def game_over(self) -> bool:
        if not self.has_kings:
            return False
        if self.mode == "game":
            return self.board.outcome(claim_draw=True) is not None
        # Lessons ignore draw rules (K+N vs K is a fine teaching position).
        return self.board.is_checkmate() or self.board.is_stalemate()

    def _status_line(self) -> str:
        board = self.board
        if self.mode == "game":
            outcome = board.outcome(claim_draw=True) if self.has_kings else None
            if outcome is not None:
                if outcome.winner is None:
                    return f"Draw ({outcome.termination.name.lower().replace('_', ' ')})."
                return "Checkmate: you win!" if outcome.winner == chess.WHITE else "Checkmate: the bot wins."
            return "Check!" if board.is_check() else ""  # the panel already says whose move it is
        return ""

    def history_san(self) -> list[str]:
        replay = chess.Board(self.board.root().fen()) if self.board.move_stack else None
        if replay is None:
            return []
        moves = []
        for move in self.board.move_stack:
            moves.append(replay.san(move))
            replay.push(move)
        return moves

    def view(self) -> dict[str, Any]:
        """The board as the UI draws it (the payload of a ``board`` event)."""

        return {
            "fen": self.board.fen(),
            "mode": self.mode,
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "interactive": self.user_can_move and not self.game_over and self.board.turn == chess.WHITE,
            "last_move": self.last_move,
            "in_check": self.has_kings and self.board.is_check(),
            "game_over": self.game_over,
            "caption": self.caption,
            "arrows": list(self.arrows),
            "highlights": list(self.highlights),
            "dots": list(self.dots),
            "history": self.history_san(),
        }

    def describe(self) -> str:
        """The position as text for the model (adapted from the CMU agent's format_chess_state)."""

        rows = ["    a b c d e f g h"]
        for rank in range(7, -1, -1):
            cells = []
            for file in range(8):
                piece = self.board.piece_at(chess.square(file, rank))
                cells.append(piece.symbol() if piece else ".")
            rows.append(f"{rank + 1} | {' '.join(cells)} | {rank + 1}")
        rows.append("    a b c d e f g h")
        legal = " ".join(sorted(self.board.san(m) for m in self.board.legal_moves)) or "(none)"
        history = " ".join(self.history_san()[-16:]) or "(none)"
        overlay = []
        if self.arrows:
            overlay.append("arrows: " + ", ".join(f"{a['from']}->{a['to']} ({a['color']})" for a in self.arrows))
        if self.dots:
            overlay.append("dots: " + " ".join(self.dots))
        if self.highlights:
            overlay.append("highlights: " + " ".join(f"{h['square']}({h['color']})" for h in self.highlights))
        lines = [
            "<chess_state>",
            f"mode: {self.mode}",
            f"fen: {self.board.fen()}",
            f"side_to_move: {'white (the user)' if self.board.turn == chess.WHITE else 'black'}",
            f"user_can_move: {self.view()['interactive']}",
            f"in_check: {self.has_kings and self.board.is_check()}",
            f"game_over: {self.game_over}" + (f" ({self._status_line()})" if self.game_over and self.mode == 'game' else ""),
            f"last_move: {self.last_move or '(none)'}",
            "board (uppercase = White, lowercase = Black):",
            *rows,
            f"moves_so_far: {history}",
            f"legal_moves_for_side_to_move: {legal}",
            *overlay,
            f"caption: {self.caption or '(none)'}",
            "</chess_state>",
        ]
        return "\n".join(lines)


def _square(name: str) -> int:
    try:
        return chess.parse_square((name or "").strip().lower())
    except ValueError:
        raise ChessError(f"{name!r} is not a square; use names like e4.") from None


def _side(board: chess.Board) -> str:
    return "White" if board.turn == chess.WHITE else "Black"


def _status_text(status: int) -> str:
    names = [name[len("STATUS_"):].lower().replace("_", " ") for name in dir(chess)
             if name.startswith("STATUS_") and name != "STATUS_VALID" and status & getattr(chess, name)]
    return ", ".join(names) or "invalid"
