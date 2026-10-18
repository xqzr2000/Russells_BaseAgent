"""One chat's chess board: the position, the game, and what the coach draws on it.

``ChessSession`` is the single source of truth for chess facts in a chat.
Coach tools and the move endpoint change it, and every change is published to
the UI as a ``board`` event built by ``view()``. The model never states a
move this object has not checked.

Two modes for now:

    lesson   the coach sets up positions (kingless ones are fine, e.g. a lone
             knight) and the user moves only when the coach unlocks the board.
             Black answers each move automatically (``black_reply="auto"``), or
             never moves so White can keep practising (``"pass"``). The user can
             step back or restart the lesson position (``rewind``).
    game     a full game: the user is White, the bot replies as Black. The user
             can take a move back (``rewind("back")``).

Every move is also described in plain words (``narration``) for beginners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import chess

from baseagent.chess.bot import choose_move

Mode = Literal["lesson", "game"]
BlackReply = Literal["auto", "pass"]
ArrowKind = Literal["best", "hint", "idea", "threat"]
HighlightColor = Literal["yellow", "green", "red", "blue"]

ARROW_COLORS: dict[str, str] = {"best": "green", "hint": "blue", "idea": "blue", "threat": "red"}
START_FEN = chess.STARTING_FEN
PIECE_NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}


class ChessError(ValueError):
    """A move or position the board rejected; the message is shown to the model or user."""


@dataclass
class MoveOutcome:
    """What a user's move did, for the move endpoint."""

    user_san: str
    user_words: str = ""  # "I moved my knight from g1 to f3."
    reply_san: str | None = None  # Black's automatic answer
    reply_words: str = ""  # "Black moved its pawn from e7 to e5."
    game_over: bool = False
    needs_coach: bool = False  # should the coach speak about this move?

    def message(self) -> str:
        """The chat message that reports the move to the coach, in plain words."""
        text = f"{self.user_words.rstrip('.')} ({self.user_san})."
        if self.reply_san:
            text += f" {self.reply_words.rstrip('.')} ({self.reply_san})."
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
    draw_claimed: bool = False
    black_reply: BlackReply = "auto"  # lessons: does Black answer the user's moves?
    narration: str = ""  # the last moves in plain words, e.g. "You moved your knight..."
    lesson_start: dict[str, Any] | None = None  # what `rewind("start")` restores

    # ------------------------------------------------------------ positions

    def set_position(self, fen: str = "start", caption: str = "", user_moves_next: bool = False,
                     black_reply: BlackReply = "auto") -> None:
        """Show a lesson position. ``fen`` may omit kings (e.g. one knight on an empty board)."""

        board = self._parse_fen(fen)
        if user_moves_next and board.turn != chess.WHITE:
            raise ChessError("The user always plays White, so a position where the user moves "
                             "next must have White to move (' w ' in the FEN).")
        self.board = board
        self.mode = "lesson"
        self.draw_claimed = False
        self.user_can_move = user_moves_next
        self.black_reply = black_reply
        self.caption = caption
        self.narration = ""
        self.last_move = None
        self.clear_overlay()
        self.save_lesson_start()

    def save_lesson_start(self) -> None:
        """Remember the lesson position and its drawings (arrows, dots...) for "Restart"."""
        if self.mode == "lesson" and not self.board.move_stack:
            self.lesson_start = {"fen": self.board.fen(), "caption": self.caption,
                                 "arrows": list(self.arrows), "highlights": list(self.highlights),
                                 "dots": list(self.dots), "user_can_move": self.user_can_move}

    def new_game(self) -> None:
        """Start a full game from the opening position: the user is White, the bot Black."""

        self.board = chess.Board()
        self.mode = "game"
        self.draw_claimed = False
        self.user_can_move = True
        self.black_reply = "auto"
        self.caption = "New game: you play White. Make your first move."
        self.narration = ""
        self.last_move = None
        self.lesson_start = None
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

    def find_move(self, text: str, side: Literal["to_move", "opponent", "either"] = "either"
                  ) -> tuple[chess.Move, chess.Board]:
        """A legal move for the side to move, for its opponent (a threat), or for either.

        ``side="either"`` tries the side to move first. Arrows for threats pass
        ``"opponent"``, so a SAN move both sides could play (two knights that can
        reach f6) is read as the opponent's.
        """

        boards = {"to_move": [self.board], "opponent": [self._other_side_board()],
                  "either": [self.board, self._other_side_board()]}[side]
        for board in boards:
            try:
                return self.parse_move(text, board), board
            except ChessError:
                continue
        who = {"to_move": f"{_side(self.board)} (the side to move)",
               "opponent": f"{_side(boards[0])} (the side not to move)",
               "either": "either side"}[side]
        raise ChessError(f"{text} is not a legal move for {who} in this position.")

    def user_move(self, text: str) -> MoveOutcome:
        """Play the user's (White's) move; in a game the bot answers at once."""

        if not self.user_can_move:
            raise ChessError("The board is locked right now. Wait for the coach.")
        if self.game_over:
            raise ChessError("The game is over. Ask the coach for a new game.")
        if self.board.turn != chess.WHITE:
            raise ChessError("It is not White's turn.")
        move = self.parse_move(text)
        outcome = MoveOutcome(user_san=self.board.san(move),
                              user_words=describe_move(self.board, move, "I"))
        narration = [describe_move(self.board, move, "you")]
        self._push(move)

        # Black answers at once, so the user is never left waiting on a frozen board.
        # A move that ends the game gets no reply (the CMU chess-app bug).
        if not self.game_over:
            if self.mode == "game" or (self.black_reply == "auto" and any(self.board.legal_moves)):
                reply = choose_move(self.board)
                outcome.reply_san = self.board.san(reply)
                outcome.reply_words = describe_move(self.board, reply, "Black")
                narration.append(outcome.reply_words)
                self._push(reply)
            else:
                # Black has nothing to move here (or the lesson says it doesn't answer):
                # it's simply the user's turn again.
                self.board.push(chess.Move.null())
                if self.black_reply == "auto":
                    narration.append("Black has nothing to move, so it's your turn again.")

        outcome.game_over = self.game_over
        self.narration = " ".join(narration)
        self.user_can_move = not outcome.game_over
        if self.mode == "game":
            outcome.needs_coach = outcome.game_over  # games stay fast: no model call per move
            self.caption = self._status_line()
        else:
            outcome.needs_coach = True  # in a lesson the coach comments on every move
            # The lesson's instruction stays until the coach gives the next one.
        return outcome

    def rewind(self, to: Literal["back", "start"] = "back") -> str:
        """Take back the user's last move (and Black's answer), or restart the lesson position."""

        if to == "start":
            if self.mode != "lesson" or self.lesson_start is None:
                raise ChessError("Restart only works in a lesson. Ask the coach for a new game instead.")
            start = self.lesson_start
            self.board = chess.Board(start["fen"])
            self.caption = start["caption"]
            self.arrows, self.highlights = list(start["arrows"]), list(start["highlights"])
            self.dots = list(start["dots"])
            self.user_can_move = start["user_can_move"]
            self.last_move, self.draw_claimed = None, False
            self.narration = "Back to the start of this lesson."
            return self.narration
        if not self.board.move_stack:
            raise ChessError("There is no move to take back yet.")
        self.board.pop()
        while self.board.move_stack and self.board.turn != chess.WHITE:
            self.board.pop()  # Black's answer (or a pass) and the user's move go together
        self.draw_claimed = False
        self.user_can_move = True
        if not self.board.move_stack and self.mode == "lesson" and self.lesson_start:
            self.caption = self.lesson_start["caption"]
            self.arrows = list(self.lesson_start["arrows"])
            self.highlights = list(self.lesson_start["highlights"])
            self.dots = list(self.lesson_start["dots"])
        else:
            self.clear_overlay()
            self.caption = self._status_line()
        moves = [m for m in self.board.move_stack if m]
        self.last_move = moves[-1].uci() if moves else None
        self.narration = "Took your last move back. Try again!"
        return self.narration

    def claim_draw(self) -> str:
        """End a game as a draw by threefold repetition or the 50-move rule, if White may claim it."""

        if self.mode != "game" or self.game_over:
            raise ChessError("There is no game in progress to claim a draw in.")
        if not self.board.can_claim_draw():
            raise ChessError("A draw can't be claimed in this position.")
        reason = ("threefold repetition" if self.board.can_claim_threefold_repetition()
                  else "the 50-move rule")
        self.draw_claimed = True
        self.user_can_move = False
        self.caption = f"Draw claimed ({reason})."
        return reason

    def _push(self, move: chess.Move) -> None:
        self.board.push(move)
        self.last_move = move.uci()
        self.clear_overlay()

    def _other_side_board(self) -> chess.Board:
        other = self.board.copy(stack=False)
        other.turn = not other.turn
        other.ep_square = None
        return other

    def suggest(self) -> tuple[chess.Move, str]:
        """A sensible move for the side to move (the bot's 2-ply search) and what it does in words."""

        if not any(self.board.legal_moves):
            raise ChessError("There is no legal move to suggest here.")
        move = choose_move(self.board)
        return move, describe_move(self.board, move, "suggest")

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

    def attacked_by(self, square_name: str) -> list[str]:
        """Squares the piece on ``square_name`` attacks (whether or not they are occupied)."""

        square = _square(square_name)
        if self.board.piece_at(square) is None:
            raise ChessError(f"There is no piece on {square_name}.")
        return sorted((chess.square_name(s) for s in self.board.attacks(square)), key=_square)

    def attackers_of(self, square_name: str) -> dict[str, list[str]]:
        """Pieces of each colour that attack ``square_name``, as SAN-style labels (e.g. Nf3)."""

        square = _square(square_name)
        found: dict[str, list[str]] = {}
        for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            labels = []
            for attacker in sorted(self.board.attackers(color, square)):
                piece = self.board.piece_at(attacker)
                letter = "" if piece.piece_type == chess.PAWN else piece.symbol().upper()
                labels.append(f"{letter}{chess.square_name(attacker)}")
            found[name] = labels
        return found

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
            # Only results that end the game by themselves; threefold repetition and the
            # 50-move rule must be claimed (claim_draw), as in over-the-board chess.
            return self.draw_claimed or self.board.outcome() is not None
        # Lessons ignore draw rules (K+N vs K is a fine teaching position).
        return self.board.is_checkmate() or self.board.is_stalemate()

    @property
    def draw_claimable(self) -> bool:
        return (self.mode == "game" and self.has_kings and not self.game_over
                and self.board.turn == chess.WHITE and self.board.can_claim_draw())

    @property
    def interactive(self) -> bool:
        return self.user_can_move and not self.game_over and self.board.turn == chess.WHITE

    def _status_line(self) -> str:
        board = self.board
        if self.mode == "game":
            if self.draw_claimed:
                return self.caption
            outcome = board.outcome() if self.has_kings else None
            if outcome is not None:
                if outcome.winner is None:
                    return f"Draw ({outcome.termination.name.lower().replace('_', ' ')})."
                return "Checkmate: you win!" if outcome.winner == chess.WHITE else "Checkmate: the bot wins."
            return "Check!" if board.is_check() else ""  # the panel already says whose move it is
        return ""

    def history_san(self) -> list[str]:
        """The moves played from the position the coach set up (passes left out)."""
        replay = chess.Board(self.board.root().fen()) if self.board.move_stack else None
        if replay is None:
            return []
        moves = []
        for move in self.board.move_stack:
            if move:  # a null move is a pass in a lesson
                moves.append(replay.san(move))
            replay.push(move)
        return moves

    def view(self) -> dict[str, Any]:
        """The board as the UI draws it (the payload of a ``board`` event)."""

        return {
            "fen": self.board.fen(),
            "mode": self.mode,
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "interactive": self.interactive,
            "last_move": self.last_move,
            "in_check": self.has_kings and self.board.is_check(),
            "game_over": self.game_over,
            "draw_claimable": self.draw_claimable,
            "caption": self.caption,
            "narration": self.narration,
            # Where each white piece may go, so the board can show dots when one is picked up.
            "legal_moves": sorted(m.uci() for m in self.board.legal_moves) if self.interactive else [],
            "can_step_back": bool(self.board.move_stack) and not self.draw_claimed,
            "can_restart": self.mode == "lesson" and self.lesson_start is not None
                           and self.board.fen() != self.lesson_start["fen"],
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
            f"user_can_move: {self.interactive}",
            f"black_answers_automatically: {self.mode == 'game' or self.black_reply == 'auto'}",
            f"in_check: {self.has_kings and self.board.is_check()}",
            f"game_over: {self.game_over}" + (f" ({self._status_line()})" if self.game_over and self.mode == 'game' else ""),
            f"last_move: {self.last_move or '(none)'}",
            f"what_just_happened: {self.narration or '(nothing yet)'}",
            f"draw_claimable_by_user: {self.draw_claimable}",
            "board (uppercase = White, lowercase = Black):",
            *rows,
            f"moves_so_far: {history}",
            f"legal_moves_for_side_to_move: {legal}",
            *overlay,
            f"caption: {self.caption or '(none)'}",
            "</chess_state>",
        ]
        return "\n".join(lines)


def describe_move(board: chess.Board, move: chess.Move,
                  who: Literal["you", "I", "Black", "suggest"]) -> str:
    """A move in plain words for a beginner, e.g. "You moved your knight from g1 to f3."

    ``board`` is the position before the move. ``who="suggest"`` phrases it as advice for
    the side to move: "Move your knight from g1 to f3."
    """

    mover_is_black = board.turn == chess.BLACK
    if who == "suggest":
        own = "its" if mover_is_black else "your"
        castled, captured_verb, moved_verb, became = "Castle", "Capture", "Move", "becomes"
        subject = ""
    else:
        subject = {"you": "You", "I": "I", "Black": "Black"}[who] + " "
        own = {"you": "your", "I": "my", "Black": "its"}[who]
        castled, captured_verb, moved_verb, became = "castled", "captured", "moved", "became"
    other = "White's" if mover_is_black else "Black's"
    piece = board.piece_at(move.from_square)
    name = PIECE_NAMES[piece.piece_type] if piece else "piece"
    start, end = chess.square_name(move.from_square), chess.square_name(move.to_square)
    if board.is_castling(move):
        side = "kingside (short)" if chess.square_file(move.to_square) > 4 else "queenside (long)"
        text = f"{subject}{castled} {side}: the king and a rook move together"
    elif board.is_capture(move):
        captured = (chess.PAWN if board.is_en_passant(move)
                    else board.piece_at(move.to_square).piece_type)
        how = " en passant" if board.is_en_passant(move) else ""
        text = f"{subject}{captured_verb} {other} {PIECE_NAMES[captured]} on {end} with {own} {name}{how}"
    else:
        text = f"{subject}{moved_verb} {own} {name} from {start} to {end}"
    if move.promotion:
        text += f", and the pawn {became} a {PIECE_NAMES[move.promotion]}"
    if board.gives_check(move):
        text += " (check!)" if not _is_mate_after(board, move) else " (checkmate!)"
    return text + "."


def _is_mate_after(board: chess.Board, move: chess.Move) -> bool:
    after = board.copy(stack=False)
    after.push(move)
    return after.is_checkmate()


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
