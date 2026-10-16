"""The chess coach: teaches chess on a board shown at the top of the chat.

Milestone 1 of docs/CHESS_COACH_PLAN.md. The coach can set up positions,
draw arrows, show where a piece can move, highlight squares, and start a game
against a very easy bot. Every tool goes through ``ChessSession``, which
checks the chess, and each change is sent to the chat room as a ``board``
event. The model reads the live position in its system prompt on every step,
so it always sees moves the user made on the board in between.
"""

from __future__ import annotations

from typing import Any, Literal

from baseagent.agent import BaseAgent
from baseagent.agents import register_agent
from baseagent.chess import ChessError, ChessSession
from baseagent.events import Board
from baseagent.tools import ToolContext, ToolError, tool

SYSTEM_PROMPT = """\
You are Jack's chess coach in Russell's BaseAgent chat room: a warm, patient teacher who \
can teach anyone, from a person who has never seen a chess board to a strong club player.

The board is shown above the chat. The user always plays White, and the board is always \
seen from White's side. Teach WITH the board: whenever you talk about a position, a piece \
or a move, put it on the board with your tools, then refer to it ("see the green arrow", \
"the dots show...").

Tools:
- `set_position` shows a position. Lesson positions may leave out the kings (e.g. \
"8/8/8/3N4/8/8/8/8 w" is a lone knight on d5). Set `user_moves_next` to let the user try \
a move; the board is locked otherwise.
- `show_move` draws an arrow: kind "best" (green), "hint" or "idea" (blue), "threat" (red, \
a move the other side could play).
- `show_piece_moves` puts dots on every square a piece can reach.
- `show_attacks` highlights what a piece attacks; `show_attackers` highlights the pieces that \
attack a square (its attackers and defenders). Use these for any claim about attacks or defence.
- `highlight` only tints squares for emphasis; `clear_overlay` removes arrows, dots and highlights.
- `new_game` starts a full game: the user is White and a very easy bot answers as Black \
automatically. You only hear about the game when the user writes or the game ends. When the \
user asks to play, start the game right away. A repeated position or 50 quiet moves doesn't end \
the game by itself: the user may claim the draw with the board's "Claim draw" button.

Rules:
- Never invent chess facts. Name only moves listed in legal_moves_for_side_to_move, or that a \
tool accepted. If a tool rejects something, fix it rather than guessing. There is no engine \
yet, so don't call a move "the best move" unless it is forced or obvious (mate in one, \
winning a free piece); otherwise talk about ideas and principles.
- When the user reports a move ("I played e4"), look at the live position below and respond \
to what actually happened. When a game ends, congratulate or console them and offer another game \
or a lesson.
- Never present an attack, defence or threat you haven't shown with a tool.
- Find out the user's level early (have they played before? do they know how the pieces \
move?) and teach at that level: one idea at a time, short messages (2 to 5 sentences), \
plain language, and end with a small question or a "your turn" task on the board.
- Write moves in backticks in standard notation, e.g. `Nf3`, `e4`, `O-O`.

Live position (updated after every move):
{state}"""


@register_agent
class ChessCoachAgent(BaseAgent):
    name = "chess_coach"
    title = "Chess coach"
    description = (
        "Learn chess on a board above the chat, from how the pieces move to playing a full game. "
        "The coach shows moves with arrows and dots."
    )
    examples = [
        "Teach me chess. I've never played.",
        "How does the knight move?",
        "Let's play a game.",
        "What are the basic opening principles?",
    ]
    default_settings = {"max_steps": 12}
    handoff_target = True  # "let's play chess" in the main chat hands the chat to the coach

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chess = ChessSession()

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(state=self.chess.describe())

    def get_tools(self):
        return [set_position, show_move, show_piece_moves, show_attacks, show_attackers, highlight,
                clear_overlay, new_game]

    def ui_state(self) -> dict[str, Any]:
        return {"board": self.chess.view()}

    async def publish_board(self, ctx: ToolContext | None = None) -> None:
        await self.emit(Board(turn=ctx.turn if ctx else self.turn, board=self.chess.view()))


def _coach(ctx: ToolContext) -> ChessCoachAgent:
    agent = ctx.agent
    if not isinstance(agent, ChessCoachAgent):
        raise ToolError("This tool only works for the chess coach.")
    return agent


async def _apply(ctx: ToolContext, change) -> Any:
    """Run one board change, turn chess errors into tool errors, and publish the board."""

    coach = _coach(ctx)
    try:
        result = change(coach.chess)
    except ChessError as exc:
        raise ToolError(str(exc)) from exc
    await coach.publish_board(ctx)
    return result


@tool(exclusive=True)
async def set_position(fen: str = "start", caption: str = "", user_moves_next: bool = False,
                       ctx: ToolContext = None) -> str:
    """Show a position on the board (clears arrows, dots and highlights).

    Args:
        fen: A FEN string, "start" for the opening position, or just the piece placement
            plus side to move for a lesson (kings may be left out), e.g. "8/8/8/3N4/8/8/8/8 w".
        caption: One short line shown under the board, e.g. "The knight jumps in an L-shape".
        user_moves_next: True unlocks the board so the user can try a move. White must be to move.
    """
    await _apply(ctx, lambda chess: chess.set_position(fen, caption, user_moves_next))
    board = _coach(ctx).chess
    unlocked = "The user can move now." if board.view()["interactive"] else "The board is locked."
    return f"Position shown ({board.board.fen()}). {unlocked}"


@tool
async def show_move(move: str, kind: Literal["best", "hint", "idea", "threat"] = "idea",
                    caption: str | None = None, keep_other_arrows: bool = True,
                    ctx: ToolContext = None) -> str:
    """Draw an arrow for a move without playing it.

    Args:
        move: The move in SAN (Nf3) or UCI (g1f3).
        kind: best = green and hint = blue (a move for the side to move); threat = red (a move
            the side NOT to move could play); idea = blue (either side).
        caption: Optional line under the board; omit to keep the current one.
        keep_other_arrows: False removes the arrows already drawn.
    """

    side = {"threat": "opponent", "best": "to_move", "hint": "to_move"}.get(kind, "either")

    def change(chess):
        parsed, board = chess.find_move(move, side)
        if not keep_other_arrows:
            chess.arrows = []
        chess.add_arrow(parsed, kind)
        if caption is not None:
            chess.caption = caption
        mover = "White" if board.turn else "Black"
        return f"Arrow drawn for {board.san(parsed)} ({mover}, {kind})."

    return await _apply(ctx, change)


@tool
async def show_piece_moves(square: str, caption: str | None = None, ctx: ToolContext = None) -> str:
    """Put dots on every square the piece on `square` can move to (replaces earlier dots).

    Args:
        square: The piece's square, e.g. d5.
        caption: Optional line under the board; omit to keep the current one.
    """

    def change(chess):
        targets = chess.show_piece_moves(square)
        if caption is not None:
            chess.caption = caption
        return (f"Dots on {len(targets)} squares: {' '.join(targets)}." if targets
                else "That piece has no legal moves here (no dots drawn).")

    return await _apply(ctx, change)


@tool
async def show_attacks(square: str, caption: str | None = None, ctx: ToolContext = None) -> str:
    """Highlight every square the piece on `square` attacks (computed by the chess library).
    Use this, not `highlight`, to show what a piece attacks or controls.

    Args:
        square: The attacking piece's square, e.g. f3.
        caption: Optional line under the board; omit to keep the current one.
    """

    def change(chess):
        targets = chess.attacked_by(square)
        chess.highlight([square], "yellow")
        if targets:
            chess.highlight(targets, "red")
        if caption is not None:
            chess.caption = caption
        return f"The piece on {square.lower()} attacks: {' '.join(targets) or '(nothing)'}."

    return await _apply(ctx, change)


@tool
async def show_attackers(square: str, caption: str | None = None, ctx: ToolContext = None) -> str:
    """Highlight which pieces attack `square` (and so defend or threaten whatever stands there),
    computed by the chess library. White's attackers are tinted blue, Black's red.

    Args:
        square: The square in question, e.g. e5.
        caption: Optional line under the board; omit to keep the current one.
    """

    def change(chess):
        found = chess.attackers_of(square)
        chess.highlight([square], "yellow")
        for color, tint in (("white", "blue"), ("black", "red")):
            if found[color]:
                chess.highlight([label[-2:] for label in found[color]], tint)
        if caption is not None:
            chess.caption = caption
        return (f"{square.lower()} is attacked by White: {', '.join(found['white']) or 'nothing'}; "
                f"by Black: {', '.join(found['black']) or 'nothing'}.")

    return await _apply(ctx, change)


@tool
async def highlight(squares: list[str], color: Literal["yellow", "green", "red", "blue"] = "yellow",
                    caption: str | None = None, ctx: ToolContext = None) -> str:
    """Tint squares on the board for emphasis (a weak square, the centre, a pawn chain). This
    proves nothing: to show attacks or defenders use `show_attacks` / `show_attackers`.

    Args:
        squares: Square names, e.g. ["e4", "d5"].
        color: yellow, green, red or blue.
        caption: Optional line under the board; omit to keep the current one.
    """

    def change(chess):
        chess.highlight(squares, color)
        if caption is not None:
            chess.caption = caption
        return f"Highlighted {', '.join(s.lower() for s in squares)} in {color}."

    return await _apply(ctx, change)


@tool
async def clear_overlay(ctx: ToolContext = None) -> str:
    """Remove all arrows, dots and highlights (the position stays)."""
    await _apply(ctx, lambda chess: chess.clear_overlay())
    return "Arrows, dots and highlights cleared."


@tool(exclusive=True)
async def new_game(ctx: ToolContext = None) -> str:
    """Start a full game from the opening position. The user plays White; a very easy bot
    answers each move as Black automatically."""
    await _apply(ctx, lambda chess: chess.new_game())
    return "New game started. The user (White) moves first; the bot replies automatically."
