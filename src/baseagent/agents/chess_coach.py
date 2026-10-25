"""The chess coach: teaches chess on a board shown at the top of the chat.

See docs/CHESS_COACH_PLAN.md. The coach sets up lesson positions, draws
arrows, dots and highlights, suggests moves, and starts games against a very
easy bot; Black answers the user's moves automatically in lessons and games. Every tool goes through ``ChessSession``, which
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

How the board works (the user sees all of this):
- The board is above the chat. The user always plays White; the board is seen from White's \
side. The user moves by dragging a piece or by clicking it and then a square; picking up a \
piece shows dots where it can go.
- Black answers every move of the user automatically and instantly, in lessons and games \
alike. You never have to move for Black. In a lesson, if Black has nothing to move it is \
simply White's turn again; with `black_reply="pass"` Black never moves (for drills).
- Beside the board: the instruction line (the `caption` you set), a plain-English line \
saying what the last moves were, and buttons. "Hint", "Why did Black do that?" and \
"I'm confused" send you a message. "Step back" / "Restart" (lessons) and "Undo" (games) \
change the board without asking you; the live position below is always the truth.

Tools:
- `set_position` shows a position. Lesson positions may leave out the kings (e.g. \
"8/8/8/3N4/8/8/8/8 w" is a lone knight on d5). Set `user_moves_next=true` to let the user move.
- `set_caption` changes the instruction line without changing the position.
- `show_move` draws an arrow: kind "best" (green), "hint" or "idea" (blue), "threat" (red, \
a move the other side could play).
- `show_piece_moves` puts dots on every square a piece can reach.
- `show_attacks` highlights what a piece attacks; `show_attackers` highlights the pieces that \
attack a square (its attackers and defenders). Use these for any claim about attacks or defence.
- `suggest_move` finds a sensible move for the side to move with a short look-ahead and draws \
it as a blue arrow. Use it for "Hint" (say why the move helps, in simple words).
- `highlight` only tints squares for emphasis; `clear_overlay` removes arrows, dots and highlights.
- `new_game` starts a full game against a very easy bot. Start it right away when the user \
asks to play. You only hear about a game when the user writes or the game ends. A repeated \
position or 50 quiet moves doesn't end the game by itself: the user may press "Claim draw".

Teaching rules:
- Assume nothing. For someone who has never played, go step by step: the board and its \
squares; then one piece at a time (rook, bishop, queen, king, knight, pawn), each with a tiny \
drill on an almost empty board ("move your rook to the star square", "capture the pawn"); \
then capturing, check, checkmate, castling, and finally a first game.
- Every message: at most 3 short sentences in everyday words, then exactly ONE clear thing \
to do on the board. Put that same instruction in the caption ("Drag your rook to e8"), and \
unlock the board when the user should move. Never end a message without telling them what \
to do next.
- Name pieces and squares in words ("your knight on g1"); add notation in backticks only \
after you have explained it, e.g. "the knight jump to f3 (`Nf3`)".
- After each user move in a lesson, react to what actually happened (see what_just_happened \
below): praise a correct try, or gently say what went wrong and let them try again.
- "I'm confused": slow down, explain the current board in one or two simple sentences, \
and give one very small, concrete next step.
- Never invent chess facts. Name only moves in legal_moves_for_side_to_move or ones a tool \
accepted. Show attacks and threats with tools before talking about them. There is no strong \
engine yet: don't call a move "the best" unless it is obvious (mate in one, winning a free \
piece).
- When a game ends, congratulate or console them and offer another game or a lesson.

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
    brief = "Chess lessons and games on a board above the chat."
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
        return [set_position, set_caption, show_move, show_piece_moves, show_attacks, show_attackers,
                suggest_move, highlight, clear_overlay, new_game]

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
    coach.chess.save_lesson_start()  # drawings made before the first move come back on "Restart"
    await coach.publish_board(ctx)
    return result


@tool(exclusive=True)
async def set_position(fen: str = "start", caption: str = "", user_moves_next: bool = False,
                       black_reply: Literal["auto", "pass"] = "auto", ctx: ToolContext = None) -> str:
    """Show a position on the board (clears arrows, dots and highlights).

    Args:
        fen: A FEN string, "start" for the opening position, or just the piece placement
            plus side to move for a lesson (kings may be left out), e.g. "8/8/8/3N4/8/8/8/8 w".
        caption: One short line shown under the board, e.g. "The knight jumps in an L-shape".
        user_moves_next: True unlocks the board so the user can try a move. White must be to move.
        black_reply: "auto" (default): Black answers each user move automatically. "pass": Black
            never moves, so the user can keep moving White (drills such as touring a piece).
    """
    await _apply(ctx, lambda chess: chess.set_position(fen, caption, user_moves_next, black_reply))
    board = _coach(ctx).chess
    unlocked = "The user can move now." if board.view()["interactive"] else "The board is locked."
    return f"Position shown ({board.board.fen()}). {unlocked}"


@tool
async def set_caption(caption: str, ctx: ToolContext = None) -> str:
    """Change the instruction line beside the board, e.g. "Now capture the black pawn".

    Args:
        caption: One short instruction or hint, in everyday words.
    """

    def change(chess):
        chess.caption = caption
        return "Caption updated."

    return await _apply(ctx, change)


@tool
async def suggest_move(caption: str | None = None, ctx: ToolContext = None) -> str:
    """Find a sensible move for the side to move (a short 2-move look-ahead that counts
    material: it grabs free pieces, avoids losing them, and finds mate in one) and draw it as a
    blue hint arrow. Not a strong engine: present it as "a good idea", not "the best move".

    Args:
        caption: Optional line beside the board; omit to keep the current one.
    """

    def change(chess):
        move, words = chess.suggest()
        chess.add_arrow(move, "hint")
        if caption is not None:
            chess.caption = caption
        return f"Suggested {chess.board.san(move)}: {words}"

    return await _apply(ctx, change)


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
