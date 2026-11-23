"""The chess coach explains lessons and positions beside the teaching board."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Literal

from baseagent.agent import BaseAgent, TurnResult
from baseagent.agents import register_agent
from baseagent.chess import ChessError, ChessSession
from baseagent.events import Board, Usage
from baseagent.handoff import COORDINATOR_NAME
from baseagent.tools import ToolContext, ToolError, ToolResult, tool

QUICK_HINT = re.compile(
    r"(?:hint(?:,? please)?(?:: what could I do next)?|(?:(?:can|could) you )?give me a hint|"
    r"(?:what (?:could|should) I (?:do|play)|what(?:'s| is) (?:the|my) (?:best|next) move)(?: next)?)[.!?]*",
    re.IGNORECASE,
)

ASK_FOR_JACK = re.compile(
    r"^(?:please[\s,]+)?(?:(?:(?:(?:let|et) me|let's|(?:can|could|may) I|"
    r"I (?:want|would like) to|I'd like to)\s+)?(?:talk|speak|chat)\s+(?:to|with)|"
    r"(?:go |switch )?back to|switch to|(?:send|pass|take) me (?:back )?to)"
    r"\s+(?:Jack(?!'s\b)|the coordinator)\b",
    re.IGNORECASE,
)
LEAVE_CHESS = re.compile(
    r"^(?:please[\s,]+)?(?:(?:let's )?(?:stop|quit|leave|exit) (?:playing )?chess|no more chess|"
    r"(?:we're|we are|I'm|I am) (?:done (?:with|playing) chess|not playing chess any ?more)|"
    r"I don't want to play chess any ?more)\b",
    re.IGNORECASE,
)
THANKS = {"please", "thanks", "thank you"}


def jack_follow_up(message: str) -> str | None:
    """Return the topic to pass to Jack, an empty string for a switch, or None."""
    remaining = message.strip()
    asked_for_jack = False
    leaving_chess = False
    while remaining:
        command = remaining.replace("’", "'")
        if match := ASK_FOR_JACK.match(command):
            asked_for_jack = True
        elif match := LEAVE_CHESS.match(command):
            leaving_chess = True
        else:
            break
        remaining = remaining[match.end():].lstrip(" \t\r\n,.!?:;–—")
        remaining = re.sub(r"^(?:and|then)\s+", "", remaining, flags=re.IGNORECASE)
    if remaining.lower().rstrip(".!?") in THANKS:
        remaining = ""
    if asked_for_jack:
        return re.sub(r"^about\s+", "", remaining, flags=re.IGNORECASE)
    return "" if leaving_chess and not remaining else None


def request_jack(agent: BaseAgent, follow_up: str) -> str:
    agent.state["handoff"] = {"to": COORDINATOR_NAME, "message": follow_up}
    return "I'll hand you back to Jack."


SYSTEM_PROMPT = """\
You are the Chess coach in Russell's BaseAgent chat room, a warm, patient chess teacher.
Jack is the coordinator and a separate agent. If the user asks to speak to Jack or wants to stop \
discussing chess, call the handoff_to_jack tool immediately. The tool switches the active agent in this chat. \
Use board tools and teaching instructions only for users who want to continue with chess.

How the board works (the user sees all of this):
- The board is above the chat. The user always plays White; the board is seen from White's \
side. The user moves by dragging a piece or by clicking it and then a square; picking up a \
piece shows dots where it can go.
- Stockfish chooses moves for either side. It replies as Black after the user moves. The user \
can select “Play next move” to watch one engine move for the current side. In single-piece drills, \
Black may pass and the practice engine can suggest a legal move without kings.
- Beside the board, the user always sees your latest explanation, the instruction (caption), \
what the last moves did, and controls. “Hint” shows the engine's move directly. “Why did Black do that?” \
and “I'm confused” ask you to explain the current position.
- “Step back”, “Restart” and “Undo” change the board. Read the live position below before answering.

Tools:
- `set_position` shows a position. Lesson positions may leave out the kings (for example, \
"8/8/8/3N4/8/8/8/8 w" is a lone knight on d5). Set `user_moves_next=true` to let the user move.
- `set_caption` changes the instruction beside the board without changing the position.
- `show_move` draws arrows for illustration or for an example the user named. Use `suggest_move` \
whenever you need a recommended move or continuation.
- `show_piece_moves` puts dots on every square a piece can reach.
- `show_attacks` highlights what a piece attacks; `show_attackers` highlights the pieces that \
attack a square (its attackers and defenders). Use these for any claim about attacks or defence.
- `suggest_move` gets the engine's move for the side to move and draws it. Use the move and \
any reply it returns as the facts you explain. Present possible replies as possibilities.
- `highlight` only tints squares for emphasis; `clear_overlay` removes arrows, dots and highlights.
- `new_game` starts a full game with Stockfish as Black. Start it right away when the user \
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
- Explain the position, the engine's suggestion and what the learner should try next in simple \
words. Never choose a move or invent a line yourself. Use the board tools to establish attacks, \
defenders or threats, and only name engine moves that came from `suggest_move` or the live board.
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
        self._hint_turn = False

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(state=self.chess.describe())

    async def run(self, user_message: str, via: str | None = None) -> TurnResult:
        follow_up = jack_follow_up(user_message)
        if follow_up is not None:
            return await self.reply_without_model(user_message, request_jack(self, follow_up), via=via)
        if not QUICK_HINT.fullmatch(user_message.strip()):
            return await super().run(user_message, via=via)
        try:
            hint = await asyncio.to_thread(self.chess.calculate_hint)
        except ChessError as exc:
            text = str(exc)
        else:
            self.chess.show_hint(hint)
            text = " ".join(part for part in (hint.explanation, hint.reply) if part)
        self._hint_turn = True
        try:
            return await self.reply_without_model(user_message, text, via=via)
        finally:
            self._hint_turn = False

    async def _finish(self, turn: int, final_text: str, steps: int, stop_reason: str, usage: Usage) -> None:
        if stop_reason == "final" and final_text.strip() and not self.state.get("handoff"):
            if not self._hint_turn:
                self.chess.explanation = final_text.strip()
            await self.publish_board()
        await super()._finish(turn, final_text, steps, stop_reason, usage)

    def get_tools(self):
        return [set_position, set_caption, show_move, show_piece_moves, show_attacks, show_attackers,
                suggest_move, highlight, clear_overlay, new_game, handoff_to_jack]

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
def handoff_to_jack(follow_up: str = "", ctx: ToolContext = None) -> ToolResult:
    """Switch the chat to Jack, the coordinator, when the user wants to leave chess.

    Args:
        follow_up: The user's new question or topic for Jack, if any. Omit for a plain switch.
    """
    return ToolResult(request_jack(_coach(ctx), follow_up.strip()), end_turn=True)


@tool(exclusive=True)
async def set_position(fen: str = "start", caption: str = "", user_moves_next: bool = False,
                       black_reply: Literal["auto", "pass"] = "auto", ctx: ToolContext = None) -> str:
    """Show a position on the board (clears arrows, dots and highlights).

    Args:
        fen: A FEN string, "start" for the opening position, or just the piece placement
            plus side to move for a lesson (kings may be left out), e.g. "8/8/8/3N4/8/8/8/8 w".
        caption: One short line shown beside the board, e.g. "The knight jumps in an L-shape".
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


@tool(exclusive=True)
async def suggest_move(caption: str | None = None, ctx: ToolContext = None) -> str:
    """Get and draw the engine's move for the side to move and a possible reply.

    Args:
        caption: Optional short instruction beside the board.
    """
    coach = _coach(ctx)
    try:
        hint = await asyncio.to_thread(coach.chess.calculate_hint)
    except ChessError as exc:
        raise ToolError(str(exc)) from exc

    def change(chess):
        chess.show_hint(hint)
        if caption is not None:
            chess.caption = caption
        source = "Stockfish" if hint.source == "stockfish" else "Practice engine"
        return f"{source} suggests {hint.san}. {hint.explanation} {hint.reply or ''}".strip()

    return await _apply(ctx, change)


@tool
async def show_move(move: str, kind: Literal["hint", "idea", "threat"] = "idea",
                    caption: str | None = None, keep_other_arrows: bool = True,
                    ctx: ToolContext = None) -> str:
    """Draw an arrow for a move without playing it.

    Args:
        move: The move in SAN (Nf3) or UCI (g1f3).
        kind: hint = blue (an exercise move for the side to move); threat = red (a move
            the other side could play); idea = blue (either side).
        caption: Optional line beside the board; omit to keep the current one.
        keep_other_arrows: False removes the arrows already drawn.
    """

    side = {"threat": "opponent", "hint": "to_move"}.get(kind, "either")

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
        caption: Optional line beside the board; omit to keep the current one.
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
        caption: Optional line beside the board; omit to keep the current one.
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
        caption: Optional line beside the board; omit to keep the current one.
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
        caption: Optional line beside the board; omit to keep the current one.
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
    """Start a game with the user as White and an automatic engine reply as Black."""
    await _apply(ctx, lambda chess: chess.new_game())
    return "New game started. The user (White) moves first; the engine replies automatically."
