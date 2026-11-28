# Chess coach

Say *teach me chess* or *let's play chess* in the main chat. Jack hands the chat to the
chess coach, a board appears at the top of the dialog panel, and the coach teaches in the
chat below it. Built for someone who has never played: Russell tests it as a true
beginner.

Code: [`src/baseagent/agents/chess_coach/`](../../src/baseagent/agents/chess_coach/).
Tests: [`tests/agents/test_chess_coach.py`](../../tests/agents/test_chess_coach.py).

## Rules it is built on

- **The model never decides chess facts.** Every move, hint and threat comes from
  python-chess or the engine through a tool. The coach chooses what to show and explains it.
- **Handoff, not delegate.** The coach talks to the user over many messages, so Jack
  switches the whole chat to it (`handoff`), and the coach returns it with
  `handoff_to_jack` when the user asks for Jack. See
  [adding an agent](../adding-an-agent.md#handoff-a-specialist-that-talks-to-the-user-directly).
- **The user always plays White,** and the board is shown from White's side. Black
  replies automatically after every move, in lessons too.
- **No saved progress.** One board per chat, kept in memory; the coach places the user's
  level fresh in each chat.

## How it works

| Part | File | What it does |
| --- | --- | --- |
| Coach agent | `agent.py` | Prompt for absolute beginners (one instruction per message, mirrored in the board caption) and the board tools |
| Board | `game.py` | `ChessSession`: the position, history, mode (`lesson` or `game`), arrows, highlights and the caption; builds the `board` event |
| Engine | `engine.py` | One shared Stockfish process with short searches; a reduced skill level for beginners |
| Practice opponent | `bot.py` | A 2-ply material search in the style of the CMU 11-768 bot, for drills without kings and when Stockfish is missing |

Coach tools: `set_position`, `set_caption`, `suggest_move`, `show_move` (hint, idea or
threat arrow), `show_piece_moves`, `show_attacks`, `show_attackers`, `highlight`,
`clear_overlay`, `new_game`, `handoff_to_jack`.

Board endpoints, answered without a model call where possible
(`src/baseagent/server/app.py`):

- `POST /api/sessions/{id}/chess/move`: the user's move, checked for legality
- `POST .../chess/hint`: a suggested move with its arrow
- `POST .../chess/advance`: play one engine move for the side to move
- `POST .../chess/rewind`: step back, or restart the lesson position

The UI (`web/src/components/ChessPanel.tsx`) draws the board, plain-English narration
of the last moves, and the helper buttons: Hint, *Why did Black do that?*,
*I'm confused*, Step back, Restart lesson, Undo. It never decides legality itself.

Stockfish is installed in the dev container and is not committed (it is GPL). Locally,
put it on your `PATH` or set `STOCKFISH_PATH`; without it, the practice opponent plays.

## Not built yet

From the original plan (October 2026), still open:

- `demonstrate` (animate a line with captions) and `ask_user_move` ("find the fork")
- Placement questions and a structured beginner course with mini-games and progress
- Guided games: instant warnings for hanging pieces, the coach speaking at key moments
- Game review with key moments stepped through on the board, and PGN download
- Lichess puzzles by theme and rating, clickable moves in the chat, a mobile layout

The coach is a fun extra, not on the critical path of [the roadmap](../roadmap.md).
