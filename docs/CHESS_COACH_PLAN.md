# BaseAgent chess coach: plan

Repo: github.com/xqzr2000/Russells_BaseAgent, branch `chess-coach` (from `main`, which already has the meadow UI).
Source: the chess agent in github.com/xqzr2000/cmu-11-768-assignment-1.
Status: 2026-10-15 — plan agreed, nothing built yet.

## Goal

- Saying "teach me chess" or "let's play chess" in the main chat hands the conversation to a chess coach. A chess board appears at the top of the dialog panel.
- The coach teaches in the dialog panel, directly below the board.
- The coach can show moves on the board: the next move, a hint, a threat, or a demonstration.
- Every level is covered, from never having played (101) to advanced coaching. In advanced mode the user plays a full game against the engine, and the coach reviews it afterwards.

## Decisions

- **Engine: Stockfish**, driven through python-chess's `chess.engine` (UCI). It is installed in the devcontainer (`apt-get install stockfish` in `post-create.sh`), not copied into the repo, because it's GPL.
- **Routing: handoff, not delegate.** The coordinator switches the chat over to the coach, and the coach talks to the user directly. `delegate` stays as it is for one-shot specialists like the data science agent.
- **The user always plays White.** The engine is always Black, and the board is always shown from White's side.
- **No persistence.** Chess state lives in memory, one game per chat. The coach places the user's level fresh in each new chat. A PGN download is the only thing saved.
- **Core rule: the LLM never decides chess facts.** Every move, hint, threat, evaluation and review verdict comes from python-chess or Stockfish through a tool. The coach chooses what to show and explains it.

## What comes over from the CMU assignment

| Reuse | Leave behind |
| --- | --- |
| `format_chess_state` (text board for the LLM) | Docker/Modal sandbox, HTTP client to a separate server |
| The `simulate_move` idea (inspect a position without changing the game) | `run_python` and the `select-move` skill |
| "Only one board-changing call per response" (`@tool(exclusive=True)`) | The single global game in `chess-app` |
| The CMU depth-2 bot, as the "very easy" opponent below Stockfish's minimum Elo | |
| The game-ending-move bug as a regression test (a move that ends the game returns the final state, with no engine reply) | |

## Architecture

### Backend: `src/baseagent/chess/`

- `game.py`, `ChessSession`. Holds the python-chess board, history, mode, level, the current overlay (arrows, highlights) and any pending exercise. One per chat, kept on the coach agent, so the code runs in-process with no server or sandbox.
- `engine.py`, a Stockfish wrapper, started in the coach's `setup()` and closed in `close()`. It provides:
  - `analyse(fen, multipv)`
  - `best_move(fen, elo)`
  - `move_loss(fen, move)`, the drop in winning chances
  - `review(game)`, which classifies each user move as an inaccuracy, mistake or blunder using Lichess-style thresholds (tuned in testing)
  - For opponent strength, Stockfish's `UCI_Elo` doesn't go much below about 1320, so lower levels use the CMU bot or Stockfish `Skill Level` 0.
- `puzzles.py`. A small committed subset of the Lichess puzzle database (CC0), about 5k puzzles filtered by theme and a rating range of roughly 600 to 2000.
- `board_view.py`. Builds the `board` event from a `ChessSession`.

### Handoff (new, reusable)

- `Session` keeps the coordinator plus an `active` agent. The coordinator gets a `handoff(agent, note)` tool; the coach gets `handoff_back(note)`.
- After a handoff, the coach answers in the same turn and the same stream, so the user doesn't have to repeat themselves. Later messages go straight to the coach.
- A new `handoff` event tells the UI which agent is active, so the board mounts and the header changes.
- Agents can opt in with `handoff_target = True`, the same pattern as `delegatable`.

### New stream event: `board`

```
board {
  fen, last_move, mode,                    # lesson | exercise | puzzle | game | review
  interactive,                             # can the user drag pieces now
  arrows: [{from, to, color}],             # green = good move, blue = idea, red = threat
  highlights: [{square, color}],
  dots: [square],                          # where a piece can go
  caption,                                 # one line shown under the board
  demo?: {steps: [{fen, move, caption}]}   # pre-validated sequence the UI animates
}
```

The session detail includes the latest board, so reopening a chat restores it. `web/src/types.ts` mirrors the event.

### Endpoints

- `POST /api/sessions/{id}/chess/move {move}`. The server checks legality, then acts by mode:
  - **game:** the engine replies immediately, with no LLM turn.
  - **exercise or puzzle:** the move is checked against the expected moves. A wrong move snaps back with a short nudge; a right move continues the line.
  - The response says whether the coach should speak (`needs_coach`). If so, the UI sends the move as a chat message (shown as a move chip), which starts a normal coach turn.
- `GET /api/sessions/{id}/chess/pgn` downloads the game.

## Showing instruction moves on the board

All of these are coach tools. Each one validates its moves with python-chess and emits a `board` event.

| Tool | What the user sees | Typical use |
| --- | --- | --- |
| `set_position(fen, caption)` | A new position (e.g. a lone knight on an empty board) | Start of a lesson or example |
| `show_move(move, kind)` | An arrow, coloured by kind: best, hint, idea, threat | "Here's the best move", "watch out for this" |
| `show_piece_moves(square)` | Dots on every square that piece can reach | Level 0: how each piece moves |
| `highlight(squares, color)` | Tinted squares | "These squares are attacked", weak squares |
| `demonstrate(moves, captions)` | The board animates the line move by move, with a caption per step and ◀ ▶ ⟲ controls, then returns to the lesson position | Checkmate patterns, a tactic, castling, an opening line |
| `ask_user_move(expected, prompt, hint)` | The board unlocks for the user; wrong tries snap back, a right move continues | "Find the fork", "Now you castle" |
| `clear_overlay()` | Removes arrows, dots and highlights | Between steps |

Moves the coach writes in backticks (e.g. `Nf3`) become clickable in the chat. Clicking one shows its arrow on the board.

## Coach agent: `agents/chess_coach.py`

- Board and overlay tools: the table above.
- Engine tools: `analyse(n)`, `check_user_move(move)`, `review_game()`.
- Game tools: `new_game(elo, coach_mode)`, `takeback()`, `resign()`, `load_puzzle(theme, rating)`, `handoff_back(note)`.
- The model sees `format_chess_state` plus the engine's top lines. It never sees a move suggestion that hasn't come from a tool.
- Curriculum as playbooks in `skills/chess/`:
  - `placement`: 2 or 3 questions, plus a puzzle if needed
  - `lesson-pieces`, `lesson-check-mate`, `lesson-special-moves`
  - `lesson-opening-principles`, `lesson-basic-mates`
  - `tactics`
  - `guided-game`
  - `game-review`
- Prompt rules: short messages, one idea at a time, always point at the board, end lessons with a "your turn" exercise.

## Levels

| Level | Content | Board behaviour |
| --- | --- | --- |
| 0 · Never played | Board and squares, each piece's moves, captures, check, checkmate, castling, en passant, promotion | Mini positions, dots, `ask_user_move` drills |
| 1 · Beginner | Piece values, opening principles, basic mates (queen, rook) | Demonstrations and guided positions, hints on |
| 2 · Improving | Tactics: forks, pins, skewers, discovered attacks, back-rank mates | Lichess puzzles by theme and rating |
| 3 · Guided play | Full game against an easy engine | Coach warns before a blunder ("Are you sure?"), hint button, short comments at key moments |
| 4 · Advanced | Full game against a chosen Elo, coach silent | Post-game review: 3 to 5 key moments, each stepped through with the better move's arrow and a demo of why |

The coach moves the user up or down within a chat based on how the exercises go.

## Frontend

- Dependencies: `react-chessboard`, `chess.js` (for smooth dragging only; the server decides legality).
- New components:
  - `ChessPanel.tsx`: the board, a caption line, and controls (hint, takeback, resign, new game, PGN, demo ◀ ▶ ⟲)
  - `ChessBoard.tsx`
- Layout: the chess panel is pinned at the top of the frosted dialog panel (board about 400 px), and messages scroll below it. It only mounts when the coach is active, so other chats are unchanged. On phones it shrinks and can collapse.
- Jack's avoid-areas don't change, since the board sits inside the panel.

## Build order

**M1 Board plumbing**
- [ ] `ChessSession` and the `board` event
- [ ] `ChessPanel` at the top of the panel
- [ ] The move endpoint
- [ ] A stub coach that sets positions, draws arrows and dots

**M2 Handoff and level 0 to 1 lessons**
- [ ] `handoff` / `handoff_back` and the `handoff` event
- [ ] `demonstrate` and `ask_user_move`
- [ ] Placement, plus the level 0 and 1 skills

**M3 Stockfish and play**
- [ ] Engine wrapper and devcontainer install
- [ ] `new_game` with Elo levels and the CMU bot below about 1320
- [ ] Hints and blunder warnings (level 3)

**M4 Review**
- [ ] `review_game`, key-moment selection, stepping through the review on the board (level 4)
- [ ] PGN download

**M5 Puzzles and polish**
- [ ] Lichess puzzle subset and the tactics skill (level 2)
- [ ] Clickable moves in the chat
- [ ] Mobile layout

## Testing

- Unit tests:
  - `ChessSession`: legality, game-over after the user's move (the CMU bug), takeback
  - The engine wrapper, using a fake engine
  - Exercise checking
  - The board event
- Server tests: the move endpoint in each mode, the handoff stream, restoring the board from the session detail.
- A coach check: across scripted lessons, every move the coach mentions in text was produced by a tool, and none is illegal.

## Later ideas

- Jack reacts to the game: sits by the board, celebrates a checkmate.
- Opt-in move sounds.
- Saved progress between chats (decided against for v1).
