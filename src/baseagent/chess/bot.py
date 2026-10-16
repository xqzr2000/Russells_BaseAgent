"""A very easy chess opponent: a deterministic 2-ply search on material.

It plays like the bot in the CMU 11-768 chess app (it never hangs a piece to
an immediate recapture and finds mate in one, but sees nothing deeper), which
makes it a fair first opponent for a beginner. Stockfish replaces it for the
stronger levels.
"""

from __future__ import annotations

import chess

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
CENTER = (chess.D4, chess.E4, chess.D5, chess.E5)
MATE = 100_000


def evaluate(board: chess.Board) -> int:
    """Score from White's point of view, in centipawns."""

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0
        return MATE if outcome.winner == chess.WHITE else -MATE
    score = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        score += sign * PIECE_VALUES[piece.piece_type]
        if square in CENTER:
            score += sign * 12
    return score


def _search(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    if depth == 0 or board.outcome(claim_draw=True) is not None:
        return evaluate(board)
    maximizing = board.turn == chess.WHITE
    best = -MATE * 2 if maximizing else MATE * 2
    for move in sorted(board.legal_moves, key=lambda m: m.uci()):
        board.push(move)
        score = _search(board, depth - 1, alpha, beta)
        board.pop()
        if maximizing:
            best = max(best, score)
            alpha = max(alpha, score)
        else:
            best = min(best, score)
            beta = min(beta, score)
        if beta <= alpha:
            break
    return best


def choose_move(board: chess.Board, depth: int = 2) -> chess.Move:
    """The bot's move for the side to move. Ties go to the first move in UCI order."""

    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    if not moves:
        raise ValueError("No legal moves: the game is over.")
    maximizing = board.turn == chess.WHITE
    best_move, best_score = moves[0], None
    work = board.copy()
    for move in moves:
        work.push(move)
        score = _search(work, depth - 1, -MATE * 2, MATE * 2)
        work.pop()
        if best_score is None or (score > best_score if maximizing else score < best_score):
            best_move, best_score = move, score
    return best_move
