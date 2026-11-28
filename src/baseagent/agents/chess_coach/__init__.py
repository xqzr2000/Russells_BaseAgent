"""The chess coach: a handoff agent that teaches beside a board.

* ``agent``: ``ChessCoachAgent``, its prompt and board tools
* ``game``: ``ChessSession``, the board, moves and what the coach draws
* ``engine``: the Stockfish wrapper
* ``bot``: the very easy opponent below Stockfish's lowest level
"""

from baseagent.agents.chess_coach.agent import ChessCoachAgent

__all__ = ["ChessCoachAgent"]
