// The chess coach's board, pinned at the top of the dialog panel.
// The server owns the chess (src/baseagent/chess/game.py): this component draws the
// board it sends and reports the user's drags. A drag is shown at once, then replaced
// by the server's position (which adds castling rooks, promotions and the bot's reply).
import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { Chessboard } from "react-chessboard";
import type { BoardState } from "../types";

type Position = Record<string, { pieceType: string }>;

const FILES = "abcdefgh";

const ARROW_COLORS: Record<string, string> = {
  green: "rgba(38, 134, 74, 0.82)",
  blue: "rgba(59, 63, 158, 0.78)",
  red: "rgba(196, 44, 32, 0.82)",
};

const HIGHLIGHT_COLORS: Record<string, string> = {
  yellow: "rgba(255, 213, 46, 0.55)",
  green: "rgba(72, 170, 96, 0.5)",
  red: "rgba(226, 64, 52, 0.5)",
  blue: "rgba(84, 110, 230, 0.42)",
};

const LAST_MOVE = "rgba(242, 181, 68, 0.42)";
const DOT = "radial-gradient(circle, rgba(23, 35, 43, 0.3) 0 21%, transparent 23%)";
const RING = "radial-gradient(circle, transparent 0 62%, rgba(23, 35, 43, 0.3) 64% 76%, transparent 78%)";
const CHECK = "radial-gradient(circle, rgba(226, 40, 30, 0.85) 0, rgba(226, 40, 30, 0.35) 45%, transparent 72%)";

/** Piece placement from a FEN, in react-chessboard's format ({e2: {pieceType: "wP"}}). */
export function fenToPosition(fen: string): Position {
  const position: Position = {};
  fen
    .split(" ")[0]
    .split("/")
    .forEach((row, i) => {
      let file = 0;
      for (const ch of row) {
        if (/\d/.test(ch)) {
          file += Number(ch);
          continue;
        }
        const color = ch === ch.toUpperCase() ? "w" : "b";
        position[`${FILES[file]}${8 - i}`] = { pieceType: color + ch.toUpperCase() };
        file += 1;
      }
    });
  return position;
}

function movePairs(history: string[]): string[] {
  const pairs: string[] = [];
  for (let i = 0; i < history.length; i += 2) {
    pairs.push(`${i / 2 + 1}. ${history[i]}${history[i + 1] ? ` ${history[i + 1]}` : ""}`);
  }
  return pairs;
}

const PROMOTIONS = [
  { piece: "q", label: "Queen", glyph: "♕" },
  { piece: "r", label: "Rook", glyph: "♖" },
  { piece: "b", label: "Bishop", glyph: "♗" },
  { piece: "n", label: "Knight", glyph: "♘" },
];

interface Props {
  board: BoardState;
  busy: boolean; // the coach is mid-turn, so the board is locked
  /** Send the move (UCI, e.g. "e7e8n") to the server; resolves to false if it was rejected. */
  onMove: (uci: string) => Promise<boolean>;
  onClaimDraw: () => void;
  notice: string | null; // a short message, e.g. why a move was rejected
}

export function ChessPanel({ board, busy, onMove, onClaimDraw, notice }: Props) {
  const [pending, setPending] = useState<{ from: string; to: string } | null>(null);
  const [promotion, setPromotion] = useState<{ from: string; to: string } | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const reduceMotion = useMemo(
    () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches,
    [],
  );

  // A new position from the server replaces the optimistic one.
  useEffect(() => {
    setPending(null);
    setPromotion(null);
  }, [board]);

  const submit = (from: string, to: string, piece = "") => {
    setPending({ from, to });
    onMove(from + to + piece).then((ok) => {
      if (!ok) setPending(null); // rejected: the piece goes back
    });
  };

  const position = useMemo(() => {
    const placed = fenToPosition(board.fen);
    if (pending && placed[pending.from]) {
      placed[pending.to] = placed[pending.from];
      delete placed[pending.from];
    }
    return placed;
  }, [board.fen, pending]);

  const squareStyles = useMemo(() => {
    const styles: Record<string, CSSProperties> = {};
    const add = (square: string, style: CSSProperties) => {
      styles[square] = { ...styles[square], ...style };
    };
    const last = pending ? `${pending.from}${pending.to}` : board.last_move;
    if (last) {
      add(last.slice(0, 2), { backgroundColor: LAST_MOVE });
      add(last.slice(2, 4), { backgroundColor: LAST_MOVE });
    }
    for (const h of board.highlights) add(h.square, { backgroundColor: HIGHLIGHT_COLORS[h.color] });
    for (const square of board.dots) add(square, { backgroundImage: position[square] ? RING : DOT });
    if (board.in_check && !pending) {
      const king = board.turn === "white" ? "wK" : "bK";
      const square = Object.keys(position).find((s) => position[s].pieceType === king);
      if (square) add(square, { backgroundImage: CHECK });
    }
    return styles;
  }, [board, pending, position]);

  const canMove = board.interactive && !busy && !pending && !promotion;
  const status = (() => {
    if (busy) return "The coach is thinking...";
    if (pending) return "Checking your move...";
    if (board.game_over) return "Game over";
    if (promotion) return "Choose a piece for your pawn";
    if (canMove) return "Your move: drag a white piece";
    return board.mode === "game" ? "Waiting..." : "Watch the board";
  })();
  const pairs = movePairs(board.history);

  return (
    <section className={`chess-panel ${collapsed ? "collapsed" : ""}`} aria-label="Chess board">
      {collapsed ? (
        <button className="chess-show" onClick={() => setCollapsed(false)}>
          Show the board
          <span className="muted small">{status}</span>
        </button>
      ) : (
        <>
          <div className="chess-board">
            <Chessboard
              options={{
                id: "coach-board",
                position,
                boardOrientation: "white",
                arrows: board.arrows.map((a) => ({
                  startSquare: a.from,
                  endSquare: a.to,
                  color: ARROW_COLORS[a.color] ?? ARROW_COLORS.blue,
                })),
                squareStyles,
                allowDrawingArrows: false,
                allowDragging: canMove,
                showAnimations: !reduceMotion,
                animationDurationInMs: 220,
                lightSquareStyle: { backgroundColor: "#f4eedb" },
                darkSquareStyle: { backgroundColor: "#8db36a" },
                boardStyle: { borderRadius: "10px", overflow: "hidden", boxShadow: "0 4px 14px rgba(28, 52, 12, 0.18)" },
                dropSquareStyle: { boxShadow: "inset 0 0 0 3px rgba(59, 63, 158, 0.55)" },
                canDragPiece: ({ piece }) => canMove && piece.pieceType.startsWith("w"),
                onPieceDrop: ({ piece, sourceSquare, targetSquare }) => {
                  if (!canMove || !targetSquare || targetSquare === sourceSquare) return false;
                  // A white pawn reaching the last rank: ask which piece it becomes.
                  if (piece.pieceType === "wP" && targetSquare.endsWith("8")) {
                    setPromotion({ from: sourceSquare, to: targetSquare });
                    return false;
                  }
                  submit(sourceSquare, targetSquare);
                  return true;
                },
              }}
            />
            {promotion && (
              <div className="promotion" role="dialog" aria-label="Promote the pawn to">
                <span className="promotion-title">Promote to</span>
                <div className="promotion-choices">
                  {PROMOTIONS.map((p) => (
                    <button
                      key={p.piece}
                      className="promotion-choice"
                      onClick={() => {
                        const { from, to } = promotion;
                        setPromotion(null);
                        submit(from, to, p.piece);
                      }}
                      title={p.label}
                      aria-label={p.label}
                    >
                      {p.glyph}
                    </button>
                  ))}
                </div>
                <button className="promotion-cancel" onClick={() => setPromotion(null)}>
                  Cancel
                </button>
              </div>
            )}
          </div>
          <div className="chess-side">
            <div className="chess-head">
              <span className={`chess-mode ${board.mode}`}>{board.mode === "game" ? "Game · you're White" : "Lesson"}</span>
              <button className="icon-btn subtle-visible" onClick={() => setCollapsed(true)} aria-label="Hide the board" title="Hide the board">
                <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden>
                  <path d="M5 12.5 10 7.5l5 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            {board.caption && <p className="chess-caption">{board.caption}</p>}
            <p className={`chess-status ${canMove ? "your-move" : ""}`} aria-live="polite">
              {status}
            </p>
            {notice && (
              <p className="chess-notice" role="alert">
                {notice}
              </p>
            )}
            {board.draw_claimable && !busy && (
              <button className="btn ghost small-btn" onClick={onClaimDraw}>
                Claim draw
              </button>
            )}
            {pairs.length > 0 && (
              <ol className="chess-moves" aria-label="Moves">
                {pairs.map((pair) => (
                  <li key={pair}>{pair}</li>
                ))}
              </ol>
            )}
            <p className="chess-legend muted small">
              <span className="swatch green" /> best <span className="swatch blue" /> idea <span className="swatch red" /> threat
            </p>
          </div>
        </>
      )}
    </section>
  );
}
