"""Chess coach, milestone 1: the board, the coach's board tools, and the move endpoint."""

from __future__ import annotations

import json

import chess
import pytest
from fastapi.testclient import TestClient

from baseagent.agents.chess_coach import ChessCoachAgent
from baseagent.chess import ChessError, ChessSession
from baseagent.chess.bot import choose_move
from baseagent.llm import FakeLLM, ScriptedLLM
from baseagent.server.app import create_app


def call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if data:
            events.append(json.loads("".join(data)))
    return events


# ------------------------------------------------------------------ ChessSession

def test_lesson_position_without_kings_shows_knight_moves():
    board = ChessSession()
    board.set_position("8/8/8/3N4/8/8/8/8 w", "Knight", user_moves_next=True)
    assert board.show_piece_moves("d5") == ["c3", "e3", "b4", "f4", "b6", "f6", "c7", "e7"]
    view = board.view()
    assert view["interactive"] and view["dots"] and not view["game_over"]
    assert {"square": "d5", "color": "yellow"} in view["highlights"]


def test_lesson_move_locks_board_and_asks_coach():
    board = ChessSession()
    board.set_position("8/8/8/3N4/8/8/8/8 w", user_moves_next=True)
    outcome = board.user_move("d5f6")
    assert outcome.needs_coach and outcome.message() == "I played Nf6."
    assert not board.view()["interactive"] and board.view()["last_move"] == "d5f6"
    with pytest.raises(ChessError, match="locked"):
        board.user_move("f6g8")


def test_rejects_bad_positions_and_moves():
    board = ChessSession()
    with pytest.raises(ChessError):
        board.set_position("not a fen")
    with pytest.raises(ChessError, match="White to move"):
        board.set_position("8/8/8/3n4/8/8/8/8 b", user_moves_next=True)
    with pytest.raises(ChessError, match="pawns on backrank"):
        board.set_position("P7/8/8/8/8/8/8/4K2k w")
    board.new_game()
    with pytest.raises(ChessError, match="not a legal move"):
        board.user_move("e2e5")


def test_find_move_accepts_threats_for_the_other_side():
    board = ChessSession()
    board.set_position("4k3/3q4/8/8/8/8/8/4K3 w")
    move, side_board = board.find_move("Qd1")  # Black's threat while White is to move
    assert side_board.turn == chess.BLACK and move.uci() == "d7d1"
    board.add_arrow(move, "threat")
    assert board.view()["arrows"] == [{"from": "d7", "to": "d1", "color": "red"}]
    with pytest.raises(ChessError, match="either side"):
        board.find_move("Qh8")


def test_game_bot_replies_and_san_history():
    board = ChessSession()
    board.new_game()
    outcome = board.user_move("e4")
    assert outcome.reply_san and not outcome.needs_coach
    view = board.view()
    assert view["turn"] == "white" and view["interactive"] and view["history"][0] == "e4"
    assert len(view["history"]) == 2


def test_game_ending_user_move_gets_no_bot_reply():
    # The CMU chess-app bug: a mating move must return the final state, not crash.
    board = ChessSession()
    board.new_game()
    board.board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    outcome = board.user_move("f7g7")
    assert outcome.game_over and outcome.reply_san is None and outcome.needs_coach
    assert board.view()["game_over"] and not board.view()["interactive"]
    assert "you win" in board.view()["caption"]


def test_bot_takes_a_free_queen_and_finds_mate():
    assert choose_move(chess.Board("r3k3/8/8/8/8/8/8/Q3K3 b - - 0 1")).uci() == "a8a1"
    assert choose_move(chess.Board("6k1/8/8/8/8/8/r5PP/7K b - - 0 1")).uci() == "a2a1"


def test_describe_lists_legal_moves_for_the_model():
    board = ChessSession()
    board.new_game()
    text = board.describe()
    assert "side_to_move: white (the user)" in text and "Nf3" in text and "e4" in text


# ------------------------------------------------------------------ coach + server

def client_with(llm, tmp_path) -> TestClient:
    return TestClient(create_app(llm=llm, log_dir=tmp_path, workspace_dir=tmp_path / "workspace"))


def test_coach_tools_publish_board_events_and_detail(tmp_path):
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [
            call("set_position", {"fen": "8/8/8/3N4/8/8/8/8 w", "user_moves_next": True,
                                  "caption": "A knight"}, "c1"),
            call("show_piece_moves", {"square": "d5"}, "c2"),
        ]},
        {"content": "", "tool_calls": [call("show_move", {"move": "Nf6", "kind": "hint"}, "c3"),
                                       call("show_move", {"move": "e2e4"}, "c4")]},
        {"content": "The dots show where the knight can go."},
    ])
    with client_with(llm, tmp_path) as client:
        session = client.post("/api/sessions", json={"agent": "chess_coach",
                                                     "settings": {"model": "gpt-test"}}).json()
        assert session["board"]["fen"].startswith("rnbqkbnr")  # a fresh board before any lesson
        events = parse_sse(client.post(f"/api/sessions/{session['id']}/messages",
                                       json={"content": "How does the knight move?"}).text)
        boards = [e["board"] for e in events if e["type"] == "board"]
        assert len(boards) == 3  # set_position, show_piece_moves, the legal arrow
        assert boards[-1]["arrows"] == [{"from": "d5", "to": "f6", "color": "blue"}]
        errors = [e for e in events if e["type"] == "tool_end" and e["is_error"]]
        assert len(errors) == 1 and "not a legal move" in errors[0]["content"]

        # The model saw the live position in its system prompt.
        system = llm.requests[-1]["messages"][0]["content"]
        assert "8/8/8/3N4/8/8/8/8 w" in system and "Nf6" in system

        detail = client.get(f"/api/sessions/{session['id']}").json()
        assert detail["board"]["interactive"] and len(detail["board"]["dots"]) == 8

        moved = client.post(f"/api/sessions/{session['id']}/chess/move", json={"move": "d5e7"}).json()
        assert moved["needs_coach"] and moved["message"] == "I played Ne7."
        assert moved["board"]["last_move"] == "d5e7" and not moved["board"]["interactive"]
        again = client.post(f"/api/sessions/{session['id']}/chess/move", json={"move": "e7g8"})
        assert again.status_code == 422 and "locked" in again.json()["detail"]


def test_game_moves_get_bot_replies_without_the_model(tmp_path):
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [call("new_game", {})]},
        {"content": "Your move!"},
    ])
    with client_with(llm, tmp_path) as client:
        session = client.post("/api/sessions", json={"agent": "chess_coach"}).json()
        client.post(f"/api/sessions/{session['id']}/messages", json={"content": "Let's play"})
        calls_before = len(llm.requests)
        moved = client.post(f"/api/sessions/{session['id']}/chess/move", json={"move": "e2e4"}).json()
        assert not moved["needs_coach"] and moved["board"]["interactive"]
        assert len(moved["board"]["history"]) == 2 and len(llm.requests) == calls_before
        bad = client.post(f"/api/sessions/{session['id']}/chess/move", json={"move": "e4e6"})
        assert bad.status_code == 422


def test_move_endpoint_needs_a_chess_chat(tmp_path):
    with client_with(ScriptedLLM([]), tmp_path) as client:
        session = client.post("/api/sessions", json={"agent": "general"}).json()
        response = client.post(f"/api/sessions/{session['id']}/chess/move", json={"move": "e2e4"})
        assert response.status_code == 404
        assert "board" not in client.get(f"/api/sessions/{session['id']}").json()


def test_fake_model_walks_through_the_board(tmp_path):
    with client_with(FakeLLM(), tmp_path) as client:
        session = client.post("/api/sessions", json={"agent": "chess_coach"}).json()
        events = parse_sse(client.post(f"/api/sessions/{session['id']}/messages",
                                       json={"content": "Teach me chess"}).text)
        board = [e for e in events if e["type"] == "board"][-1]["board"]
        assert board["dots"] and board["interactive"]
        assert events[-1]["final_text"].startswith("(fake model)")


def test_coach_is_registered():
    assert ChessCoachAgent.name == "chess_coach" and ChessCoachAgent.examples
