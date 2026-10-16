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


def test_lesson_move_keeps_the_board_playable_and_asks_coach():
    board = ChessSession()
    board.set_position("8/8/8/3N4/8/8/8/8 w", user_moves_next=True)
    outcome = board.user_move("d5f6")
    assert outcome.needs_coach and outcome.message() == "I moved my knight from d5 to f6 (Nf6)."
    # Black has nothing to move, so it is White's turn again rather than a frozen board.
    view = board.view()
    assert view["interactive"] and view["turn"] == "white" and view["last_move"] == "d5f6"
    assert "your turn again" in view["narration"] and view["history"] == ["Nf6"]
    board.user_move("f6g8")
    board.user_can_move = False  # the coach can still lock the board
    with pytest.raises(ChessError, match="locked"):
        board.user_move("g8f6")


def test_black_answers_in_lessons_unless_told_to_pass():
    board = ChessSession()
    board.set_position("4k3/pppp4/8/8/8/8/8/R3K3 w", "Capture a pawn", user_moves_next=True)
    outcome = board.user_move("a1a7")
    assert outcome.reply_san and board.view()["turn"] == "white"
    assert outcome.message().startswith("I captured Black's pawn on a7 with my rook (Rxa7). Black ")
    assert board.narration.startswith("You captured Black's pawn on a7 with your rook. Black ")

    board.set_position("4k3/pppp4/8/8/8/8/8/R3K3 w", user_moves_next=True, black_reply="pass")
    outcome = board.user_move("a1a3")
    assert outcome.reply_san is None and board.view()["turn"] == "white"
    assert board.board.fen().startswith("4k3/pppp4/8/8/8/R7/8/4K3 w")


def test_rewind_steps_back_and_restarts_the_lesson():
    board = ChessSession()
    board.set_position("4k3/pppp4/8/8/8/8/8/R3K3 w", "Capture a pawn", user_moves_next=True)
    board.show_piece_moves("a1")
    board.save_lesson_start()  # what the coach's tools do after drawing on a fresh lesson
    board.user_move("a1a7")
    board.user_move("a7b7") if board.board.piece_at(chess.B7) else board.user_move("a7a8")
    assert len(board.history_san()) == 4 and board.view()["can_step_back"]
    board.rewind("back")
    assert len(board.history_san()) == 2 and board.view()["interactive"]
    board.rewind("back")
    # Back at the lesson start: the coach's dots and caption come back too.
    assert board.history_san() == [] and board.view()["dots"] and board.caption == "Capture a pawn"
    with pytest.raises(ChessError, match="no move to take back"):
        board.rewind("back")
    board.user_move("a1a2")
    board.rewind("start")
    view = board.view()
    assert view["history"] == [] and not view["can_restart"] and view["dots"]
    board.new_game()
    with pytest.raises(ChessError, match="only works in a lesson"):
        board.rewind("start")
    board.user_move("e2e4")
    board.rewind("back")
    assert board.board.fen() == chess.STARTING_FEN and board.view()["interactive"]


def test_plain_words_for_special_moves():
    from baseagent.chess.game import describe_move

    castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert describe_move(castle, chess.Move.from_uci("e1g1"), "you").startswith("You castled kingside")
    promo = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    assert "became a knight" in describe_move(promo, chess.Move.from_uci("e7e8n"), "you")
    mate = chess.Board("6k1/8/8/8/8/8/r5PP/7K b - - 0 1")
    assert describe_move(mate, chess.Move.from_uci("a2a1"), "Black").endswith("(checkmate!).")


def test_suggest_move_is_legal_and_described():
    board = ChessSession()
    board.set_position("r3k3/8/8/8/8/8/8/Q3K3 w - - 0 1", user_moves_next=True)
    move, words = board.suggest()
    assert move.uci() == "a1a8" and words == "Capture Black's rook on a8 with your queen (check!)."


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
        assert moved["needs_coach"] and moved["message"] == "I moved my knight from d5 to e7 (Ne7)."
        assert moved["board"]["last_move"] == "d5e7" and moved["board"]["interactive"]
        bad = client.post(f"/api/sessions/{session['id']}/chess/move", json={"move": "e7e8"})
        assert bad.status_code == 422 and "not a legal move" in bad.json()["detail"]

        back = client.post(f"/api/sessions/{session['id']}/chess/rewind", json={"to": "back"}).json()
        assert back["board"]["fen"].startswith("8/8/8/3N4/8/8/8/8 w") and not back["needs_coach"]
        assert back["board"]["dots"]  # back at the start, the lesson's drawings return
        again = client.post(f"/api/sessions/{session['id']}/chess/rewind", json={"to": "start"}).json()
        assert again["board"]["history"] == []


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


# ------------------------------------------------------------------ review fixes

def test_threat_arrow_uses_the_opponent_when_both_sides_can_play_the_move():
    board = ChessSession()
    board.set_position("4k1n1/8/8/8/4N3/8/8/4K3 w - - 0 1")  # Ne4 and Ng8 both reach f6
    assert board.find_move("Nf6", "opponent")[0].uci() == "g8f6"
    assert board.find_move("Nf6", "to_move")[0].uci() == "e4f6"
    with pytest.raises(ChessError, match="side not to move"):
        board.find_move("Kd2", "opponent")


def test_repetition_is_claimable_not_automatic():
    board = ChessSession()
    board.new_game()
    board.board = chess.Board("4k1n1/8/8/8/8/8/8/4K1N1 w - - 0 1")
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.board.push_uci(move)
    assert not board.game_over and board.view()["draw_claimable"]
    assert board.claim_draw() == "threefold repetition"
    assert board.game_over and not board.view()["interactive"] and "Draw claimed" in board.view()["caption"]
    with pytest.raises(ChessError):
        board.claim_draw()


def test_underpromotion_and_default_queen():
    board = ChessSession()
    board.set_position("8/4P3/8/8/8/8/8/k6K w - - 0 1", user_moves_next=True)
    board.user_move("e7e8n")
    assert board.board.piece_at(chess.E8).symbol() == "N"
    board.set_position("8/4P3/8/8/8/8/8/k6K w - - 0 1", user_moves_next=True)
    board.user_move("e7e8")
    assert board.board.piece_at(chess.E8).symbol() == "Q"


def test_attack_lookups_come_from_the_board():
    board = ChessSession()
    board.set_position("4k3/8/8/4p3/8/5N2/8/4K3 w - - 0 1")
    assert "e5" in board.attacked_by("f3") and "d4" in board.attacked_by("f3")
    assert board.attackers_of("e5") == {"white": ["Nf3"], "black": []}
    assert board.attackers_of("d4") == {"white": ["Nf3"], "black": ["e5"]}


def test_claim_draw_endpoint(tmp_path):
    llm = ScriptedLLM([{"content": "", "tool_calls": [call("new_game", {})]}, {"content": "Go!"}])
    with client_with(llm, tmp_path) as client:
        session = client.post("/api/sessions", json={"agent": "chess_coach"}).json()
        client.post(f"/api/sessions/{session['id']}/messages", json={"content": "Let's play"})
        refused = client.post(f"/api/sessions/{session['id']}/chess/claim-draw")
        assert refused.status_code == 422 and "can't be claimed" in refused.json()["detail"]


# ------------------------------------------------------------------ handoff from the main chat

def test_lets_play_chess_hands_the_chat_to_the_coach_and_back(tmp_path):
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [call("handoff", {"agent": "chess_coach"})]},
        {"content": "", "tool_calls": [call("new_game", {})]},
        {"content": "You're White. Your move!"},
        {"content": "", "tool_calls": [call("handoff_back", {"note": "wants arithmetic"})]},
        {"content": "It is 42."},
    ])
    with client_with(llm, tmp_path) as client:
        session = client.post("/api/sessions", json={"agent": "coordinator"}).json()
        sid = session["id"]
        assert "board" not in session and session["active_agent"]["name"] == "coordinator"

        events = parse_sse(client.post(f"/api/sessions/{sid}/messages",
                                       json={"content": "let's play chess"}).text)
        starts = [e for e in events if e["type"] == "turn_start"]
        assert [(e["agent"], e["turn"], e["via"]) for e in starts] == [
            ("coordinator", 1, None), ("chess_coach", 2, "coordinator")]
        assert starts[1]["user_message"] == "let's play chess"
        handoff = next(e for e in events if e["type"] == "handoff")
        assert handoff["to_agent"] == "chess_coach" and handoff["ui_state"]["board"]
        assert any(e["type"] == "board" and e["board"]["mode"] == "game" for e in events)
        ends = [e for e in events if e["type"] == "turn_end"]
        assert ends[0]["final_text"] == "Handing you over to the Chess coach."
        assert ends[1]["final_text"] == "You're White. Your move!"
        assert len(llm.requests) == 3  # the coordinator made one call, the coach two

        # The coordinator's tool list offered the handoff, and the coach can hand back.
        assert any(t["function"]["name"] == "handoff" for t in llm.requests[0]["tools"])
        assert any(t["function"]["name"] == "handoff_back" for t in llm.requests[1]["tools"])

        detail = client.get(f"/api/sessions/{sid}").json()
        assert detail["active_agent"]["name"] == "chess_coach" and detail["board"]["interactive"]
        assert len([e for e in detail["events"] if e["type"] == "turn_start"]) == 2
        assert detail["title"] == "let's play chess"
        moved = client.post(f"/api/sessions/{sid}/chess/move", json={"move": "e2e4"}).json()
        assert len(moved["board"]["history"]) == 2

        events = parse_sse(client.post(f"/api/sessions/{sid}/messages",
                                       json={"content": "actually, what is 6*7?"}).text)
        starts = [e for e in events if e["type"] == "turn_start"]
        assert [(e["agent"], e["turn"], e["via"]) for e in starts] == [
            ("chess_coach", 3, None), ("coordinator", 4, "chess_coach")]
        assert "wants arithmetic" in starts[1]["user_message"]
        assert events[-1]["final_text"] == "It is 42."
        detail = client.get(f"/api/sessions/{sid}").json()
        assert detail["active_agent"]["name"] == "coordinator" and "board" not in detail
        assert client.post(f"/api/sessions/{sid}/chess/move", json={"move": "d2d4"}).status_code == 404


def test_fake_model_lets_play_chess_from_the_main_chat(tmp_path):
    with client_with(FakeLLM(), tmp_path) as client:
        sid = client.post("/api/sessions", json={"agent": "coordinator"}).json()["id"]
        events = parse_sse(client.post(f"/api/sessions/{sid}/messages",
                                       json={"content": "Let's play chess"}).text)
        board = [e for e in events if e["type"] == "board"][-1]["board"]
        assert board["mode"] == "game" and board["interactive"]
        assert events[-1]["final_text"].startswith("(fake model) New game!")
        lesson = parse_sse(client.post(f"/api/sessions/{sid}/messages",
                                       json={"content": "Teach me how the knight moves"}).text)
        assert [e for e in lesson if e["type"] == "board"][-1]["board"]["dots"]
