"""API tests: sessions, SSE streaming, settings, using a scripted model."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from baseagent.llm import FAKE_MODEL, FakeLLM, ScriptedLLM
from baseagent.server.app import create_app


def parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if data:
            events.append(json.loads("".join(data)))
    return events


def client_with(llm, tmp_path) -> TestClient:
    return TestClient(create_app(llm=llm, log_dir=tmp_path))


def test_agents_and_config(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-mini")  # ignored without a key
    with client_with(FakeLLM(), tmp_path) as client:
        agents = client.get("/api/agents").json()
        assert any(a["name"] == "general" for a in agents)
        config = client.get("/api/config").json()
        assert config["openai_key_configured"] is False and config["default_model"] == FAKE_MODEL
        assert client.get("/api/models").json()["models"] == [FAKE_MODEL]


def test_chat_turn_streams_events_and_persists(tmp_path):
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "calculator", "arguments": json.dumps({"expression": "6*7"})}}]},
        {"content": "It is **42**."},
    ])
    with client_with(llm, tmp_path) as client:
        session = client.post("/api/sessions", json={"agent": "general",
                                                     "settings": {"model": "gpt-test"}}).json()
        response = client.post(f"/api/sessions/{session['id']}/messages", json={"content": "6 times 7?"})
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        types = [e["type"] for e in events]
        assert types[0] == "turn_start" and types[-1] == "turn_end"
        assert next(e for e in events if e["type"] == "tool_end")["content"] == "42"
        assert events[-1]["final_text"] == "It is **42**."

        detail = client.get(f"/api/sessions/{session['id']}").json()
        assert detail["settings"]["model"] == "gpt-test"
        assert [e["type"] for e in detail["events"]][-1] == "turn_end"
        assert (tmp_path / f"{session['id']}.json").is_file()
        assert client.get("/api/sessions").json()[0]["title"] == "6 times 7?"


def test_settings_patch_switches_model(tmp_path):
    with client_with(FakeLLM(), tmp_path) as client:
        sid = client.post("/api/sessions", json={"agent": "general"}).json()["id"]
        updated = client.patch(f"/api/sessions/{sid}/settings",
                               json={"model": "gpt-5", "reasoning_effort": "low", "max_steps": 5}).json()
        assert updated["model"] == "gpt-5" and updated["reasoning_effort"] == "low"
        assert client.patch(f"/api/sessions/{sid}/settings", json={"bogus": 1}).status_code == 422
        assert client.post("/api/sessions", json={"agent": "nope"}).status_code == 404


def test_fake_model_end_to_end(tmp_path):
    with client_with(FakeLLM(), tmp_path) as client:
        sid = client.post("/api/sessions", json={"agent": "general",
                                                 "settings": {"model": FAKE_MODEL}}).json()["id"]
        events = parse_sse(client.post(f"/api/sessions/{sid}/messages",
                                       json={"content": "what is 2+2*10"}).text)
        assert "22" in events[-1]["final_text"]
        assert any(e["type"] == "text_delta" for e in events)
