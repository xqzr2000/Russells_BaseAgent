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
    return TestClient(create_app(llm=llm, log_dir=tmp_path, workspace_dir=tmp_path / "workspace"))


def test_agents_and_config(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-mini")  # ignored without a key
    with client_with(FakeLLM(), tmp_path) as client:
        agents = client.get("/api/agents").json()
        assert any(a["name"] == "general" for a in agents)
        config = client.get("/api/config").json()
        assert config["openai_key_configured"] is False and config["default_model"] == FAKE_MODEL
        assert config["default_agent"] == "coordinator"
        assert {"coordinator", "data_science"} <= {a["name"] for a in agents}
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


CSV = "name,kind,age\nRex,dog,3\nTom,cat,\nRex,dog,3\n"


def test_upload_eda_and_download_notebook(tmp_path):
    """The chat-room flow offline: drop a CSV, ask for EDA, download the notebook."""
    import nbformat

    with client_with(FakeLLM(), tmp_path) as client:
        sid = client.post("/api/sessions", json={"settings": {"model": FAKE_MODEL}}).json()["id"]
        uploaded = client.put(f"/api/sessions/{sid}/files/my pets.csv", content=CSV.encode()).json()
        assert uploaded == {"name": "my_pets.csv", "path": "data/my_pets.csv", "size": len(CSV)}

        events = parse_sse(client.post(f"/api/sessions/{sid}/messages", json={
            "content": "do EDA on it", "files": [uploaded["path"]]}).text)
        start = events[0]
        assert start["agent"] == "coordinator"
        assert start["user_message"] == "do EDA on it\n\nAttached file: data/my_pets.csv"
        delegate = next(e for e in events if e["type"] == "tool_end" and e["name"] == "delegate")
        assert not delegate["is_error"], delegate["content"]
        assert "Rows: 3  Columns: 3  Duplicate rows: 1" in delegate["content"]
        notebook = delegate["attachments"][0]
        assert notebook["url"] == f"/api/sessions/{sid}/files/my_pets_eda.ipynb"
        assert any(e["type"] == "tool_progress" for e in events)
        assert events[-1]["type"] == "turn_end" and events[-1]["stop_reason"] == "final"
        assert client.get("/api/sessions").json()[0]["title"] == "do EDA on it"

        response = client.get(notebook["url"])
        assert response.status_code == 200
        assert 'attachment; filename="my_pets_eda.ipynb"' in response.headers["content-disposition"]
        nb = nbformat.reads(response.text, as_version=4)
        nbformat.validate(nb)
        assert any("pd.read_csv('data/my_pets.csv')" in c.source for c in nb.cells)
        assert (tmp_path / f"{sid}.data_science.json").is_file()  # the specialist's own log


def test_file_endpoints_reject_bad_input(tmp_path):
    with client_with(FakeLLM(), tmp_path) as client:
        sid = client.post("/api/sessions", json={"agent": "general"}).json()["id"]
        assert client.put(f"/api/sessions/{sid}/files/run.sh", content=b"echo").status_code == 415
        assert client.put(f"/api/sessions/{sid}/files/empty.csv", content=b"").status_code == 422
        # uploads can't escape data/: "/" never reaches the handler, other names are flattened
        assert client.put(f"/api/sessions/{sid}/files/..%2F..%2Fx.csv", content=b"a\n1\n").status_code == 405
        escaped = client.put(f"/api/sessions/{sid}/files/..%5C..%5Cx.csv", content=b"a\n1\n").json()
        assert escaped["path"] == "data/x.csv"
        assert not list(tmp_path.glob("**/x.csv"))[1:]  # exactly one copy, inside data/
        (tmp_path / "secret.txt").write_text("secret")
        for traversal in ("..%2F..%2F..%2Fsecret.txt", "%2E%2E/%2E%2E/%2E%2E/secret.txt"):
            assert client.get(f"/api/sessions/{sid}/files/{traversal}").status_code == 404
        assert client.get(f"/api/sessions/{sid}/files/data/missing.csv").status_code == 404
        assert client.get(f"/api/sessions/{sid}/files/data/x.csv").status_code == 200
        missing = client.post(f"/api/sessions/{sid}/messages",
                              json={"content": "hi", "files": ["data/nope.csv"]})
        assert missing.status_code == 422 and "not found" in missing.json()["detail"]
        outside = client.post(f"/api/sessions/{sid}/messages",
                              json={"content": "hi", "files": ["../../secret.csv"]})
        assert outside.status_code == 422
        assert client.put("/api/sessions/nope/files/a.csv", content=b"a").status_code == 404
