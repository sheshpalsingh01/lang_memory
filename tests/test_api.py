"""API endpoint tests (FastAPI TestClient). Offline except the one /end test, which
needs Groq and is opt-in via RUN_LLM_TESTS=1."""

import os

import pytest
from fastapi.testclient import TestClient

from lang_memory import api


@pytest.fixture
def client():
    # fresh per-campaign memory bound to THIS test's tmp DATA_DIR + empty session registry
    api._memory.cache_clear()
    api._sessions.clear()
    api._seen.clear()
    with TestClient(api.app) as c:        # `with` runs startup (warmup + reaper) / shutdown
        yield c


def _start(client, **kw):
    body = {"caller_id": "+91-test", "base_prompt": "You are an agent.", **kw}
    r = client.post("/calls", json=body)
    assert r.status_code == 200
    return r.json()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_start_returns_call_id_and_clean_prompt(client):
    j = _start(client)
    assert j["call_id"]
    assert "gAAAA" not in j["system_prompt"]            # no PII ciphertext leaked into prompt


def test_turn_records(client):
    cid = _start(client)["call_id"]
    assert client.post(f"/calls/{cid}/turn", json={"role": "caller", "text": "hi"}).json()["ok"]


def test_invalid_role_is_400(client):
    cid = _start(client)["call_id"]
    assert client.post(f"/calls/{cid}/turn", json={"role": "bot", "text": "x"}).status_code == 400


def test_state_fills_slots_and_reports_complete(client):
    cid = _start(client, required_slots=["date"])["call_id"]
    client.post(f"/calls/{cid}/state", json={"intent": "book", "slots": {"date": "2026-07-01"}})
    st = client.get(f"/calls/{cid}/state").json()
    assert st["is_complete"] is True
    assert st["key_points"]["intent"] == "book"


def test_profile_persists_and_caller_lookup(client):
    cid = _start(client, caller_id="+91-prof")["call_id"]
    client.post(f"/calls/{cid}/profile",
                json={"profile": {"name": "Ravi"}, "preferences": {"channel": "sms"}})
    j = client.get("/callers/+91-prof").json()
    assert j["profile"]["name"] == "Ravi" and j["preferences"]["channel"] == "sms"


def test_caller_lookup_hides_case_and_ciphertext(client):
    # store an identity field AND a structured case (mimicking an encrypted PII field)
    calling = api._memory().start_call("+91-case").calling
    calling.update_profile(name="Asha")
    calling.set_case("kyc", fields={"pan_no": "ENCRYPTED_BLOB"}, missing=["pan"])
    prof = client.get("/callers/+91-case").json()["profile"]
    assert prof == {"name": "Asha"}            # scalar identity only — nested case not exposed
    assert "kyc" not in prof and "ENCRYPTED_BLOB" not in str(prof)


def test_unknown_call_id_is_404(client):
    assert client.post("/calls/nope/turn", json={"role": "caller", "text": "x"}).status_code == 404
    assert client.get("/calls/nope/state").status_code == 404


def test_knowledge_is_campaign_scoped(client):
    client.post("/knowledge?campaign_id=loans", json={"text": "EMI starts after disbursal"})
    loans = client.get("/knowledge/search",
                       params={"q": "EMI", "campaign_id": "loans", "rerank": False}).json()["results"]
    assert any("EMI" in h["text"] for h in loans)
    # a different campaign cannot see it
    other = client.get("/knowledge/search",
                       params={"q": "EMI", "campaign_id": "insurance", "rerank": False}).json()["results"]
    assert other == []


@pytest.mark.skipif(not os.getenv("RUN_LLM_TESTS"), reason="set RUN_LLM_TESTS=1 (Groq /end call)")
def test_end_persists_and_clears_session(client):
    cid = _start(client, caller_id="+91-end")["call_id"]
    client.post(f"/calls/{cid}/turn", json={"role": "caller", "text": "Book a demo for Monday."})
    client.post(f"/calls/{cid}/turn", json={"role": "agent", "text": "Booked Monday, confirmation by SMS."})
    res = client.post(f"/calls/{cid}/end").json()
    assert "outcome" in res
    # the live session is forgotten after /end
    assert client.get(f"/calls/{cid}/state").status_code == 404
