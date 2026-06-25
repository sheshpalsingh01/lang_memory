"""PII compliance: encryption round-trip, redaction, and right-to-erasure."""

from lang_memory import VoiceAgentMemory
from lang_memory.crypto import encrypt, decrypt, redact_pii


def test_encrypt_roundtrip():
    token = encrypt("ABCDE1234F")
    assert token != "ABCDE1234F"          # stored value is ciphertext
    assert decrypt(token) == "ABCDE1234F"  # recoverable


def test_decrypt_passthrough_on_plaintext():
    # legacy / unencrypted values must not blow up
    assert decrypt("just-plain-text") == "just-plain-text"


def test_redact_pii():
    out = redact_pii("PAN ABCDE1234F, aadhaar 1234-5678-9012, mail a@b.com")
    assert "ABCDE1234F" not in out and "[PAN]" in out
    assert "1234-5678-9012" not in out and "[AADHAAR]" in out
    assert "a@b.com" not in out and "[EMAIL]" in out


def test_forget_erases_everything():
    m = VoiceAgentMemory()
    c = m.start_call("erase-me")
    c.calling.update_profile(name="X")
    c.calling.add_episode("the caller booked a demo", {})
    assert m.start_call("erase-me").calling.get_profile() == {"name": "X"}

    m.forget("erase-me")
    assert m.start_call("erase-me").calling.get_profile() == {}
    assert m.start_call("erase-me").calling.recall_episodes("demo") == []
