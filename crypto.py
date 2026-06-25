"""
crypto.py
─────────
PII helpers for compliance (DPDP / GDPR). Two tools:

  • redact_pii(text)  — replace document numbers in FREE TEXT (summaries, transcripts)
                        with tags like [PAN]. Data minimisation: most stored text then
                        holds no PII at all, so there's nothing sensitive to protect there.
  • encrypt / decrypt — for the few CANONICAL PII values you must keep (e.g. the actual
                        pan_no in a KYC case). AES via Fernet, key from the environment.

Production MUST set LANG_MEMORY_ENC_KEY (a Fernet key). The dev fallback below only
exists so local runs / tests work; it is NOT secret.
"""

import re
from functools import lru_cache

from .config import settings

# Dev-only fallback key. In production set LANG_MEMORY_ENC_KEY in the environment / KMS.
_DEV_KEY = "zTRAZc3qt-kvjvQgeAfeJbKQEk5vzoeHrUZeUfe5T2w="


@lru_cache(maxsize=1)
def _fernet():
    from cryptography.fernet import Fernet
    return Fernet(settings.ENC_KEY or _DEV_KEY)


def encrypt(text: str) -> str:
    """Encrypt a single PII value (no-op on empty)."""
    if not text:
        return text
    return _fernet().encrypt(str(text).encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a value. If it isn't ciphertext (e.g. legacy plaintext), return it as-is."""
    if not token:
        return token
    try:
        return _fernet().decrypt(str(token).encode()).decode()
    except Exception:
        return token


# Document-number patterns to scrub from free text.
_PII_PATTERNS = [
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),            "[PAN]"),
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),       "[AADHAAR]"),
    (re.compile(r"\b[A-Z][0-9]{7}\b"),                     "[PASSPORT]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),          "[EMAIL]"),
]


def redact_pii(text: str) -> str:
    """Mask document numbers / emails in free text before it is stored or embedded."""
    for pattern, tag in _PII_PATTERNS:
        text = pattern.sub(tag, text)
    return text
