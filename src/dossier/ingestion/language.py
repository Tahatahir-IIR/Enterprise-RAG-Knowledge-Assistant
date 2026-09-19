from __future__ import annotations

import re

_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")


def detect_language(text: str) -> str:
    """Script-based detection: enough to separate French from Arabic documents.

    Returns 'ar' when Arabic script dominates, 'fr' otherwise. Darija written in
    Latin letters is treated as 'fr' for retrieval purposes; the multilingual
    embedding model handles the semantics.
    """
    arabic = len(_ARABIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if arabic == 0 and latin == 0:
        return "fr"
    return "ar" if arabic > latin else "fr"
