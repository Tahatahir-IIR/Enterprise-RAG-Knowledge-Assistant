from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)


def post_with_retry(
    url: str, json: dict[str, Any], timeout: float, headers: dict[str, str] | None = None, attempts: int = 4
) -> httpx.Response:
    """POST with exponential backoff on 5xx and connection errors.

    Local model servers (Ollama) occasionally return 500 while swapping models
    in and out of memory; a short wait is enough for the request to succeed.
    """
    delay = 1.0
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            r = httpx.post(url, json=json, headers=headers or {}, timeout=timeout)
            if r.status_code < 500:
                r.raise_for_status()
                return r
            last = httpx.HTTPStatusError(f"{r.status_code} from {url}: {r.text[:200]}", request=r.request, response=r)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            last = exc
        if attempt < attempts:
            log.warning("request to %s failed (%s); retry %d/%d in %.0fs", url, last, attempt, attempts - 1, delay)
            time.sleep(delay)
            delay *= 2
    assert last is not None
    raise last
