"""Small urllib helpers: polite User-Agent, retries with backoff, byte ranges."""

from __future__ import annotations

import time
import urllib.error
import urllib.request

USER_AGENT = "StationTruth (+https://github.com/JerryWoof/StationTruth)"

# Status codes worth retrying (rate limiting and transient server errors).
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class HTTPStatusError(RuntimeError):
    def __init__(self, url: str, status: int, body: bytes = b""):
        super().__init__(f"HTTP {status} for {url}: {body[:200]!r}")
        self.url = url
        self.status = status
        self.body = body


def request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    retries: int = 4,
    backoff: float = 2.0,
    timeout: float = 60.0,
    opener: urllib.request.OpenerDirector | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Perform an HTTP request; return ``(status, headers, body)``.

    Retries on network errors and on 408/429/5xx with exponential backoff.
    Other HTTP errors raise :class:`HTTPStatusError` immediately.
    """
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    open_ = opener.open if opener is not None else urllib.request.urlopen
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with open_(req, timeout=timeout) as resp:
                body = resp.read()
                return resp.status, dict(resp.headers.items()), body
        except urllib.error.HTTPError as exc:
            body = exc.read() if exc.fp is not None else b""
            if exc.code in _RETRY_STATUS and attempt < retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = backoff * 2**attempt
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                time.sleep(min(delay, 120.0))
                attempt += 1
                continue
            if 300 <= exc.code < 400 and exc.headers is not None:
                # Redirect deliberately not followed (see storage); hand it back.
                return exc.code, dict(exc.headers.items()), body
            raise HTTPStatusError(url, exc.code, body) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt >= retries:
                raise
            time.sleep(backoff * 2**attempt)
            attempt += 1


def get(url: str, **kwargs) -> bytes:
    return request(url, **kwargs)[2]


def get_range(url: str, start: int, end: int | None, **kwargs) -> bytes:
    """Fetch bytes ``start..end`` (inclusive); ``end=None`` reads to the end of the file."""
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    headers = dict(kwargs.pop("headers", None) or {})
    headers["Range"] = rng
    status, _, body = request(url, headers=headers, **kwargs)
    if status == 200 and start > 0:
        # Server ignored the Range header and sent the full file.
        body = body[start : None if end is None else end + 1]
    return body


def exists(url: str, **kwargs) -> bool:
    """True if ``url`` answers 200/206 to a HEAD request (404/403 -> False)."""
    try:
        status, _, _ = request(url, method="HEAD", **kwargs)
    except HTTPStatusError as exc:
        if exc.status in (403, 404):
            return False
        raise
    return status in (200, 206)
