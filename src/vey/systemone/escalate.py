"""Tier 3: send what the local lanes cannot answer to another System One server.

The wire format is the same everywhere, so the target can be Jev, Clef, Laya's
server, a larger d1 behind SGLang, or another Vey. Two cases go there:

  * a question the local lanes answered below the confidence floor, and
  * a request the local lanes cannot read at all: images (``assets``) and
    ``box`` questions. These are forwarded unchanged, including a multipart body.

Nothing is sent unless an escalation URL is configured.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

FORWARDED_HEADERS = ("TypeSafe-Danger-Zone-Images", "TypeSafe-Danger-Zone-Long-Context")


class EscalationError(RuntimeError):
    """The escalation target was unreachable or returned an error."""

    def __init__(self, message: str, status: int | None = None, body: bytes | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class Escalator:
    def __init__(self, url: str, api_key: str | None = None, model: str = "jev-latest",
                 timeout: float = 30.0):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"escalation URL must be http(s)://host[:port], got {url!r}")
        base = url.rstrip("/")
        self.endpoint = base if base.endswith("/v1/systemone") else base + "/v1/systemone"
        self.host = parsed.netloc
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _request(self, data: bytes, content_type: str, headers: dict[str, str] | None) -> tuple[int, bytes, str]:
        req = urllib.request.Request(self.endpoint, data=data, method="POST")
        req.add_header("Content-Type", content_type)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        incoming = {k.lower(): v for k, v in (headers or {}).items()}  # header names are case-insensitive
        for h in FORWARDED_HEADERS:
            if incoming.get(h.lower()):
                req.add_header(h, incoming[h.lower()])
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read(), r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("Content-Type", "application/json")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise EscalationError(f"escalation target {self.host} unreachable: {e}") from e

    def post_json(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Send a JSON request; return the parsed 200 response or raise EscalationError."""
        payload = dict(body, model=self.model)
        status, raw, _ = self._request(json.dumps(payload).encode(), "application/json", headers)
        if status != 200:
            raise EscalationError(f"escalation target {self.host} returned HTTP {status}", status, raw)
        try:
            return json.loads(raw)
        except ValueError as e:
            raise EscalationError(f"escalation target {self.host} returned invalid JSON") from e

    def proxy_raw(self, data: bytes, content_type: str, headers: dict[str, str] | None = None
                  ) -> tuple[int, bytes, str]:
        """Forward a body unchanged (used for multipart image requests)."""
        return self._request(data, content_type, headers)


__all__ = ["Escalator", "EscalationError", "FORWARDED_HEADERS"]
