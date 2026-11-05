"""HTTP server for the System One wire format, on the standard library only.

    POST /v1/systemone   answer typed questions about a state (Jev / Clef / Laya compatible)
    GET  /v1/models      model names this server answers to
    GET  /v1/info        lanes, counters, resident memory (a Vey extension; the console reads it)
    GET  /healthz        liveness, never authenticated
    GET  /               the console

Binds to 127.0.0.1 by default. It is a small threaded server meant for local use
and for sitting behind a reverse proxy; it does not terminate TLS. Set
``VEY_API_KEY`` (or ``--api-key``) to require ``Authorization: Bearer <key>`` on
every ``/v1`` route.
"""
from __future__ import annotations

import hmac
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from .engine import Engine
from .escalate import EscalationError
from .schema import ApiError, bad_request, parse_request

DEFAULT_MAX_BODY = 1 << 20          # 1 MiB of JSON
DEFAULT_MAX_PROXY_BODY = 25 << 20   # images forwarded to the escalation target


def resident_mb() -> float | None:
    """Resident set size of this process in MB (Linux), else the peak from ``resource``."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except Exception:
        return None


def _console_html() -> bytes:
    return resources.files("vey.systemone").joinpath("console.html").read_bytes()


def make_server(engine: Engine, host: str = "127.0.0.1", port: int = 8000, api_key: str | None = None,
                max_body: int = DEFAULT_MAX_BODY, max_proxy_body: int = DEFAULT_MAX_PROXY_BODY,
                quiet: bool = False) -> ThreadingHTTPServer:
    api_key = api_key or None

    class Handler(BaseHTTPRequestHandler):
        server_version = "vey-systemone"
        protocol_version = "HTTP/1.1"

        # ------------------------------------------------------------ plumbing
        def log_message(self, fmt, *args):  # noqa: A003
            if not quiet:
                sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

        def _send(self, status: int, body: bytes, content_type: str = "application/json",
                  extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if content_type == "application/json" else "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode())

        def _authorized(self) -> bool:
            if not api_key:
                return True
            got = self.headers.get("Authorization", "")
            return got.startswith("Bearer ") and hmac.compare_digest(got[7:].strip().encode(), api_key.encode())

        def _read_body(self, limit: int) -> bytes:
            length = self.headers.get("Content-Length")
            if length is None or not length.isdigit():
                raise ApiError(411, {"detail": "Content-Length is required"})
            if int(length) > limit:
                # drain nothing; close so the client does not keep sending
                self.close_connection = True
                raise ApiError(413, {"detail": f"Request body too large (limit {limit} bytes)"})
            return self.rfile.read(int(length))

        def _guard(self, fn) -> None:
            try:
                fn()
            except ApiError as e:
                self._json(e.status, e.body)
            except EscalationError as e:
                self._json(502, {"detail": str(e)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:  # never leak internals to the client
                traceback.print_exc()
                self._json(500, {"detail": "internal error"})

        # ----------------------------------------------------------------- GET
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/console"):
                return self._send(200, _console_html(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'self' 'unsafe-inline'; connect-src 'self'"})
            if path == "/healthz":
                return self._json(200, {"status": "ok"})
            if path in ("/v1/models", "/v1/info"):
                if not self._authorized():
                    return self._json(401, {"detail": "Missing or invalid API key"})
                if path == "/v1/models":
                    c = engine.cfg
                    names = [*c.aliases, c.model_id]
                    return self._json(200, {"models": [
                        {"name": n, "description": c.description + (f" Alias of {c.model_id}." if n != c.model_id else ""),
                         "release_date": c.release_date} for n in names]})
                info = engine.info()
                info["resident_mb"] = resident_mb()
                info["auth_required"] = bool(api_key)
                return self._json(200, info)
            self._json(404, {"detail": "Not Found"})

        # ---------------------------------------------------------------- POST
        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/v1/systemone":
                self.close_connection = True  # the unread body would corrupt a keep-alive connection
                return self._json(404, {"detail": "Not Found"})
            if not self._authorized():
                self.close_connection = True
                return self._json(401, {"detail": "Missing or invalid API key"})
            self._guard(self._systemone)

        def _systemone(self) -> None:
            ctype = (self.headers.get("Content-Type") or "application/json").split(";")[0].strip().lower()
            hdrs = {k: v for k, v in self.headers.items()}
            if ctype == "multipart/form-data":
                if engine.escalator is None:
                    self._read_body(max_proxy_body)
                    raise bad_request("multipart (image) requests need an escalation endpoint; "
                                      "start the server with --escalate-url")
                data = self._read_body(max_proxy_body)
                status, raw, out_type = engine.escalator.proxy_raw(data, self.headers["Content-Type"], hdrs)
                return self._send(status, raw, out_type.split(";")[0] or "application/json")
            raw = self._read_body(max_body)
            try:
                body = json.loads(raw)
            except ValueError as e:
                raise ApiError(422, {"detail": [{"type": "json_invalid", "loc": ["body"],
                                                 "msg": f"JSON decode error: {e}"}]})
            explain = None
            if isinstance(body, dict) and "explain" in body:
                explain = bool(body.pop("explain"))  # a Vey extension; never forwarded
            if self.headers.get("X-Vey-Explain") == "1":
                explain = True
            req = parse_request(body)
            self._json(200, engine.answer(req, raw_body=body, headers=hdrs, explain=explain))

    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.request_queue_size = 64
    return ThreadingHTTPServer((host, port), Handler)


__all__ = ["make_server", "resident_mb", "DEFAULT_MAX_BODY", "DEFAULT_MAX_PROXY_BODY"]
