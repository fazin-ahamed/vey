"""``vey-serve``: run the System One server."""
from __future__ import annotations

import argparse
import os
import sys

from .calibration import Calibration
from .engine import Engine, EngineConfig
from .escalate import Escalator
from .nli import LazyNLI, OnnxNLI
from .server import DEFAULT_MAX_BODY, DEFAULT_MAX_PROXY_BODY, make_server, resident_mb


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vey-serve", description="Jev / Clef / Laya compatible System One server.")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; use a reverse proxy to expose it)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", help="require this bearer token on /v1 routes (prefer the VEY_API_KEY environment variable)")
    p.add_argument("--backend", choices=("onnx", "torch"), default="onnx",
                   help="encoder runtime: onnx needs no PyTorch (default); torch is the reference (pip install 'vey[torch]')")
    p.add_argument("--variant", default="model_int8", help="ONNX file of the encoder: model_int8 (default) or model (fp32)")
    p.add_argument("--onnx-dir", help="a local directory with onnx/<variant>.onnx and tokenizer.json instead of the Hub")
    p.add_argument("--threads", type=int, default=None, help="ONNX Runtime threads per model (default min(4, CPUs))")
    p.add_argument("--cache-size", type=int, default=50_000, help="embedding cache entries (LRU)")
    p.add_argument("--calibration", help="JSON file of temperatures from scripts/fit_calibration.py")
    p.add_argument("--no-nli", action="store_true", help="disable the NLI lane; noul questions then need an exact field or escalation")
    p.add_argument("--nli-dir", help="a local directory with the NLI ONNX model instead of the Hub")
    p.add_argument("--no-exact", action="store_true", help="disable the exact field-match lane")
    p.add_argument("--preload", action="store_true", help="load the NLI model at start instead of on first use")
    p.add_argument("--floor", type=float, default=0.0,
                   help="confidence below this escalates (with --escalate-url) or is flagged in the trace (default 0: off)")
    p.add_argument("--escalate-url", help="another System One server for what Vey cannot answer, e.g. https://api.typesafe.ai")
    p.add_argument("--escalate-model", default="jev-latest", help="model name to ask the escalation target for")
    p.add_argument("--escalate-timeout", type=float, default=30.0)
    p.add_argument("--max-body", type=int, default=DEFAULT_MAX_BODY, help="max JSON request bytes")
    p.add_argument("--max-proxy-body", type=int, default=DEFAULT_MAX_PROXY_BODY, help="max multipart bytes forwarded to the escalation target")
    p.add_argument("--explain", action="store_true", help="include the lane trace in every response")
    p.add_argument("--quiet", action="store_true")
    return p


def build_engine(a: argparse.Namespace) -> Engine:
    if a.backend == "torch":
        from ..semantic import Embedder, load_encoder

        tok, enc = load_encoder(device="cpu")
        embedder = Embedder(tok, enc, "cpu")
    else:
        from ..semantic.onnx_encoder import OnnxEmbedder

        kw = dict(variant=a.variant, threads=a.threads, cache_size=a.cache_size)
        embedder = OnnxEmbedder.from_dir(a.onnx_dir, **kw) if a.onnx_dir else OnnxEmbedder.from_hub(**kw)
    if a.no_nli:
        nli = LazyNLI(loader=lambda: (_ for _ in ()).throw(RuntimeError("disabled with --no-nli")))
    elif a.nli_dir:
        nli = LazyNLI(loader=lambda: OnnxNLI.from_dir(a.nli_dir, threads=a.threads))
    else:
        nli = LazyNLI(loader=lambda: OnnxNLI.from_hub(threads=a.threads))
    if a.preload and not a.no_nli:
        nli.get()
    escalator = None
    if a.escalate_url:
        escalator = Escalator(a.escalate_url, api_key=os.environ.get("VEY_ESCALATE_API_KEY"),
                              model=a.escalate_model, timeout=a.escalate_timeout)
    cal = Calibration.load(a.calibration) if a.calibration else Calibration.default()
    cfg = EngineConfig(exact_lane=not a.no_exact, floor=a.floor, explain=a.explain)
    return Engine(embedder, nli=nli, calibration=cal, escalator=escalator, config=cfg)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    api_key = a.api_key or os.environ.get("VEY_API_KEY")
    engine = build_engine(a)
    srv = make_server(engine, a.host, a.port, api_key=api_key, max_body=a.max_body,
                      max_proxy_body=a.max_proxy_body, quiet=a.quiet)
    host, port = srv.server_address[:2]
    print(f"vey-serve: http://{host}:{port}  console at /  ({resident_mb()} MB resident, "
          f"auth {'on' if api_key else 'off'}, escalation {'on' if engine.escalator else 'off'})", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
