"""System One: a Jev / Clef / Laya compatible decision server on Vey's lanes.

    from vey.systemone import Engine, make_server
    from vey.semantic import OnnxEmbedder

    engine = Engine(OnnxEmbedder.from_hub())
    make_server(engine, port=8000).serve_forever()

or from a shell: ``vey-serve --port 8000``. See ``docs/SYSTEMONE.md``.
"""
from .calibration import Calibration, ece, fit_noul_temperature, fit_temperature
from .engine import Engine, EngineConfig, select_state
from .escalate import EscalationError, Escalator
from .nli import LazyNLI, OnnxNLI, to_statement
from .schema import ApiError, Question, Request, parse_request
from .server import make_server

__all__ = ["Engine", "EngineConfig", "make_server", "Escalator", "EscalationError", "Calibration",
           "fit_temperature", "fit_noul_temperature", "ece", "LazyNLI", "OnnxNLI", "to_statement",
           "select_state", "parse_request", "Request", "Question", "ApiError"]
