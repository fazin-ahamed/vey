"""Pure regressions for the accelerated lane helpers."""
import json
import signal
import subprocess

import pytest

import accelerated_thermal as T
import accelerated_microbench as M


def test_temperature_probe_parses_real_tool():
    temp = T.probe_temperature()
    assert 20 <= temp <= 110


def test_governor_pauses_hot_and_resumes_cooled_sleeper(monkeypatch, tmp_path):
    sleeper = subprocess.Popen(["sleep", "60"])
    log = tmp_path / "gov.log"
    clock = {"t": 100.0}
    temps = [50, 50, 80, 80, 65, 65, 65]

    class FakeTime:
        @staticmethod
        def time():
            return clock["t"]

        @staticmethod
        def sleep(_seconds):
            clock["t"] += 60.0
            if not temps:
                raise SystemExit(0)

        @staticmethod
        def strftime(fmt, t=None):
            return "00:00:00"

        @staticmethod
        def gmtime(t=None):
            return ()

    monkeypatch.setattr(T, "time", FakeTime)
    monkeypatch.setattr(T, "probe_temperature", lambda: temps.pop(0))
    monkeypatch.setattr(T, "read_meminfo", lambda: (16 * 1024**3, 0))
    calls = []
    class Sig:
        def __init__(self):
            self.set = []

        def apply(self, pid, sig):
            self.set.append(sig)
    signal_holder = Sig()
    monkeypatch.setattr(T.os, "kill", lambda pid, sig: signal_holder.apply(pid, sig))
    governor = T.Governor(sleeper.pid, log, poll=1, cool_seconds=60)
    with pytest.raises(SystemExit):
        governor.run()
    assert signal.SIGSTOP in signal_holder.set
    assert signal.SIGCONT in signal_holder.set
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(e.get("status") == "PAUSED" for e in lines)
    assert any(e.get("status") == "RESUMED" for e in lines)
    sleeper.terminate(); sleeper.wait()


def test_microbench_plan_cost_stays_under_cap():
    plan = M.plan(spend_ledger_usd=0.0)
    assert plan["cost_usd_estimated"]["total"] < M.CAP_USD_THIS_PERIOD
    assert set(plan["cost_usd_estimated"]) == {"T4", "L4", "A10", "A100-40GB", "total"}


def test_microbench_plan_reports_when_exhausted():
    plan = M.plan(spend_ledger_usd=1.49)
    assert plan["cost_usd_estimated"]["total"] > M.CAP_USD_THIS_PERIOD
