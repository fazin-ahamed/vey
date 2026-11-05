import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vey.systemone import Calibration, ece, fit_noul_temperature, fit_temperature
from vey.systemone.calibration import bucket, key_for, softmax


def test_fit_recovers_the_temperature_that_generated_the_labels():
    rng = np.random.default_rng(0)
    true_t = 0.08
    rows = [rng.normal(0, 0.1, 6) for _ in range(4000)]
    labels = [int(rng.choice(6, p=softmax(r, true_t))) for r in rows]
    assert fit_temperature(rows, labels) == pytest.approx(true_t, rel=0.15)


def test_fit_noul_recovers_the_temperature():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, 6000)
    y = (rng.random(6000) < 1 / (1 + np.exp(-z / 2.0))).astype(int).tolist()
    assert fit_noul_temperature(z.tolist(), y) == pytest.approx(2.0, rel=0.15)


def test_ece_is_near_zero_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(2)
    conf = rng.uniform(0.5, 1.0, 20000)
    calibrated = (rng.random(20000) < conf).tolist()
    assert ece(conf.tolist(), calibrated) < 0.02
    assert ece([0.99] * 1000, [i % 2 == 0 for i in range(1000)]) > 0.4


def test_buckets_and_keys():
    assert [bucket(k) for k in (2, 3, 5, 6, 10, 11, 255)] == ["2", "3-5", "3-5", "6-10", "6-10", "11+", "11+"]
    assert key_for("choice", 77) == "choice:11+" and key_for("noul") == "noul" and key_for("score", 5) == "score"


def test_load_overrides_defaults_and_keeps_the_rest(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"temperatures": {"choice:2": 0.5}}))
    c = Calibration.load(str(p))
    assert c.temperature("choice", 2) == 0.5
    assert c.temperature("noul") == Calibration().temperature("noul")
    assert Calibration.default().temperature("choice", 4) > 0


def test_importing_the_server_does_not_import_torch():
    code = ("import sys, vey, vey.semantic, vey.systemone;"
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)[:3]")
    src = str(Path(__file__).resolve().parents[2] / "src")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": src})
    assert r.returncode == 0, r.stderr
