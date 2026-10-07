#!/usr/bin/env python3
"""Thermal governor for the local RTX 3060 research lane.

Monitors host temperature via nvidia-smi and swaps the registered training
process between SIGSTOP and SIGCONT once the ceiling/floor is crossed. Pure
governor: no model changes, no resumptions, every action is timestamped and
retained. Fails closed if the temperature probe is unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
PROTOCOL_PIN = {"path": str(here / "accelerated_local_protocol.json")}


def probe_temperature():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=False)
    if out.returncode != 0:
        raise RuntimeError("nvidia-smi temperature probe failed: " + out.stderr.strip())
    values = [int(line.strip()) for line in out.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no temperature values")
    return max(values)


def read_meminfo(path="/proc/meminfo"):
    fields = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            parts = value.split()
            if parts and parts[0].isdigit():
                fields[name] = int(parts[0]) * 1024
    return fields["MemAvailable"], fields["SwapTotal"] - fields["SwapFree"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Governor:
    def __init__(self, pid, log, poll=5, pause_C=78, resume_C=70, cool_seconds=60):
        self.pid = pid
        self.log = Path(log)
        self.poll = poll
        self.pause_C = pause_C
        self.resume_C = resume_C
        self.cool_seconds = cool_seconds
        self.paused_at = None

    def write(self, event):
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**event, "t": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())},
                                    sort_keys=True) + "\n")

    def run(self):
        self.write({"schema": "vey.accel.thermal-governor.v1", "status": "STARTED", "pid": self.pid})
        while True:
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                self.write({"status": "PROCESS_ENDED"})
                return 0
            except PermissionError:
                self.write({"status": "PERMISSION_DENIED"})
                return 2
            temp = probe_temperature()
            available, swap_used = read_meminfo()
            event = {"temperature_C": temp, "MemAvailable_bytes": available, "swap_used_bytes": swap_used}
            if self.paused_at is None and temp >= self.pause_C:
                os.kill(self.pid, signal.SIGSTOP)
                self.paused_at = time.time()
                event.update(status="PAUSED", reason="temperature")
            elif self.paused_at is not None:
                elapsed = time.time() - self.paused_at
                if temp <= self.resume_C and elapsed >= self.cool_seconds:
                    os.kill(self.pid, signal.SIGCONT)
                    event.update(status="RESUMED", paused_seconds=round(elapsed, 1))
                    self.paused_at = None
                else:
                    event.update(status="PAUSED", waiting_for="cooldown" if elapsed < self.cool_seconds else "temperature")
            self.write(event)
            time.sleep(self.poll)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll", type=int, default=5)
    args = parser.parse_args(argv)
    governor = Governor(args.pid, args.log, poll=args.poll)
    return governor.run()


if __name__ == "__main__":
    raise SystemExit(main())
