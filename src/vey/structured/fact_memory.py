"""Temporal fact memory.

The membership question (was tool X seen in the last D events) is an exact
fact. RegisterBank V1 was trained to approximate it and, with the registers
zeroed, scored identically, so the network had learned to ignore its own
state. An exact table answers the question with no parameters and no gradient.

4096 buckets of uint16 event indices: 8 KB. A zero means "never seen", so the
first event is stored as 1. Modular subtraction covers a lookback of 65535
events, past the longest horizon the register bank claimed.
"""
from __future__ import annotations

import numpy as np

BUCKETS = 4096
MOD = 65536


class FactMemory:
    def __init__(self, buckets: int = BUCKETS):
        self.last_seen = np.zeros(buckets, dtype=np.uint16)
        self.clock = 0

    def observe(self, bucket: int) -> None:
        self.clock += 1
        self.last_seen[bucket] = np.uint16(self.clock % MOD)

    def age(self, bucket: int) -> int | None:
        stored = int(self.last_seen[bucket])
        if stored == 0:
            return None
        return (self.clock - stored) % MOD

    def seen_within(self, bucket: int, distance: int) -> bool:
        age = self.age(bucket)
        return age is not None and age <= distance

    @property
    def bytes(self) -> int:
        return self.last_seen.nbytes
