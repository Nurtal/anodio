"""Params of an Anonymization, drawn deterministically from a Seed."""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass

import numpy as np

# Interval in which each Param is drawn from the Seed.
SAFE_RANGES: dict[str, tuple[float, float]] = {
    "mcadams_alpha": (0.5, 0.9),
}


@dataclass(frozen=True)
class Params:
    seed: int
    mcadams_alpha: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Params:
        return cls(**data)


def random_seed() -> int:
    return secrets.randbelow(2**31)


def draw_params(seed: int) -> Params:
    rng = np.random.default_rng(seed)
    low, high = SAFE_RANGES["mcadams_alpha"]
    return Params(seed=seed, mcadams_alpha=float(rng.uniform(low, high)))
