"""Params of an Anonymization, drawn deterministically from a Seed."""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass

import numpy as np

# Stages of the Chain, in application order.
STAGES: tuple[str, ...] = ("mcadams", "psola", "vtln")

# Intervals in which each Param is drawn from the Seed. The gaps around the
# identity exclude values too weak to anonymize.
SAFE_RANGES: dict[str, list[tuple[float, float]]] = {
    "mcadams_alpha": [(0.5, 0.9)],
    "pitch_semitones": [(-5.0, -2.0), (2.0, 5.0)],
    "formant_ratio": [(0.85, 0.95), (1.05, 1.15)],
    "vtln_alpha": [(-0.15, -0.05), (0.05, 0.15)],
}

# Param values for which a Stage leaves the signal unchanged.
IDENTITY: dict[str, float] = {
    "mcadams_alpha": 1.0,
    "pitch_semitones": 0.0,
    "formant_ratio": 1.0,
    "vtln_alpha": 0.0,
}


@dataclass(frozen=True)
class Params:
    seed: int
    mcadams_alpha: float
    pitch_semitones: float
    formant_ratio: float
    vtln_alpha: float
    stages: tuple[str, ...] = STAGES

    def outside_safe_ranges(self) -> list[str]:
        return [
            name
            for name, ranges in SAFE_RANGES.items()
            if not any(low <= getattr(self, name) <= high for low, high in ranges)
        ]

    def to_dict(self) -> dict:
        return {**asdict(self), "stages": list(self.stages)}

    @classmethod
    def from_dict(cls, data: dict) -> Params:
        return cls(**{**data, "stages": tuple(data["stages"])})


def random_seed() -> int:
    return secrets.randbelow(2**31)


def draw_params(
    seed: int,
    overrides: dict[str, float] | None = None,
    stages: tuple[str, ...] = STAGES,
) -> Params:
    """Draw every Param in its Safe range from the Seed, then apply the Overrides."""
    unknown = set(overrides or {}) - set(SAFE_RANGES)
    if unknown:
        raise ValueError(f"Params inconnus : {sorted(unknown)}")
    unknown_stages = set(stages) - set(STAGES)
    if unknown_stages:
        raise ValueError(f"Stages inconnus : {sorted(unknown_stages)}")

    rng = np.random.default_rng(seed)
    drawn = {}
    for name, ranges in SAFE_RANGES.items():
        low, high = ranges[rng.integers(len(ranges))]
        drawn[name] = float(rng.uniform(low, high))
    drawn.update(overrides or {})
    return Params(seed=seed, stages=tuple(s for s in STAGES if s in stages), **drawn)
