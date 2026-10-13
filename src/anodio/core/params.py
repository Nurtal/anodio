"""Params of an Anonymization, drawn deterministically from a Seed."""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from anodio.core import stages as registry
from anodio.core.stages import ParamSpec


def stage_names() -> tuple[str, ...]:
    """Names of the Stages of the Chain, in application order."""
    return tuple(stage.name for stage in registry.chain_stages())


def param_specs() -> dict[str, ParamSpec]:
    """Every Param by name, in the order they are drawn from the Seed."""
    return {spec.name: spec for stage in registry.REGISTRY for spec in stage.params}


@dataclass(frozen=True)
class Params:
    seed: int
    values: Mapping[str, float]
    stages: tuple[str, ...] = field(default_factory=stage_names)

    def __getattr__(self, name: str) -> float:
        """Each Param reads as an attribute, e.g. `params.pitch_semitones`."""
        try:
            return self.__dict__["values"][name]
        except KeyError:
            raise AttributeError(name) from None

    def outside_safe_ranges(self) -> list[str]:
        return [name for name, spec in param_specs().items() if not spec.is_safe(self.values[name])]

    def to_dict(self) -> dict:
        return {"seed": self.seed, **self.values, "stages": list(self.stages)}

    @classmethod
    def from_dict(cls, data: dict) -> Params:
        values = {name: value for name, value in data.items() if name not in ("seed", "stages")}
        return cls(seed=data["seed"], values=values, stages=tuple(data["stages"]))


def random_seed() -> int:
    return secrets.randbelow(2**31)


def draw_params(
    seed: int,
    overrides: Mapping[str, float] | None = None,
    stages: tuple[str, ...] | None = None,
) -> Params:
    """Draw every Param in its Safe range from the Seed, then apply the Overrides.

    `stages` are the enabled Stages (all of them by default), in any order.
    """
    specs, names = param_specs(), stage_names()
    enabled = names if stages is None else stages
    unknown = set(overrides or {}) - set(specs)
    if unknown:
        raise ValueError(f"Params inconnus : {sorted(unknown)}")
    unknown_stages = set(enabled) - set(names)
    if unknown_stages:
        raise ValueError(f"Stages inconnus : {sorted(unknown_stages)}")

    rng = np.random.default_rng(seed)
    drawn = {}
    for name, spec in specs.items():
        low, high = spec.safe_ranges[rng.integers(len(spec.safe_ranges))]
        drawn[name] = float(rng.uniform(low, high))
    drawn.update(overrides or {})
    return Params(seed=seed, values=drawn, stages=tuple(s for s in names if s in enabled))
