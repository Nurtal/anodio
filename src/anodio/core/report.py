"""Report: the outcome of evaluating an Original / Anonymized pair."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.metadata import version

from anodio.core.params import Params

Measure = float | int | str | None


@dataclass
class AxisResult:
    measures: dict[str, Measure]
    thresholds: dict[str, float]
    verdict: bool | None  # None for the Informational Axis, which is never judged
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "measures": self.measures,
            "thresholds": self.thresholds,
            "verdict": self.verdict,
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AxisResult:
        return cls(data["measures"], data["thresholds"], data["verdict"], data.get("failures", []))


@dataclass
class Report:
    axes: dict[str, AxisResult]
    params: Params | None = None
    anodio_version: str = field(default_factory=lambda: version("anodio"))

    def to_json(self) -> str:
        return json.dumps(
            {
                "anodio_version": self.anodio_version,
                "params": self.params.to_dict() if self.params else None,
                "axes": {name: axis.to_dict() for name, axis in self.axes.items()},
            },
            indent=2,
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> Report:
        data = json.loads(text)
        return cls(
            axes={name: AxisResult.from_dict(axis) for name, axis in data["axes"].items()},
            params=Params.from_dict(data["params"]) if data["params"] else None,
            anodio_version=data["anodio_version"],
        )
