"""Thresholds of the Clinical Axis, which limit the deviation of each Clinical feature.

In a Report and on the command line, they are flat `key: value` pairs whose key is
`max_relative_deviation` (the default for every feature), `max_relative_deviation:<feature>`
or `max_absolute_deviation:<feature>`. This module is the only one that reads those keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class DeviationKind(StrEnum):
    RELATIVE = "relative"  # |anonymized - original| / |original|
    ABSOLUTE = "absolute"  # |anonymized - original|, in the unit of the feature


@dataclass(frozen=True)
class DeviationKey:
    kind: DeviationKind
    feature: str | None = None  # None: the default for every feature

    @classmethod
    def parse(cls, key: str) -> DeviationKey | None:
        """None when `key` does not limit a deviation."""
        name, separator, feature = key.partition(":")
        for kind in DeviationKind:
            if name == f"max_{kind}_deviation" and bool(separator) == bool(feature):
                return cls(kind, feature or None)
        return None

    def __str__(self) -> str:
        return f"max_{self.kind}_deviation" + (f":{self.feature}" if self.feature else "")


@dataclass(frozen=True)
class DeviationLimit:
    kind: DeviationKind
    value: float


class ClinicalThresholds:
    def __init__(self, thresholds: Mapping[str, float]) -> None:
        parsed = ((DeviationKey.parse(key), value) for key, value in thresholds.items())
        self._limits = {key: value for key, value in parsed if key is not None}

    def limit_for(self, feature: str) -> DeviationLimit:
        """An absolute limit wins over a relative one, which wins over the default."""
        absolute = self._limits.get(DeviationKey(DeviationKind.ABSOLUTE, feature))
        if absolute is not None:
            return DeviationLimit(DeviationKind.ABSOLUTE, absolute)
        default = self._limits[DeviationKey(DeviationKind.RELATIVE)]
        relative = self._limits.get(DeviationKey(DeviationKind.RELATIVE, feature), default)
        return DeviationLimit(DeviationKind.RELATIVE, relative)
