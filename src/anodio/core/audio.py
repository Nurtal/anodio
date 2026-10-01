"""Mono audio container and .wav I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class AnodioError(Exception):
    """An error whose message can be shown as-is to the user."""


@dataclass(frozen=True, eq=False)
class Audio:
    samples: np.ndarray  # mono, float32
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def load_wav(path: str | Path) -> Audio:
    path = Path(path)
    if not path.exists():
        raise AnodioError(f"Fichier introuvable : {path}")
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AnodioError(f"Impossible de lire {path.name} comme un fichier .wav : {exc}") from exc
    return Audio(data.mean(axis=1).astype(np.float32), int(sample_rate))


def save_wav(path: str | Path, audio: Audio) -> None:
    sf.write(Path(path), audio.samples, audio.sample_rate, subtype="FLOAT")
