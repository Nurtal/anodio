"""Mono audio container and .wav I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AnodioError(Exception):
    """An error whose message can be shown as-is to the user."""


# Every measure of a Report is taken at this rate, whatever the Original's own sample rate.
ANALYSIS_SAMPLE_RATE = 16000

# Praat's pitch analysis needs several periods of the lowest F0 searched.
MIN_DURATION_SECONDS = 0.5


@dataclass(frozen=True, eq=False)
class Audio:
    samples: np.ndarray  # mono, float32
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def resample(audio: Audio, sample_rate: int) -> Audio:
    """A copy of `audio` at `sample_rate` (the recording itself when it is already at that rate)."""
    if audio.sample_rate == sample_rate:
        return audio
    divisor = int(np.gcd(sample_rate, audio.sample_rate))
    samples = resample_poly(audio.samples, sample_rate // divisor, audio.sample_rate // divisor)
    return Audio(samples.astype(np.float32), sample_rate)


def load_wav(path: str | Path) -> Audio:
    path = Path(path)
    if not path.exists():
        raise AnodioError(f"Fichier introuvable : {path}")
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AnodioError(f"Impossible de lire {path.name} comme un fichier .wav : {exc}") from exc
    audio = Audio(data.mean(axis=1).astype(np.float32), int(sample_rate))
    if audio.duration < MIN_DURATION_SECONDS:
        raise AnodioError(
            f"{path.name} est trop court ({audio.duration:.2f} s) : il faut au moins {MIN_DURATION_SECONDS} s de signal."
        )
    return audio


def save_wav(path: str | Path, audio: Audio) -> None:
    sf.write(Path(path), audio.samples, audio.sample_rate, subtype="FLOAT")
