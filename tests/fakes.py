"""Deterministic stand-ins for the models, for the fast test suite."""

from __future__ import annotations

import numpy as np

from anodio.core import Audio


class SpectrumEmbedder:
    """A crude speaker embedding: the long-term log spectrum (it mostly reflects the formants)."""

    name = "fake-long-term-spectrum"

    def embed(self, audio: Audio) -> np.ndarray:
        frame = 512
        x = audio.samples[: len(audio.samples) // frame * frame].reshape(-1, frame)
        x = x[np.sqrt(np.mean(x**2, axis=1)) > 1e-4]
        spectrum = np.log(np.mean(np.abs(np.fft.rfft(x * np.hanning(frame), axis=1)) ** 2, axis=0) + 1e-12)
        return spectrum - spectrum.mean()


class ScriptedTranscriber:
    """Returns a fixed transcript per recording (keyed by object identity)."""

    name = "fake-scripted"

    def __init__(self, transcripts: dict[int, str] | None = None, default: str = "") -> None:
        self.transcripts = transcripts or {}
        self.default = default

    def transcribe(self, audio: Audio) -> str:
        return self.transcripts.get(id(audio), self.default)
