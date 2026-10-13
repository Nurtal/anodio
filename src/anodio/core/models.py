"""Access to the pretrained models, behind small injectable interfaces.

Models are downloaded once (`anodio download-models`) into a local cache, then
loaded from disk only: no network access at run time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np

from anodio.core.audio import ANALYSIS_SAMPLE_RATE, AnodioError, Audio, resample

ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"


class ModelsMissingError(AnodioError):
    pass


class SpeakerEmbedder(Protocol):
    name: str

    def embed(self, audio: Audio) -> np.ndarray: ...


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: Audio) -> str: ...


WHISPER_SIZES = ("tiny", "base", "small", "medium")
DEFAULT_WHISPER_SIZE = "small"


def models_dir() -> Path:
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "anodio" / "models"
    return Path(os.environ.get("ANODIO_MODELS_DIR", default))


def _require(path: Path, what: str) -> Path:
    if not path.exists() or not any(path.iterdir()):
        raise ModelsMissingError(
            f"Le modèle {what} est absent de {path}. Lancez une fois « anodio download-models » "
            "(connexion requise), puis anodio fonctionne hors ligne."
        )
    return path


def to_model_rate(audio: Audio) -> np.ndarray:
    return resample(audio, ANALYSIS_SAMPLE_RATE).samples.astype(np.float32)


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN trained on VoxCeleb: the ASV attacker of the Privacy Axis."""

    name = ECAPA_REPO

    def __init__(self) -> None:
        path = _require(models_dir() / "ecapa", "ECAPA (vérification du locuteur)")
        from speechbrain.inference.speaker import EncoderClassifier

        self._model = EncoderClassifier.from_hparams(source=str(path), savedir=str(path), run_opts={"device": "cpu"})

    def embed(self, audio: Audio) -> np.ndarray:
        import torch

        with torch.no_grad():
            embedding = self._model.encode_batch(torch.from_numpy(to_model_rate(audio))[None, :])
        return embedding.squeeze().numpy()


class WhisperTranscriber:
    """faster-whisper, in French, int8 on CPU."""

    def __init__(self, size: str = DEFAULT_WHISPER_SIZE) -> None:
        if size not in WHISPER_SIZES:
            raise AnodioError(f"Taille de Whisper inconnue : {size} (choix : {', '.join(WHISPER_SIZES)}).")
        path = _require(models_dir() / f"whisper-{size}", f"Whisper {size} (transcription)")
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper-{size}"
        self._model = WhisperModel(str(path), device="cpu", compute_type="int8")

    def transcribe(self, audio: Audio) -> str:
        # The VAD filter and not conditioning on previous text keep Whisper from
        # hallucinating sentences over silences.
        segments, _ = self._model.transcribe(
            to_model_rate(audio), language="fr", vad_filter=True, condition_on_previous_text=False
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def download_models(target: Path | None = None, whisper_size: str = DEFAULT_WHISPER_SIZE) -> list[Path]:
    """Fetch every model into the local cache. The only step that needs a network connection."""
    from huggingface_hub import snapshot_download

    target = target or models_dir()
    ecapa = Path(snapshot_download(ECAPA_REPO, local_dir=target / "ecapa"))
    whisper = Path(
        snapshot_download(f"Systran/faster-whisper-{whisper_size}", local_dir=target / f"whisper-{whisper_size}")
    )
    return [ecapa, whisper]
