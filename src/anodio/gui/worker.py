"""Anonymization + evaluation off the GUI thread."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from anodio.core import AnodioError, Audio, ModelsMissingError, Params, anonymize, evaluate
from anodio.core.features import spectrogram
from anodio.core.models import EcapaEmbedder, WhisperTranscriber


class ModelCache:
    """Models are loaded once and reused across runs."""

    def __init__(self) -> None:
        self.embedder: EcapaEmbedder | None = None
        self.transcribers: dict[str, WhisperTranscriber] = {}


class AnonymizationWorker(QObject):
    progress = Signal(str)
    warning = Signal(str, str)
    finished = Signal(object, object, object)  # anonymized Audio, Report, (spectrogram, spectrogram)
    failed = Signal(str)

    def __init__(
        self,
        original: Audio,
        params: Params,
        thresholds: dict[str, dict[str, float]],
        whisper_size: str,
        models: ModelCache,
    ) -> None:
        super().__init__()
        self.original, self.params, self.thresholds = original, params, thresholds
        self.whisper_size, self.models = whisper_size, models

    def run(self) -> None:
        try:
            embedder, transcriber = self._load_models()
            self.progress.emit("Anonymisation…")
            anonymized = anonymize(self.original, self.params)
            self.progress.emit("Évaluation (Privacy, contenu, prosodie, clinique)…")
            report = evaluate(
                self.original,
                anonymized,
                params=self.params,
                thresholds=self.thresholds,
                embedder=embedder,
                transcriber=transcriber,
            )
            self.progress.emit("Préparation des visualisations…")
            spectrograms = (spectrogram(self.original), spectrogram(anonymized))
        except AnodioError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface any failure instead of killing the thread silently
            self.failed.emit(f"Erreur inattendue : {exc!r}")
            return
        self.finished.emit(anonymized, report, spectrograms)

    def _load_models(self):
        if self.models.embedder is None:
            self.progress.emit("Chargement du modèle ECAPA…")
            try:
                self.models.embedder = EcapaEmbedder()
            except ModelsMissingError as exc:
                self.warning.emit("Privacy non évaluée", str(exc))
        if self.whisper_size not in self.models.transcribers:
            self.progress.emit(f"Chargement de Whisper {self.whisper_size}…")
            try:
                self.models.transcribers[self.whisper_size] = WhisperTranscriber(self.whisper_size)
            except ModelsMissingError as exc:
                self.warning.emit("Contenu non évalué", str(exc))
        return self.models.embedder, self.models.transcribers.get(self.whisper_size)
