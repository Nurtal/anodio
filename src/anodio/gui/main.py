"""Minimal desktop GUI: load an Original, pick a Seed, anonymize, export."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from anodio.core import (
    AnodioError,
    Audio,
    ModelsMissingError,
    Report,
    SpeakerEmbedder,
    Transcriber,
    anonymize,
    apply_thresholds,
    evaluate,
    load_wav,
    save_wav,
)
from anodio.core.models import DEFAULT_WHISPER_SIZE, WHISPER_SIZES, EcapaEmbedder, WhisperTranscriber
from anodio.gui.params_panel import ParamsPanel
from anodio.gui.report_view import ReportView
from anodio.gui.thresholds_panel import ThresholdsPanel

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("anodio — anonymisation de la voix")
        self.original: Audio | None = None
        self.original_path: Path | None = None
        self.anonymized: Audio | None = None
        self.report: Report | None = None
        self._embedder: SpeakerEmbedder | None = None
        self._transcribers: dict[str, Transcriber] = {}
        self._warned_missing: set[str] = set()

        self.file_label = QLabel("Aucun fichier chargé.")
        load_button = QPushButton("Charger un .wav…")
        load_button.clicked.connect(self.load)

        self.params_panel = ParamsPanel()

        self.whisper_size = QComboBox()
        self.whisper_size.addItems(WHISPER_SIZES)
        self.whisper_size.setCurrentText(DEFAULT_WHISPER_SIZE)
        models_row = QHBoxLayout()
        models_row.addWidget(QLabel("Modèle Whisper (contenu) :"))
        models_row.addWidget(self.whisper_size)
        models_row.addStretch(1)

        self.anonymize_button = QPushButton("Anonymiser")
        self.anonymize_button.setEnabled(False)
        self.anonymize_button.clicked.connect(self.run_anonymization)

        self.export_button = QPushButton("Exporter…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)

        self.results = ReportView()
        self.thresholds_panel = ThresholdsPanel()
        self.thresholds_panel.changed.connect(self.rejudge)
        scope = QLabel(
            "La Privacy est évaluée face à un système automatique de vérification du locuteur (ECAPA) "
            "uniquement, pas face à un auditeur humain."
        )
        scope.setWordWrap(True)
        scope.setStyleSheet("color: gray; font-size: 11px;")

        controls = QVBoxLayout()
        controls.addWidget(load_button)
        controls.addWidget(self.file_label)
        controls.addWidget(self.params_panel)
        controls.addLayout(models_row)
        controls.addWidget(self.thresholds_panel)
        controls.addStretch(1)
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_widget)

        left = QVBoxLayout()
        left.addWidget(scroll, 1)
        left.addWidget(self.anonymize_button)
        left.addWidget(self.export_button)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.results, "Rapport")
        right = QVBoxLayout()
        right.addWidget(self.tabs, 1)
        right.addWidget(scope)

        layout = QHBoxLayout()
        layout.addLayout(left, 2)
        layout.addLayout(right, 3)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(1400, 900)

    def load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Charger un enregistrement", "", "Fichiers WAV (*.wav)")
        if not path:
            return
        try:
            self.original = load_wav(path)
        except AnodioError as exc:
            QMessageBox.critical(self, "Erreur de chargement", str(exc))
            return
        self.original_path = Path(path)
        self.anonymized = self.report = None
        self.export_button.setEnabled(False)
        self.anonymize_button.setEnabled(True)
        self.results.clear()
        self.file_label.setText(
            f"{self.original_path.name} — {self.original.duration:.1f} s, {self.original.sample_rate} Hz"
        )

    def run_anonymization(self) -> None:
        assert self.original is not None
        params = self.params_panel.params()
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            embedder = self.embedder()
            transcriber = self.transcriber()
            self.anonymized = anonymize(self.original, params)
            self.report = evaluate(
                self.original,
                self.anonymized,
                params=params,
                thresholds=self.thresholds_panel.thresholds(),
                embedder=embedder,
                transcriber=transcriber,
            )
        finally:
            QGuiApplication.restoreOverrideCursor()
        self.export_button.setEnabled(True)
        self.results.show_report(self.report)

    def rejudge(self) -> None:
        if self.report is not None:
            self.report = apply_thresholds(self.report, self.thresholds_panel.thresholds())
            self.results.show_report(self.report)

    def embedder(self) -> SpeakerEmbedder | None:
        """The ASV attacker, loaded on first use. Without models, the Privacy Axis is skipped."""
        if self._embedder is None:
            try:
                self._embedder = EcapaEmbedder()
            except ModelsMissingError as exc:
                self._warn_missing("Privacy non évaluée", exc)
        return self._embedder

    def transcriber(self) -> Transcriber | None:
        """Whisper of the selected size, loaded on first use. Without it, the Content Axis is skipped."""
        size = self.whisper_size.currentText()
        if size not in self._transcribers:
            try:
                self._transcribers[size] = WhisperTranscriber(size)
            except ModelsMissingError as exc:
                self._warn_missing("Contenu non évalué", exc)
                return None
        return self._transcribers[size]

    def _warn_missing(self, title: str, error: ModelsMissingError) -> None:
        if str(error) in self._warned_missing:
            return
        self._warned_missing.add(str(error))
        QGuiApplication.restoreOverrideCursor()
        QMessageBox.warning(self, title, str(error))
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def export(self) -> None:
        assert self.anonymized is not None and self.report is not None and self.original_path is not None
        default = str(self.original_path.with_name(f"{self.original_path.stem}_anon.wav"))
        path, _ = QFileDialog.getSaveFileName(self, "Exporter la version anonymisée", default, "Fichiers WAV (*.wav)")
        if not path:
            return
        wav_path = Path(path).with_suffix(".wav")
        save_wav(wav_path, self.anonymized)
        wav_path.with_suffix(".json").write_text(self.report.to_json(), encoding="utf-8")
        QMessageBox.information(self, "Export", f"Exporté :\n{wav_path}\n{wav_path.with_suffix('.json')}")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
