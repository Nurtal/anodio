"""Minimal desktop GUI: load an Original, pick a Seed, anonymize, export."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from anodio.core import (
    AnodioError,
    Audio,
    ModelsMissingError,
    Report,
    SpeakerEmbedder,
    anonymize,
    evaluate,
    load_wav,
    save_wav,
)
from anodio.core.models import EcapaEmbedder
from anodio.gui.params_panel import ParamsPanel
from anodio.gui.report_view import ReportView

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("anodio — anonymisation de la voix")
        self.original: Audio | None = None
        self.original_path: Path | None = None
        self.anonymized: Audio | None = None
        self.report: Report | None = None
        self._embedder: SpeakerEmbedder | None = None

        self.file_label = QLabel("Aucun fichier chargé.")
        load_button = QPushButton("Charger un .wav…")
        load_button.clicked.connect(self.load)

        self.params_panel = ParamsPanel()

        self.anonymize_button = QPushButton("Anonymiser")
        self.anonymize_button.setEnabled(False)
        self.anonymize_button.clicked.connect(self.run_anonymization)

        self.export_button = QPushButton("Exporter…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)

        self.results = ReportView()
        scope = QLabel(
            "La Privacy est évaluée face à un système automatique de vérification du locuteur (ECAPA) "
            "uniquement, pas face à un auditeur humain."
        )
        scope.setWordWrap(True)
        scope.setStyleSheet("color: gray; font-size: 11px;")

        layout = QVBoxLayout()
        layout.addWidget(load_button)
        layout.addWidget(self.file_label)
        layout.addWidget(self.params_panel)
        layout.addWidget(self.anonymize_button)
        layout.addWidget(self.results, 1)
        layout.addWidget(scope)
        layout.addWidget(self.export_button)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(820, 900)

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
            self.anonymized = anonymize(self.original, params)
            self.report = evaluate(self.original, self.anonymized, params=params, embedder=embedder)
        finally:
            QGuiApplication.restoreOverrideCursor()
        self.export_button.setEnabled(True)
        self.results.show_report(self.report)

    def embedder(self) -> SpeakerEmbedder | None:
        """The ASV attacker, loaded on first use. Without models, the Privacy Axis is skipped."""
        if self._embedder is None:
            try:
                self._embedder = EcapaEmbedder()
            except ModelsMissingError as exc:
                QGuiApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "Privacy non évaluée", str(exc))
                QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        return self._embedder

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
