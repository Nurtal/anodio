"""Desktop GUI: load an Original, set the Params, anonymize, inspect the Report, export."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from anodio.core import AnodioError, Audio, Report, apply_thresholds, load_wav, save_wav
from anodio.core.models import DEFAULT_WHISPER_SIZE, WHISPER_SIZES
from anodio.gui.params_panel import ParamsPanel
from anodio.gui.player import ABPlayer
from anodio.gui.plots import F0View, HistogramView, SpectrogramView
from anodio.gui.report_view import ReportView
from anodio.gui.thresholds_panel import ThresholdsPanel
from anodio.gui.worker import AnonymizationWorker, ModelCache

ABOUT = """<h3>anodio</h3>
<p>Anonymisation de la voix avec évaluation de la Privacy et de l'utilité.</p>
<p><b>Limites à garder en tête :</b></p>
<ul>
<li>La Privacy est évaluée face à un système automatique de vérification du locuteur (ECAPA)
uniquement, ni face à un auditeur humain, ni face à un attaquant entraîné sur des voix anonymisées.</li>
<li>Le choix du traitement du signal (pour préserver les features cliniques) donne une Privacy
modérée : mesurez-la sur vos données avec <code>anodio evaluate-corpus</code>.</li>
<li>Une même Seed donne la même pseudo-voix : des enregistrements anonymisés avec une Seed commune
peuvent être reliés entre eux (Linkability). Gardez les Seeds confidentielles.</li>
<li>Le seuil cosinus de 0,25 est indicatif et doit être recalibré pour votre langue et votre modèle.</li>
<li>Le HNR dépend en partie du conduit vocal, que l'anonymisation déplace volontairement.</li>
</ul>"""

PRIVACY_SCOPE = (
    "La Privacy est évaluée face à un système automatique de vérification du locuteur (ECAPA) "
    "uniquement, pas face à un auditeur humain."
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("anodio — anonymisation de la voix")
        self.original: Audio | None = None
        self.original_path: Path | None = None
        self.anonymized: Audio | None = None
        self.report: Report | None = None
        self.models = ModelCache()
        self._warned: set[str] = set()
        self._thread: QThread | None = None
        self._worker: AnonymizationWorker | None = None

        self.file_label = QLabel("Aucun fichier chargé.")
        load_button = QPushButton("Charger un .wav…")
        load_button.clicked.connect(self.load)
        self.params_panel = ParamsPanel()
        self.whisper_size = QComboBox()
        self.whisper_size.addItems(WHISPER_SIZES)
        self.whisper_size.setCurrentText(DEFAULT_WHISPER_SIZE)
        self.thresholds_panel = ThresholdsPanel()
        self.thresholds_panel.changed.connect(self.rejudge)

        self.anonymize_button = QPushButton("Anonymiser")
        self.anonymize_button.setEnabled(False)
        self.anonymize_button.clicked.connect(self.run_anonymization)
        self.export_button = QPushButton("Exporter…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        about_button = QPushButton("À propos et limites…")
        about_button.clicked.connect(lambda: QMessageBox.about(self, "À propos d'anodio", ABOUT))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        self.status = QLabel()

        self.results = ReportView()
        self.spectrograms = SpectrogramView()
        self.f0_view = F0View()
        self.histograms = HistogramView()
        self.player = ABPlayer()
        scope = QLabel(PRIVACY_SCOPE)
        scope.setWordWrap(True)
        scope.setStyleSheet("color: gray; font-size: 11px;")

        models_row = QHBoxLayout()
        models_row.addWidget(QLabel("Modèle Whisper (contenu) :"))
        models_row.addWidget(self.whisper_size)
        models_row.addStretch(1)
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
        left.addWidget(self.progress_bar)
        left.addWidget(self.status)
        left.addWidget(self.export_button)
        left.addWidget(about_button)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.results, "Rapport")
        self.tabs.addTab(self.spectrograms, "Spectrogrammes")
        self.tabs.addTab(self.f0_view, "Intonation (F0)")
        self.tabs.addTab(self.histograms, "Distributions")
        right = QVBoxLayout()
        right.addWidget(self.player)
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
        if path:
            self.load_path(Path(path))

    def load_path(self, path: Path) -> None:
        try:
            self.original = load_wav(path)
        except AnodioError as exc:
            QMessageBox.critical(self, "Erreur de chargement", str(exc))
            return
        self.original_path = path
        self.anonymized = self.report = None
        self.export_button.setEnabled(False)
        self.anonymize_button.setEnabled(True)
        self.results.clear()
        for view in (self.spectrograms, self.f0_view, self.histograms):
            view.clear()
        self.player.set_audio(self.original, None)
        self.file_label.setText(f"{path.name} — {self.original.duration:.1f} s, {self.original.sample_rate} Hz")

    def run_anonymization(self) -> None:
        assert self.original is not None
        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = AnonymizationWorker(
            self.original,
            self.params_panel.params(),
            self.thresholds_panel.thresholds(),
            self.whisper_size.currentText(),
            self.models,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.warning.connect(self._warn)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        for signal in (self._worker.finished, self._worker.failed):
            signal.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    def wait_for_worker(self) -> None:
        """Block until the current run is over (used by scripts and tests)."""
        if self._thread is not None:
            self._thread.wait()
            QApplication.processEvents()

    def _on_finished(self, anonymized: Audio, report: Report, spectrograms) -> None:
        assert self.original is not None
        self.anonymized, self.report = anonymized, report
        self._set_busy(False)
        self.status.setText("Terminé.")
        self.export_button.setEnabled(True)
        self.results.show_report(report)
        self.spectrograms.show(*spectrograms)
        self.f0_view.show(report)
        self.histograms.show(report)
        self.player.set_audio(self.original, anonymized)

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self.status.setText("Échec.")
        QMessageBox.critical(self, "Erreur", message)

    def _set_busy(self, busy: bool) -> None:
        self.anonymize_button.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if busy:
            self.export_button.setEnabled(False)

    def _warn(self, title: str, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            QMessageBox.warning(self, title, message)

    def rejudge(self) -> None:
        if self.report is not None:
            self.report = apply_thresholds(self.report, self.thresholds_panel.thresholds())
            self.results.show_report(self.report)

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
