"""Visual comparison of the Original and the Anonymized recording."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from anodio.core import Report

Spectrogram = tuple[np.ndarray, np.ndarray, np.ndarray]  # times, frequencies, dB
MAX_COLUMNS = 2000

ORIGINAL_COLOR = "#2471a3"
ANONYMIZED_COLOR = "#d35400"


class _Canvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(6, 4), layout="constrained")
        super().__init__(self.figure)

    def clear(self) -> None:
        self.figure.clear()
        self.draw_idle()


class SpectrogramView(_Canvas):
    """Side by side spectrograms: the formants move, the timing does not."""

    def show(self, original: Spectrogram, anonymized: Spectrogram) -> None:  # type: ignore[override]
        self.figure.clear()
        axes = self.figure.subplots(1, 2, sharex=True, sharey=True)
        for ax, (times, freqs, db), title in ((axes[0], original, "Original"), (axes[1], anonymized, "Anonymisé")):
            # Max-pool the columns of long recordings: the image only has so many pixels.
            step = max(1, int(np.ceil(db.shape[1] / MAX_COLUMNS)))
            if step > 1:
                usable = db.shape[1] // step * step
                db = db[:, :usable].reshape(db.shape[0], -1, step).max(axis=2)
            ax.imshow(
                db,
                origin="lower",
                aspect="auto",
                extent=(times[0], times[-1], freqs[0], freqs[-1]),
                vmin=-80,
                vmax=0,
                cmap="magma",
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.set_xlabel("Temps (s)")
        axes[0].set_ylabel("Fréquence (Hz)")
        self.draw_idle()


class F0View(_Canvas):
    """Normalized F0 contours (semitones from each voice's median), superimposed."""

    def show(self, report: Report) -> None:  # type: ignore[override]
        self.figure.clear()
        ax = self.figure.subplots()
        analysis = report.analysis
        if analysis is None:
            return
        for contour, label, color in (
            (analysis.original_f0, "Original", ORIGINAL_COLOR),
            (analysis.anonymized_f0, "Anonymisé", ANONYMIZED_COLOR),
        ):
            ax.plot(contour.times, contour.semitones_from_median(), ".", ms=2, label=label, color=color)
        correlation = report.axes["prosody"].measures.get("f0_correlation")
        title = "Contours de F0 normalisés"
        if isinstance(correlation, float):
            title += f" — corrélation {correlation:.2f}"
        ax.set_title(title)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Demi-tons / médiane du locuteur")
        ax.legend(loc="upper right")
        self.draw_idle()


class HistogramView(_Canvas):
    """Feature distributions behind the Informational distances."""

    def show(self, report: Report) -> None:  # type: ignore[override]
        self.figure.clear()
        analysis = report.analysis
        if analysis is None:
            return
        info = report.axes["informational"].measures
        panels = [
            ("F0 (Hz)", "f0", analysis.original_f0.hz[analysis.original_f0.voiced],
             analysis.anonymized_f0.hz[analysis.anonymized_f0.voiced]),
            ("Énergie (dB)", "energy", analysis.original_energy_db[analysis.original_energy_db > -35],
             analysis.anonymized_energy_db[analysis.anonymized_energy_db > -35]),
            ("MFCC c1", "mfcc", analysis.original_mfcc[:, 0], analysis.anonymized_mfcc[:, 0]),
            ("MFCC c2", "mfcc", analysis.original_mfcc[:, 1], analysis.anonymized_mfcc[:, 1]),
        ]
        for ax, (title, key, a, b) in zip(self.figure.subplots(2, 2).flat, panels):
            if len(a) and len(b):
                bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 40)
                ax.hist(a, bins=bins, alpha=0.6, density=True, color=ORIGINAL_COLOR, label="Original")
                ax.hist(b, bins=bins, alpha=0.6, density=True, color=ANONYMIZED_COLOR, label="Anonymisé")
            js = info.get(f"{key}.js")
            ax.set_title(f"{title} — JS {js:.2f}" if isinstance(js, float) else title, fontsize=9)
            ax.tick_params(labelsize=7)
        self.figure.axes[0].legend(fontsize=7)
        self.draw_idle()
