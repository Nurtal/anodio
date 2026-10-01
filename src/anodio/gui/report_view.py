"""Report display: one Verdict per Axis, the Clinical features table and the other measures."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget

from anodio.core import Report

AXIS_LABELS = {
    "privacy": "Privacy",
    "content": "Contenu",
    "prosody": "Prosodie",
    "clinical": "Clinique",
    "informational": "Informatif",
}

CLINICAL_LABELS = {
    "f0_cv": "Coefficient de variation du F0",
    "f0_sd_semitones": "Écart-type du F0 (demi-tons)",
    "jitter_local_pct": "Jitter local (%)",
    "shimmer_local_pct": "Shimmer local (%)",
    "hnr_db": "HNR (dB)",
    "speech_rate_syll_per_s": "Débit (syll./s)",
    "pause_count": "Nombre de pauses",
    "mean_pause_s": "Durée moyenne des pauses (s)",
}

VERDICT_TEXT = {True: "✅ réussi", False: "❌ échec", None: "informatif"}


class ReportView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.verdicts = QLabel()
        self.verdicts.setWordWrap(True)
        self.clinical_table = QTableWidget(0, 4)
        self.clinical_table.setHorizontalHeaderLabels(["Feature clinique", "Original", "Anonymisé", "Écart"])
        self.clinical_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.clinical_table.verticalHeader().setVisible(False)
        self.details = QTextEdit()
        self.details.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.verdicts)
        layout.addWidget(self.clinical_table, 2)
        layout.addWidget(self.details, 1)

    def clear(self) -> None:
        self.verdicts.clear()
        self.clinical_table.setRowCount(0)
        self.details.clear()

    def show_report(self, report: Report) -> None:
        self.verdicts.setText(
            "   ".join(f"<b>{AXIS_LABELS.get(n, n)}</b> : {VERDICT_TEXT[a.verdict]}" for n, a in report.axes.items())
        )
        self._fill_clinical(report)
        self.details.setPlainText(format_details(report))

    def _fill_clinical(self, report: Report) -> None:
        clinical = report.axes.get("clinical")
        self.clinical_table.setRowCount(0)
        if clinical is None:
            return
        failing = {failure.split(" :")[0] for failure in clinical.failures}
        features = [key.removesuffix(".original") for key in clinical.measures if key.endswith(".original")]
        self.clinical_table.setRowCount(len(features))
        for row, name in enumerate(features):
            m = clinical.measures
            if f"{name}.absolute_deviation" in m:
                deviation = f"{_fmt(m[f'{name}.absolute_deviation'])} dB"
            else:
                value = m.get(f"{name}.relative_deviation")
                deviation = f"{100 * value:.1f} %" if isinstance(value, (int, float)) else "—"
            cells = [CLINICAL_LABELS.get(name, name), _fmt(m[f"{name}.original"]), _fmt(m[f"{name}.anonymized"]), deviation]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if name in failing:
                    item.setBackground(QColor(231, 76, 60, 60))
                self.clinical_table.setItem(row, col, item)


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def format_details(report: Report) -> str:
    lines = []
    if report.params:
        lines.append("Params")
        lines += [f"  {key} : {_fmt(value)}" for key, value in report.params.to_dict().items()]
        lines += [f"  ⚠ {name} hors de sa Safe range" for name in report.params.outside_safe_ranges()]
        lines.append("")
    for name, axis in report.axes.items():
        lines.append(f"{AXIS_LABELS.get(name, name)} : {VERDICT_TEXT[axis.verdict]}")
        lines += [f"  ⚠ {failure}" for failure in axis.failures]
        if name != "clinical":
            lines += [f"  {key} : {_fmt(value)}" for key, value in axis.measures.items()]
        lines.append("")
    return "\n".join(lines)
