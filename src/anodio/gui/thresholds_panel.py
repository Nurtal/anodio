"""Editable Thresholds; every change re-judges the current Report without re-running anything."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QGroupBox, QLabel

from anodio.core import DEFAULT_THRESHOLDS
from anodio.core.thresholds import DeviationKey, DeviationKind
from anodio.gui.report_view import AXIS_LABELS, CLINICAL_LABELS

KEY_LABELS = {
    "max_cosine_similarity": "Similarité cosinus max.",
    "max_wer": "WER max.",
    "min_f0_correlation": "Corrélation F0 min.",
    "max_relative_deviation": "Écart relatif max. (par défaut)",
}


def key_label(key: str) -> str:
    deviation = DeviationKey.parse(key)
    if deviation is None or deviation.feature is None:
        return KEY_LABELS.get(key, key)
    unit = "écart absolu max." if deviation.kind is DeviationKind.ABSOLUTE else "écart relatif max."
    return f"{CLINICAL_LABELS.get(deviation.feature, deviation.feature)} — {unit}"


class ThresholdsPanel(QGroupBox):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__("Thresholds")
        self.boxes: dict[tuple[str, str], QDoubleSpinBox] = {}
        layout = QFormLayout(self)
        for axis, values in DEFAULT_THRESHOLDS.items():
            layout.addRow(QLabel(f"<b>{AXIS_LABELS.get(axis, axis)}</b>"))
            for key, value in values.items():
                box = QDoubleSpinBox()
                box.setDecimals(3)
                box.setRange(-1.0, 1000.0)
                box.setSingleStep(0.01 if abs(value) < 5 else 1.0)
                box.setValue(value)
                box.valueChanged.connect(self.changed)
                self.boxes[(axis, key)] = box
                layout.addRow(key_label(key), box)

    def thresholds(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for (axis, key), box in self.boxes.items():
            result.setdefault(axis, {})[key] = box.value()
        return result
