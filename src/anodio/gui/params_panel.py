"""Seed, Param sliders (locked to the Seed by default) and Stage toggles."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from anodio.core import Params, draw_params, param_specs, random_seed
from anodio.core.stages import ParamSpec, chain_stages

STEPS = 200


class ParamSlider(QWidget):
    changed = Signal()

    def __init__(self, spec: ParamSpec) -> None:
        super().__init__()
        self.spec = spec
        self.low, self.high = spec.bounds
        self.lock = QCheckBox("Seed")
        self.lock.setChecked(True)
        self.lock.setToolTip("Coché : valeur tirée de la Seed. Décoché : Override manuel.")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.setEnabled(False)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(110)
        self.lock.toggled.connect(lambda locked: self.slider.setEnabled(not locked))
        self.lock.toggled.connect(self.changed)
        self.slider.valueChanged.connect(self._refresh_label)
        self.slider.valueChanged.connect(self.changed)
        ranges = " ∪ ".join(f"[{lo:g} ; {hi:g}]" for lo, hi in spec.safe_ranges)
        self.setToolTip(f"Safe range : {ranges}")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(spec.label), 2)
        row.addWidget(self.slider, 3)
        row.addWidget(self.value_label)
        row.addWidget(self.lock)

    @property
    def overridden(self) -> bool:
        return not self.lock.isChecked()

    def value(self) -> float:
        return round(self.low + (self.high - self.low) * self.slider.value() / STEPS, 4)

    def set_value(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round((value - self.low) / (self.high - self.low) * STEPS))
        self.slider.blockSignals(False)
        self._refresh_label()

    def _refresh_label(self) -> None:
        value = self.value()
        outside = not self.spec.is_safe(value)
        self.value_label.setText(f"{value:+.3f}" + ("  ⚠ hors plage" if outside else ""))
        self.value_label.setStyleSheet("color: #c0392b;" if outside else "")


class ParamsPanel(QGroupBox):
    """Produces the Params of the next Anonymization."""

    def __init__(self) -> None:
        super().__init__("Paramètres")
        self.seed_box = QSpinBox()
        self.seed_box.setRange(0, 2**31 - 1)
        self.seed_box.setValue(random_seed())
        self.seed_box.setToolTip(
            "La même Seed donne la même pseudo-voix : les enregistrements anonymisés avec une Seed "
            "commune peuvent être reliés entre eux (Linkability). Gardez les Seeds confidentielles."
        )
        random_button = QPushButton("Seed aléatoire")
        random_button.clicked.connect(lambda: self.seed_box.setValue(random_seed()))

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed :"))
        seed_row.addWidget(self.seed_box, 1)
        seed_row.addWidget(random_button)

        stages = chain_stages()
        self.stage_boxes = {stage.name: QCheckBox(stage.label) for stage in stages}
        stages_row = QGridLayout()
        for i, box in enumerate(self.stage_boxes.values()):
            box.setChecked(True)
            stages_row.addWidget(box, 0, i)

        self.sliders = {name: ParamSlider(spec) for name, spec in param_specs().items()}
        self._stage_of = {spec.name: stage.name for stage in stages for spec in stage.params}
        layout = QVBoxLayout(self)
        layout.addLayout(seed_row)
        layout.addWidget(QLabel("Stages actifs :"))
        layout.addLayout(stages_row)
        for slider in self.sliders.values():
            layout.addWidget(slider)

        self.seed_box.valueChanged.connect(self._sync_from_seed)
        for slider in self.sliders.values():
            slider.changed.connect(self._sync_from_seed)
        for stage, box in self.stage_boxes.items():
            box.toggled.connect(self._sync_stage_enabled)
        self._sync_from_seed()

    def params(self) -> Params:
        overrides = {name: s.value() for name, s in self.sliders.items() if s.overridden}
        stages = tuple(stage for stage, box in self.stage_boxes.items() if box.isChecked())
        return draw_params(self.seed_box.value(), overrides=overrides, stages=stages)

    def _sync_from_seed(self) -> None:
        drawn = draw_params(self.seed_box.value())
        for name, slider in self.sliders.items():
            if not slider.overridden:
                slider.set_value(getattr(drawn, name))

    def _sync_stage_enabled(self) -> None:
        for name, slider in self.sliders.items():
            slider.setEnabled(self.stage_boxes[self._stage_of[name]].isChecked())
