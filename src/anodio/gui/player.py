"""A/B listening: switch between Original and Anonymized at the same position."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from anodio.core import Audio, save_wav


class ABPlayer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._dir = tempfile.TemporaryDirectory(prefix="anodio-")
        self._sources: dict[str, QUrl] = {}
        self._current = "original"
        self.player = QMediaPlayer(self)
        self.output = QAudioOutput(self)
        self.player.setAudioOutput(self.output)

        self.play_button = QPushButton("▶ Lecture")
        self.switch_button = QPushButton("⇄ Basculer A/B")
        self.stop_button = QPushButton("⏹")
        self.label = QLabel()
        self.play_button.clicked.connect(self.toggle_play)
        self.switch_button.clicked.connect(self.switch)
        self.stop_button.clicked.connect(self.player.stop)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        for widget in (self.play_button, self.switch_button, self.stop_button):
            row.addWidget(widget)
        row.addWidget(self.label, 1)
        self.set_audio(None, None)

    def set_audio(self, original: Audio | None, anonymized: Audio | None) -> None:
        self.player.stop()
        self._sources.clear()
        for name, audio in (("original", original), ("anonymized", anonymized)):
            if audio is not None:
                path = Path(self._dir.name) / f"{name}.wav"
                save_wav(path, audio)
                self._sources[name] = QUrl.fromLocalFile(str(path))
        self._current = "original"
        self.play_button.setEnabled("original" in self._sources)
        self.switch_button.setEnabled(len(self._sources) == 2)
        self._load(0)

    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def switch(self) -> None:
        position = self.player.position()
        playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self._current = "anonymized" if self._current == "original" else "original"
        self._load(position)
        if playing:
            self.player.play()

    def _load(self, position: int) -> None:
        source = self._sources.get(self._current)
        self.label.setText({"original": "A : original", "anonymized": "B : anonymisé"}[self._current] if source else "")
        if source is not None:
            self.player.setSource(source)
            self.player.setPosition(position)
