import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from synth import utterance  # noqa: E402


@pytest.fixture
def speech():
    return utterance()


@pytest.fixture(scope="session")
def app():
    """The QApplication that any widget needs."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
