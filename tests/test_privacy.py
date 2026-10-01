import numpy as np
import pytest
from fakes import SpectrumEmbedder

from anodio.core import AnodioError, ModelsMissingError, anonymize, draw_params, evaluate
from anodio.core.models import EcapaEmbedder, models_dir


def test_recording_compared_to_itself_fails_privacy(speech):
    report = evaluate(speech, speech, embedder=SpectrumEmbedder())

    privacy = report.axes["privacy"]
    assert privacy.verdict is False
    assert privacy.measures["cosine_similarity"] == pytest.approx(1.0)
    assert privacy.measures["embedder"] == SpectrumEmbedder.name
    assert privacy.failures


def test_anonymization_lowers_the_similarity(speech):
    report = evaluate(speech, anonymize(speech, draw_params(0)), embedder=SpectrumEmbedder())

    assert report.axes["privacy"].measures["cosine_similarity"] < 0.99


def test_privacy_verdict_follows_its_threshold(speech):
    anonymized = anonymize(speech, draw_params(0))
    lenient = {"privacy": {"max_cosine_similarity": 1.01}}

    report = evaluate(speech, anonymized, embedder=SpectrumEmbedder(), thresholds=lenient)

    assert report.axes["privacy"].verdict is True


def test_without_embedder_there_is_no_privacy_axis(speech):
    assert "privacy" not in evaluate(speech, speech).axes


def test_missing_models_raise_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ANODIO_MODELS_DIR", str(tmp_path))

    with pytest.raises(ModelsMissingError, match="anodio download-models") as raised:
        EcapaEmbedder()
    assert isinstance(raised.value, AnodioError)


def _ecapa_or_skip():
    try:
        return EcapaEmbedder()
    except ModelsMissingError:
        pytest.skip(f"Modèles absents de {models_dir()}")


@pytest.mark.slow
def test_real_ecapa_recognises_the_same_recording(speech):
    embedder = _ecapa_or_skip()

    same = evaluate(speech, speech, embedder=embedder).axes["privacy"].measures["cosine_similarity"]
    anon = evaluate(speech, anonymize(speech, draw_params(0)), embedder=embedder)

    assert same == pytest.approx(1.0, abs=1e-4)
    assert anon.axes["privacy"].measures["cosine_similarity"] < same


@pytest.mark.slow
def test_real_ecapa_runs_offline(speech, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    embedder = _ecapa_or_skip()

    assert np.all(np.isfinite(embedder.embed(speech)))
