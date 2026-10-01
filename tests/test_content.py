import pytest
from fakes import ScriptedTranscriber

from anodio.core import ModelsMissingError, anonymize, draw_params, evaluate
from anodio.core.models import WhisperTranscriber, models_dir


def _content(original, anonymized, transcripts, **kwargs):
    transcriber = ScriptedTranscriber({id(original): transcripts[0], id(anonymized): transcripts[1]})
    return evaluate(original, anonymized, transcriber=transcriber, **kwargs).axes["content"]


@pytest.fixture
def anonymized(speech):
    return anonymize(speech, draw_params(0))


def test_identical_transcripts_pass_with_zero_wer(speech, anonymized):
    content = _content(speech, anonymized, ("le chat dort", "le chat dort"))

    assert content.verdict is True
    assert content.measures["wer"] == 0


def test_wer_counts_word_errors_against_the_original_transcript(speech, anonymized):
    content = _content(speech, anonymized, ("le petit chat dort ici", "le chien dort"))

    # One substitution (petit -> chien) and two deletions (chat, ici) over five words.
    assert content.measures["wer"] == pytest.approx(3 / 5)
    assert content.verdict is False
    assert content.failures


def test_case_and_punctuation_are_ignored(speech, anonymized):
    content = _content(speech, anonymized, ("Le chat, l'oiseau dort.", "le chat l'oiseau dort"))

    assert content.measures["wer"] == 0


def test_both_transcripts_are_reported(speech, anonymized):
    content = _content(speech, anonymized, ("bonjour", "bonsoir"))

    assert content.measures["transcript_original"] == "bonjour"
    assert content.measures["transcript_anonymized"] == "bonsoir"
    assert content.measures["transcriber"] == ScriptedTranscriber.name


def test_wer_threshold_is_configurable(speech, anonymized):
    content = _content(speech, anonymized, ("a b c d", "a b c e"), thresholds={"content": {"max_wer": 0.3}})

    assert content.measures["wer"] == pytest.approx(0.25)
    assert content.verdict is True


def test_empty_original_transcript_cannot_be_judged(speech, anonymized):
    content = _content(speech, anonymized, ("", "bonjour"))

    assert content.verdict is False
    assert content.measures["wer"] is None


def test_without_transcriber_there_is_no_content_axis(speech):
    assert "content" not in evaluate(speech, speech).axes


def test_missing_whisper_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ANODIO_MODELS_DIR", str(tmp_path))

    with pytest.raises(ModelsMissingError, match="anodio download-models"):
        WhisperTranscriber("small")


@pytest.mark.slow
def test_real_whisper_transcribes_offline_without_hallucinating_on_silence(speech, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        transcriber = WhisperTranscriber("small")
    except ModelsMissingError:
        pytest.skip(f"Modèles absents de {models_dir()}")
    from anodio.core import Audio
    import numpy as np

    assert transcriber.transcribe(Audio(np.zeros(32000, np.float32), 16000)) == ""
    assert isinstance(transcriber.transcribe(speech), str)
