import numpy as np

from anodio.core import Audio, Report, anonymize, draw_params, evaluate


def test_recording_compared_to_itself_keeps_its_timing(speech):
    report = evaluate(speech, speech)

    prosody = report.axes["prosody"]
    assert prosody.verdict is True
    assert prosody.measures["pause_count_original"] == 1
    assert prosody.measures["pause_count_anonymized"] == 1
    assert prosody.measures["pause_boundary_error_ms"] == 0


def test_anonymization_keeps_the_timing(speech):
    report = evaluate(speech, anonymize(speech, draw_params(0)))

    assert report.axes["prosody"].verdict is True


def test_moved_pause_fails_the_timing_check(speech):
    delay = int(0.15 * speech.sample_rate)
    shifted = Audio(np.concatenate([np.zeros(delay, np.float32), speech.samples[:-delay]]), speech.sample_rate)

    report = evaluate(speech, shifted)

    assert report.axes["prosody"].verdict is False
    assert report.axes["prosody"].measures["pause_boundary_error_ms"] >= 140


def test_report_carries_params_and_round_trips_through_json(speech):
    params = draw_params(5)
    report = evaluate(speech, anonymize(speech, params), params=params)

    restored = Report.from_json(report.to_json())

    assert restored == report
    assert restored.params == params
