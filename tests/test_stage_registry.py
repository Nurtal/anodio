"""A Stage is described once, in the registry: the Params, the Chain, the CLI and the GUI follow."""

import json

import numpy as np
import pytest

from anodio.cli.main import main
from anodio.core import anonymize, draw_params, param_specs, save_wav, stage_names, stages
from anodio.core.stages import ParamSpec, Stage


def _invert(samples, sample_rate, values, seed):
    return -values["invert_gain"] * samples


INVERT = Stage(
    name="invert",
    label="Inversion de polarité",
    position=99,
    params=(ParamSpec("invert_gain", "Gain de l'inversion", safe_ranges=((0.5, 0.8),), identity=1.0, bounds=(0.0, 1.0)),),
    apply=_invert,
)


@pytest.fixture
def registered(monkeypatch):
    """The only thing a new Stage needs: an entry in the registry."""
    monkeypatch.setattr(stages, "REGISTRY", (*stages.REGISTRY, INVERT))


def test_registry_describes_the_chain_in_application_order():
    assert stage_names() == ("psola", "mcadams", "vtln")
    assert list(param_specs()) == ["mcadams_alpha", "pitch_semitones", "formant_ratio", "vtln_alpha"]


def test_registered_stage_joins_the_chain_and_gets_its_params(registered):
    before = draw_params(3, stages=("psola",))

    params = draw_params(3)

    assert params.stages == ("psola", "mcadams", "vtln", "invert")
    assert 0.5 <= params.invert_gain <= 0.8
    assert draw_params(3, overrides={"invert_gain": 1.0}).outside_safe_ranges() == ["invert_gain"]
    assert params.to_dict()["invert_gain"] == params.invert_gain
    assert type(params).from_dict(json.loads(json.dumps(params.to_dict()))) == params
    assert before.stages == ("psola",)


def test_registering_a_stage_keeps_the_pseudo_voice_of_every_seed(registered, monkeypatch):
    with_stage = draw_params(7).values
    monkeypatch.undo()

    assert {name: with_stage[name] for name in draw_params(7).values} == draw_params(7).values


def test_registered_stage_is_applied_by_the_chain(registered, speech):
    anonymized = anonymize(speech, draw_params(0, stages=("invert",)))

    # The Chain restores the peak level of the Original.
    np.testing.assert_allclose(anonymized.samples, -speech.samples, atol=1e-6)


def test_registered_stage_is_available_from_the_cli(registered, speech, tmp_path):
    original_path = tmp_path / "a.wav"
    save_wav(original_path, speech)

    exit_code = main([
        "anonymize", str(original_path), "--seed", "1", "--stages", "invert", "--set", "invert_gain=0.6",
        "--no-privacy", "--no-content",
    ])

    assert exit_code == 0
    params = json.loads((tmp_path / "a_anon.json").read_text())["params"]
    assert params["stages"] == ["invert"]
    assert params["invert_gain"] == 0.6


def test_registered_stage_is_available_from_the_gui(registered, app):
    from anodio.gui.params_panel import ParamsPanel

    panel = ParamsPanel()

    assert panel.stage_boxes["invert"].text() == "Inversion de polarité"
    assert list(panel.sliders) == ["mcadams_alpha", "pitch_semitones", "formant_ratio", "vtln_alpha", "invert_gain"]
    assert "invert" in panel.params().stages
    panel.stage_boxes["invert"].setChecked(False)
    assert "invert" not in panel.params().stages
    assert not panel.sliders["invert_gain"].isEnabled()


def test_unknown_stage_in_params_is_an_error(speech):
    params = type(draw_params(0))(seed=0, values=draw_params(0).values, stages=("nope",))

    with pytest.raises(ValueError, match="nope"):
        anonymize(speech, params)
