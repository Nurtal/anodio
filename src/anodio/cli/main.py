"""Command-line interface.

    anodio anonymize <fichier.wav> [--seed N] ...
    anodio batch <dossier> --out-dir <sortie> [--seed N | --seed-map seeds.csv] ...
    anodio evaluate-corpus <dossier> [--seed N | --seed-map seeds.csv] [--output eval.json] ...
    anodio download-models
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from anodio.core import (
    DEFAULT_THRESHOLDS,
    STAGES,
    AnodioError,
    Params,
    Report,
    SpeakerEmbedder,
    Transcriber,
    anonymize,
    draw_params,
    evaluate_corpus,
    evaluate,
    load_wav,
    random_seed,
    save_wav,
)
from anodio.core.models import (
    DEFAULT_WHISPER_SIZE,
    WHISPER_SIZES,
    EcapaEmbedder,
    WhisperTranscriber,
    download_models,
    models_dir,
)

# Model factories; replaced by fakes in the tests.
load_embedder = EcapaEmbedder
load_transcriber = WhisperTranscriber

VERDICT_TEXT = {True: "réussi", False: "échec", None: "non jugé"}


@dataclass
class Processing:
    """Everything but the Seed that determines how a file is processed."""

    overrides: dict[str, float]
    stages: tuple[str, ...]
    thresholds: dict[str, dict[str, float]]
    embedder: SpeakerEmbedder | None
    transcriber: Transcriber | None

    def params(self, seed: int) -> Params:
        try:
            return draw_params(seed, overrides=self.overrides, stages=self.stages)
        except ValueError as exc:
            raise AnodioError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anodio", description="Anonymisation de la voix.")
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("anonymize", help="Anonymiser un fichier .wav.")
    one.add_argument("input", type=Path, help="Fichier .wav original.")
    one.add_argument("--seed", type=int, default=None, help="Seed (tirée au hasard si absente).")
    one.add_argument("--out-dir", type=Path, default=None, help="Dossier de sortie (par défaut : celui de l'original).")
    _add_processing_arguments(one)

    batch = sub.add_parser(
        "batch",
        help="Anonymiser tous les .wav d'un dossier. Le Speaker d'un fichier est le nom de son sous-dossier.",
    )
    batch.add_argument("input", type=Path, help="Dossier du corpus (corpus/<speaker>/*.wav).")
    batch.add_argument("--out-dir", type=Path, required=True, help="Dossier de sortie (même arborescence).")
    batch.add_argument("--seed", type=int, default=None, help="Seed commune (ou de repli avec --seed-map).")
    batch.add_argument("--seed-map", type=Path, default=None, help="CSV « speaker,seed » : une Seed par Speaker.")
    _add_processing_arguments(batch)

    corpus = sub.add_parser(
        "evaluate-corpus",
        help="Corpus evaluation : EER de l'ASV attacker (original, ignorant, lazy-informed) et agrégats d'utilité.",
    )
    corpus.add_argument("input", type=Path, help="Dossier du corpus (corpus/<speaker>/*.wav), au moins 2 Speakers.")
    corpus.add_argument("--seed", type=int, default=None, help="Seed commune (ou de repli avec --seed-map).")
    corpus.add_argument("--seed-map", type=Path, default=None, help="CSV « speaker,seed » : une Seed par Speaker.")
    corpus.add_argument(
        "--attacker-seed", type=int, default=None, help="Seed de l'attaquant lazy-informed (tirée au hasard si absente)."
    )
    corpus.add_argument("--output", type=Path, default=None, help="Fichier JSON de résultats (sinon : sortie standard).")
    _add_processing_arguments(corpus)

    download = sub.add_parser(
        "download-models", help="Télécharger les modèles une fois pour toutes (connexion requise)."
    )
    _add_whisper_argument(download)
    return parser


def _add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-privacy",
        action="store_true",
        help="Ne pas évaluer la Privacy (le modèle ECAPA n'est alors pas nécessaire).",
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Ne pas évaluer le contenu (Whisper n'est alors pas nécessaire).",
    )
    _add_whisper_argument(parser)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PARAM=VALEUR",
        help="Override d'un Param (répétable), ex. --set pitch_semitones=3.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"Stages actifs, séparés par des virgules (défaut : {','.join(STAGES)}).",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="AXE.CLÉ=VALEUR",
        help="Threshold (répétable), ex. --threshold privacy.max_cosine_similarity=0.3.",
    )


def _add_whisper_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--whisper-size",
        choices=WHISPER_SIZES,
        default=DEFAULT_WHISPER_SIZE,
        help=f"Taille du modèle Whisper (défaut : {DEFAULT_WHISPER_SIZE}).",
    )


def _processing_from_args(args: argparse.Namespace) -> Processing:
    overrides = {}
    for item in args.set:
        name, sep, value = item.partition("=")
        if not sep:
            raise AnodioError(f"Override invalide « {item} » : attendu PARAM=VALEUR.")
        overrides[name.strip()] = _number(name, value)
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    processing = Processing(overrides, stages, _thresholds_from_args(args), None, None)
    for name in processing.params(0).outside_safe_ranges():
        print(f"Attention : {name} = {overrides[name]} est hors de sa Safe range.", file=sys.stderr)
    processing.embedder = None if args.no_privacy else load_embedder()
    processing.transcriber = None if args.no_content else load_transcriber(args.whisper_size)
    return processing


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for item in args.threshold:
        name, sep, value = item.partition("=")
        axis, dot, key = name.strip().partition(".")
        if not sep or not dot:
            raise AnodioError(f"Threshold invalide « {item} » : attendu AXE.CLÉ=VALEUR.")
        if axis not in DEFAULT_THRESHOLDS:
            raise AnodioError(f"Axe inconnu « {axis} » (choix : {', '.join(DEFAULT_THRESHOLDS)}).")
        thresholds.setdefault(axis, {})[key] = _number(name, value)
    return thresholds


def _number(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise AnodioError(f"Valeur non numérique pour {name} : {value}") from exc


def _process_file(input_path: Path, out_dir: Path, params: Params, processing: Processing) -> Report:
    original = load_wav(input_path)
    anonymized = anonymize(original, params)
    report = evaluate(
        original,
        anonymized,
        params=params,
        thresholds=processing.thresholds,
        embedder=processing.embedder,
        transcriber=processing.transcriber,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{input_path.stem}_anon"
    save_wav(out_dir / f"{stem}.wav", anonymized)
    (out_dir / f"{stem}.json").write_text(report.to_json(), encoding="utf-8")
    return report


def _anonymize_one(args: argparse.Namespace) -> int:
    processing = _processing_from_args(args)
    params = processing.params(random_seed() if args.seed is None else args.seed)
    out_dir = args.out_dir or args.input.parent
    report = _process_file(args.input, out_dir, params, processing)
    print(f"{out_dir / args.input.stem}_anon.wav (seed {params.seed}) — verdict global : {VERDICT_TEXT[report.verdict]}")
    for axis, failure in report.failures:
        print(f"  [{axis}] {failure}")
    return 0


def _read_seed_map(path: Path) -> dict[str, int]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return {row["speaker"].strip(): int(row["seed"]) for row in csv.DictReader(handle)}
    except (OSError, KeyError, ValueError) as exc:
        raise AnodioError(f"Fichier de Seeds illisible ({path}) : attendu un CSV « speaker,seed ». {exc}") from exc


def _speaker(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else path.stem


def _anonymize_batch(args: argparse.Namespace) -> int:
    if not args.input.is_dir():
        raise AnodioError(f"Dossier introuvable : {args.input}")
    seed_map = _read_seed_map(args.seed_map) if args.seed_map else {}
    fallback_seed = args.seed if args.seed is not None else (None if seed_map else random_seed())
    processing = _processing_from_args(args)
    entries = []
    for path in sorted(args.input.rglob("*.wav")):
        speaker = _speaker(args.input, path)
        relative = path.relative_to(args.input)
        entry: dict = {"file": str(relative), "speaker": speaker}
        try:
            seed = seed_map.get(speaker, fallback_seed)
            if seed is None:
                raise AnodioError(f"Le Speaker « {speaker} » n'a pas de Seed dans {args.seed_map}.")
            entry["seed"] = seed
            report = _process_file(path, args.out_dir / relative.parent, processing.params(seed), processing)
            entry["verdict"] = report.verdict
            entry["failures"] = [f"[{axis}] {failure}" for axis, failure in report.failures]
            print(f"{relative} : {VERDICT_TEXT[report.verdict]}")
        except AnodioError as exc:
            entry["error"] = str(exc)
            print(f"{relative} : ERREUR — {exc}", file=sys.stderr)
        entries.append(entry)

    failed = sum("error" in entry for entry in entries)
    summary = {
        "processed": len(entries) - failed,
        "failed": failed,
        "passed": sum(entry.get("verdict") is True for entry in entries),
        "files": entries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{summary['processed']} fichier(s) traité(s), {failed} en erreur, {summary['passed']} utilisable(s).")
    return 2 if failed else 0


def _evaluate_corpus(args: argparse.Namespace) -> int:
    if not args.input.is_dir():
        raise AnodioError(f"Dossier introuvable : {args.input}")
    if args.no_privacy:
        raise AnodioError("La Corpus evaluation mesure la Privacy : --no-privacy n'a pas de sens ici.")
    seed_map = _read_seed_map(args.seed_map) if args.seed_map else {}
    fallback_seed = args.seed if args.seed is not None else (None if seed_map else random_seed())
    recordings = []
    for path in sorted(args.input.rglob("*.wav")):
        speaker = _speaker(args.input, path)
        if speaker not in seed_map and fallback_seed is None:
            raise AnodioError(f"Le Speaker « {speaker} » n'a pas de Seed dans {args.seed_map}.")
        recordings.append((speaker, load_wav(path)))
    processing = _processing_from_args(args)
    assert processing.embedder is not None
    attacker_seed = args.attacker_seed if args.attacker_seed is not None else random_seed()
    try:
        result = evaluate_corpus(
            recordings,
            params_for=lambda speaker: processing.params(seed_map.get(speaker, fallback_seed)),  # type: ignore[arg-type]
            attacker_seed=attacker_seed,
            embedder=processing.embedder,
            transcriber=processing.transcriber,
            thresholds=processing.thresholds,
            on_progress=lambda done, total: print(f"\r{done}/{total} enregistrements", end="", file=sys.stderr),
        )
    except ValueError as exc:
        raise AnodioError(str(exc)) from exc
    print(file=sys.stderr)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        for condition, values in result["privacy"].items():
            print(f"EER {condition} : {100 * values['eer']:.1f} %")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "anonymize":
            return _anonymize_one(args)
        if args.command == "batch":
            return _anonymize_batch(args)
        if args.command == "evaluate-corpus":
            return _evaluate_corpus(args)
        if args.command == "download-models":
            print(f"Téléchargement des modèles dans {models_dir()}…")
            for path in download_models(whisper_size=args.whisper_size):
                print(f"  {path}")
    except AnodioError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
