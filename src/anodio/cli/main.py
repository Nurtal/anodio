"""Command-line interface: `anodio anonymize <fichier.wav> --seed N`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anodio.core import (
    STAGES,
    AnodioError,
    Params,
    SpeakerEmbedder,
    Transcriber,
    anonymize,
    draw_params,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anodio", description="Anonymisation de la voix.")
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("anonymize", help="Anonymiser un fichier .wav.")
    one.add_argument("input", type=Path, help="Fichier .wav original.")
    one.add_argument("--seed", type=int, default=None, help="Seed (tirée au hasard si absente).")
    one.add_argument("--out-dir", type=Path, default=None, help="Dossier de sortie (par défaut : celui de l'original).")
    one.add_argument(
        "--no-privacy",
        action="store_true",
        help="Ne pas évaluer la Privacy (le modèle ECAPA n'est alors pas nécessaire).",
    )
    one.add_argument(
        "--no-content",
        action="store_true",
        help="Ne pas évaluer le contenu (Whisper n'est alors pas nécessaire).",
    )
    _add_whisper_argument(one)
    _add_params_arguments(one)

    download = sub.add_parser(
        "download-models", help="Télécharger les modèles une fois pour toutes (connexion requise)."
    )
    _add_whisper_argument(download)
    return parser


def _add_whisper_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--whisper-size",
        choices=WHISPER_SIZES,
        default=DEFAULT_WHISPER_SIZE,
        help=f"Taille du modèle Whisper (défaut : {DEFAULT_WHISPER_SIZE}).",
    )


def _add_params_arguments(parser: argparse.ArgumentParser) -> None:
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


def _params_from_args(args: argparse.Namespace, seed: int | None) -> Params:
    overrides = {}
    for item in args.set:
        name, sep, value = item.partition("=")
        if not sep:
            raise AnodioError(f"Override invalide « {item} » : attendu PARAM=VALEUR.")
        try:
            overrides[name.strip()] = float(value)
        except ValueError as exc:
            raise AnodioError(f"Valeur non numérique pour {name} : {value}") from exc
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    try:
        params = draw_params(random_seed() if seed is None else seed, overrides=overrides, stages=stages)
    except ValueError as exc:
        raise AnodioError(str(exc)) from exc
    for name in params.outside_safe_ranges():
        print(f"Attention : {name} = {getattr(params, name)} est hors de sa Safe range.", file=sys.stderr)
    return params


def _anonymize_file(
    input_path: Path,
    params: Params,
    out_dir: Path | None,
    embedder: SpeakerEmbedder | None,
    transcriber: Transcriber | None,
) -> None:
    original = load_wav(input_path)
    anonymized = anonymize(original, params)
    report = evaluate(original, anonymized, params=params, embedder=embedder, transcriber=transcriber)

    out_dir = out_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{input_path.stem}_anon"
    save_wav(out_dir / f"{stem}.wav", anonymized)
    (out_dir / f"{stem}.json").write_text(report.to_json(), encoding="utf-8")
    print(f"{out_dir / stem}.wav (seed {params.seed})")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "anonymize":
            params = _params_from_args(args, args.seed)
            embedder = None if args.no_privacy else load_embedder()
            transcriber = None if args.no_content else load_transcriber(args.whisper_size)
            _anonymize_file(args.input, params, args.out_dir, embedder, transcriber)
        elif args.command == "download-models":
            print(f"Téléchargement des modèles dans {models_dir()}…")
            for path in download_models(whisper_size=args.whisper_size):
                print(f"  {path}")
    except AnodioError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
