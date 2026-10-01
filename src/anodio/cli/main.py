"""Command-line interface: `anodio anonymize <fichier.wav> --seed N`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anodio.core import (
    STAGES,
    AnodioError,
    Params,
    anonymize,
    draw_params,
    evaluate,
    load_wav,
    random_seed,
    save_wav,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anodio", description="Anonymisation de la voix.")
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("anonymize", help="Anonymiser un fichier .wav.")
    one.add_argument("input", type=Path, help="Fichier .wav original.")
    one.add_argument("--seed", type=int, default=None, help="Seed (tirée au hasard si absente).")
    one.add_argument("--out-dir", type=Path, default=None, help="Dossier de sortie (par défaut : celui de l'original).")
    _add_params_arguments(one)
    return parser


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


def _anonymize_file(input_path: Path, params: Params, out_dir: Path | None) -> None:
    original = load_wav(input_path)
    anonymized = anonymize(original, params)
    report = evaluate(original, anonymized, params=params)

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
            _anonymize_file(args.input, _params_from_args(args, args.seed), args.out_dir)
    except AnodioError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
