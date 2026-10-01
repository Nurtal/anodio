"""Command-line interface: `anodio anonymize <fichier.wav> --seed N`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anodio.core import AnodioError, anonymize, draw_params, evaluate, load_wav, random_seed, save_wav


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anodio", description="Anonymisation de la voix.")
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("anonymize", help="Anonymiser un fichier .wav.")
    one.add_argument("input", type=Path, help="Fichier .wav original.")
    one.add_argument("--seed", type=int, default=None, help="Seed (tirée au hasard si absente).")
    one.add_argument("--out-dir", type=Path, default=None, help="Dossier de sortie (par défaut : celui de l'original).")
    return parser


def _anonymize_file(input_path: Path, seed: int | None, out_dir: Path | None) -> None:
    original = load_wav(input_path)
    params = draw_params(random_seed() if seed is None else seed)
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
            _anonymize_file(args.input, args.seed, args.out_dir)
    except AnodioError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
