"""Fetch a small French multi-speaker corpus for `anodio evaluate-corpus`.

Takes a few utterances per Speaker from the test split of Multilingual LibriSpeech
(French, CC BY 4.0, read LibriVox audiobooks) through the Hugging Face dataset viewer,
and lays them out as corpus/<speaker>/*.wav.

    uv run python scripts/fetch_mls_french.py corpus/mls-french
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import soundfile

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = {"dataset": "facebook/multilingual_librispeech", "config": "french", "split": "test"}
PAGE = 100


def rows(offset: int) -> tuple[list[dict], int]:
    query = urllib.parse.urlencode({**DATASET, "offset": offset, "length": PAGE})
    with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as response:
        page = json.load(response)
    return [entry["row"] for entry in page["rows"]], page["num_rows_total"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--speakers", type=int, default=10)
    parser.add_argument("--utterances", type=int, default=6, help="per Speaker")
    args = parser.parse_args()

    kept: dict[str, int] = {}
    transcripts: dict[str, str] = {}
    offset, total = 0, 1
    while offset < total:
        page, total = rows(offset)
        offset += PAGE
        for row in page:
            speaker = row["speaker_id"]
            if speaker not in kept and len(kept) == args.speakers:
                continue
            if kept.get(speaker, 0) == args.utterances:
                continue
            with urllib.request.urlopen(row["audio"][0]["src"], timeout=60) as response:
                samples, sample_rate = soundfile.read(io.BytesIO(response.read()))
            folder = args.corpus / speaker
            folder.mkdir(parents=True, exist_ok=True)
            soundfile.write(folder / f"{row['id']}.wav", samples, sample_rate, subtype="PCM_16")
            kept[speaker] = kept.get(speaker, 0) + 1
            transcripts[row["id"]] = row["transcript"]
            print(f"{speaker}/{row['id']}.wav")
        if len(kept) == args.speakers and all(count == args.utterances for count in kept.values()):
            break
    (args.corpus / "transcripts.json").write_text(json.dumps(transcripts, ensure_ascii=False, indent=2))
    print(f"{sum(kept.values())} enregistrements, {len(kept)} Speakers dans {args.corpus}")


if __name__ == "__main__":
    main()
