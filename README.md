# anodio

anodio rend un enregistrement de parole (.wav, un seul locuteur) **non réidentifiable par un système automatique de vérification du locuteur**, tout en préservant ce qui sert à la recherche : le timing, le contenu verbal, la forme de l'intonation et les features cliniques de la voix (jitter, shimmer, variabilité du F0…).

Chaque anonymisation produit un **Report** qui mesure, axe par axe, ce qui a changé et ce qui a été conservé, et rend un verdict à partir de seuils réglables.

Tout fonctionne **en local, hors ligne, sur CPU**. Le vocabulaire du projet est défini dans [`CONTEXT.md`](CONTEXT.md).

## Installation

Il faut [uv](https://docs.astral.sh/uv/) et Python 3.12.

```bash
uv sync
uv run anodio download-models          # une seule fois, connexion requise (~1 Go)
```

`download-models` récupère ECAPA-TDNN (vérification du locuteur, pour l'axe Privacy) et Whisper `small` (transcription, pour l'axe Contenu) dans `~/.cache/anodio/models`. Pour changer cet emplacement, définissez `ANODIO_MODELS_DIR`. Pour une autre taille de Whisper, ajoutez `--whisper-size tiny|base|small|medium`. Ensuite, plus aucun accès réseau n'a lieu.

## Interface graphique

```bash
uv run anodio-gui
```

1. **Charger un .wav.** Les fichiers stéréo sont convertis en mono, et toutes les fréquences d'échantillonnage sont acceptées. La durée, la fréquence d'échantillonnage et la forme d'onde de l'original s'affichent aussitôt, pour vérifier que c'est le bon fichier ; celle de la version anonymisée s'ajoute en dessous après l'anonymisation. L'Anonymization et l'export gardent la fréquence d'origine ; les mesures du Report sont prises sur une copie à 16 kHz, pour que les Reports restent comparables d'un fichier à l'autre.
2. **Choisir une Seed.** La Seed détermine les Params de l'Anonymization, donc la Pseudo-voice. Chaque curseur est verrouillé sur la Seed. En décochant « Seed », on impose une valeur manuelle (un Override), et un avertissement s'affiche si elle sort de sa Safe range. Chaque Stage de la Chain peut être désactivé.
3. **Anonymiser.** Le calcul tourne en arrière-plan. Compter environ 2 min 30 pour 5 min d'audio.
4. **Lire le Report.** On y trouve le verdict global et la raison de chaque échec, les transcriptions côte à côte et le tableau des features cliniques. Les onglets montrent les spectrogrammes, les contours d'intonation superposés et les distributions de features. Le lecteur A/B bascule entre l'original et la version anonymisée à la même position.
5. **Ajuster les seuils.** Les verdicts sont recalculés immédiatement, sans relancer l'anonymisation.
6. **Exporter** le `.wav` anonymisé et son Report `.json`.

## Ligne de commande

```bash
# Un fichier
uv run anodio anonymize entretien.wav --seed 42
uv run anodio anonymize entretien.wav --seed 42 --set pitch_semitones=3 --stages psola,mcadams \
    --threshold privacy.max_cosine_similarity=0.3 --whisper-size medium

# Un corpus : corpus/<speaker>/*.wav, une Seed par Speaker
uv run anodio batch corpus/ --out-dir anonymise/ --seed-map seeds.csv   # CSV « speaker,seed »

# Corpus evaluation : EER de l'ASV attacker et agrégats d'utilité
uv run anodio evaluate-corpus corpus/ --seed-map seeds.csv --output evaluation.json
```

`--no-privacy` et `--no-content` désactivent les axes qui nécessitent un modèle. `uv run anodio <commande> --help` liste toutes les options.

## Ce que fait la Chain

| Stage | Effet | Param (Safe range) |
|---|---|---|
| **PSOLA** | Décale le F0 et les formants. Le jitter et le shimmer de l'original sont reportés période par période. | `pitch_semitones` ∈ [−5 ; −2] ∪ [2 ; 5], `formant_ratio` ∈ [0,85 ; 0,95] ∪ [1,05 ; 1,15] |
| **McAdams** | Déplace l'enveloppe spectrale (pôles LPC élevés à la puissance α). | `mcadams_alpha` ∈ [0,65 ; 0,9] |
| **VTLN** | Applique un warping bilinéaire de l'enveloppe spectrale. | `vtln_alpha` ∈ [−0,15 ; −0,05] ∪ [0,05 ; 0,15] |

Les Safe ranges excluent volontairement les valeurs proches de l'identité, qui n'anonymisent pas. L'excitation glottique (le résidu LPC) n'est jamais resynthétisée : c'est ce qui préserve les features cliniques (voir [l'ADR 0001](docs/adr/0001-chaine-dsp-plutot-que-conversion-neuronale.md)).

## Les axes du Report

| Axe | Mesure | Seuil par défaut |
|---|---|---|
| **Privacy** | Similarité cosinus entre les embeddings ECAPA de l'original et de la version anonymisée | < 0,25 |
| **Contenu** | WER de la transcription Whisper de la version anonymisée, en prenant celle de l'original comme référence | < 5 % |
| **Prosodie** | Corrélation des contours de F0 normalisés (demi-tons / médiane), durée ; position des pauses et DTW de l'énergie rapportées sans être jugées | corrélation ≥ 0,9 ; durée identique |
| **Clinique** | CV et écart-type (en demi-tons) du F0, jitter, shimmer, HNR, débit, nombre et durée des pauses | écart relatif ≤ 10 % (shimmer : 25 %) ; HNR : écart ≤ 3 dB |
| **Informatif** | KL, Jensen-Shannon et Wasserstein entre les distributions de MFCC, de F0 et d'énergie | jamais jugé |

Le **verdict global** est réussi quand tous les axes jugés réussissent. Les distances de l'axe Informatif quantifient l'ampleur du changement, mais **elles ne prouvent ni la non-réidentifiabilité ni la préservation de l'information**.

## Limites connues

- **La Privacy n'est évaluée que face à un ASV automatique (ECAPA).** Ni un auditeur humain, en particulier un proche, ni un attaquant entraîné sur des voix anonymisées ne sont pris en compte. La Corpus evaluation ajoute les conditions *ignorant* et *lazy-informed*, où l'attaquant connaît l'Anonymizer mais pas la Seed.
- **La Privacy attendue est modérée.** Pour préserver les features cliniques, anodio utilise du traitement du signal plutôt qu'une conversion de voix neuronale (ADR 0001). Sur 10 locuteurs français (voir ci-dessous), l'EER de l'ASV passe de 0 % sans anonymisation à 26 % (*ignorant*) et 36 % (*lazy-informed*), loin des 50 % d'un anonymat parfait. Mesurez-le sur vos données avec `evaluate-corpus`.
- **Une Seed partagée crée de la Linkability.** Deux enregistrements anonymisés avec la même Seed portent la même pseudo-voix : on peut les relier entre eux, ce qui est voulu pour le suivi d'un même locuteur. Ne réutilisez pas une Seed entre locuteurs ou entre études si ce lien doit rester impossible. Stockez les Seeds comme une donnée sensible.
- **Le seuil cosinus de 0,25 est indicatif.** Il dépend du modèle et de la langue. Recalibrez-le avec `evaluate-corpus` sur des données françaises.
- **Le HNR dépend en partie du conduit vocal**, que la Chain déplace volontairement. L'axe Clinique le signale toujours.
- **Sur de vraies voix, l'utilité n'est pas encore au rendez-vous.** Les tests automatiques reposent sur des voix synthétiques, où les features cliniques sont préservées. Sur 60 enregistrements de 10 locuteurs français (Multilingual LibriSpeech, livres audio lus, compression avec pertes), avec les Thresholds par défaut : 5 % des enregistrements réussissent l'axe Clinique (écart médian de 23 % sur le jitter, 41 % sur le shimmer, 5,7 dB sur le HNR), 57 % l'axe Prosodie, 10 % l'axe Contenu (WER moyen de 29 %). Vérifiez le Report de vos propres fichiers avant tout usage clinique.

Pour reproduire ces chiffres :

```bash
uv run python scripts/fetch_mls_french.py corpus/mls-french      # 10 locuteurs × 6 énoncés, CC BY 4.0
uv run anodio evaluate-corpus corpus/mls-french --seed-map seeds.csv --attacker-seed 7
```

## Développement

```bash
uv run pytest                 # suite rapide (modèles remplacés par des fakes)
uv run pytest -m slow         # vrais modèles ECAPA et Whisper (après download-models)
uv run mypy src
```

Le cœur (`anodio.core`) n'a aucune dépendance à Qt. Son API publique est `draw_params`, `anonymize`, `evaluate`, `apply_thresholds` et `evaluate_corpus`. L'interface graphique et la ligne de commande sont de fines couches au-dessus.

Pour ajouter un Stage à la Chain, il suffit d'une entrée dans `REGISTRY` (`src/anodio/core/stages.py`) : les Params, la ligne de commande et l'interface graphique en découlent.
