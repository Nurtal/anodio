# Journal de travail

Ce journal suit les corrections faites après la revue de code (volets Standards et Spec) de l'implémentation v1 (tickets #2 à #11), dans l'ordre où elles ont été réalisées.

## 2026-10-01 — Corrections issues de la revue

Points retenus pour une correction immédiate :

1. Le Verdict global peut être « réussi » alors que la Privacy n'a pas été évaluée.
2. Le traitement par lot s'interrompt sur toute erreur qui n'est pas une `AnodioError`.
3. L'ADR 0001 n'est plus à jour sur l'ordre de la Chain et sur le PSOLA.
4. Le glossaire contient deux défauts : la définition de l'ASV attacker contredit la Lazy-informed attack, et la condition « original » de la Corpus evaluation n'a pas d'entrée.
5. Le README emploie un vocabulaire que le glossaire demande d'éviter.
6. Le seuil de WER est appliqué en ≤ 5 %, alors que la spec demande < 5 %.

Points reportés en tickets : analyse des features à 16 kHz, affichage de la forme d'onde, nettoyage des code smells.

Points soumis à ta décision : voir la section « Questions ouvertes » en bas.

### 1. Verdict global sans Privacy — corrigé

- **Problème.** Si le modèle ECAPA ou Whisper manquait, l'Axis correspondant disparaissait du Report, et le Verdict global pouvait afficher « réussi » sans que la Privacy ait été vérifiée.
- **Correction.** `Report` définit désormais quatre Axes obligatoires (`REQUIRED_AXES` : privacy, content, prosody, clinical). Tout Axis absent rend le Verdict global **False**. Il apparaît dans `Report.failures` (« Axe non évalué (modèle absent ou désactivé) ») et dans le JSON sous la clé `missing_axes`. La GUI l'affiche comme « ⚠ non évalué ».
- **Conséquence.** `--no-privacy` et `--no-content` donnent maintenant toujours un Verdict global en échec. C'est voulu : un résultat dont la Privacy n'a pas été vérifiée n'est pas « utilisable ».
- **Tests.** Deux nouveaux tests dans `tests/test_verdicts.py`, écrits avant le code (red → green).

### 2. Traitement par lot interrompu par une erreur imprévue — corrigé

- **Problème.** Seules les `AnodioError` étaient isolées fichier par fichier. En reproduisant le cas, j'ai trouvé qu'un fichier trop court (400 échantillons) faisait planter l'analyse de pitch de Praat avec une `PraatError`, ce qui arrêtait tout le lot.
- **Corrections.**
  - `load_wav` refuse les enregistrements de moins de 0,5 s avec un message clair (« … est trop court (0.03 s) : il faut au moins 0.5 s de signal »). En dessous, Praat ne peut pas mesurer le F0.
  - `anodio batch` attrape désormais toute exception par fichier, l'inscrit dans `summary.json` (« Erreur inattendue : … ») et passe au fichier suivant.
- **Tests.** Trois nouveaux tests : un fichier trop court est refusé, un fichier trop court est signalé sans interrompre le lot, une exception arbitraire sur un fichier n'interrompt pas le lot. Deux anciens tests d'I/O utilisaient des signaux de 0,05 et 0,1 s ; je les ai allongés à 1 s sans changer ce qu'ils vérifient.

### 3 à 6. Seuil de WER, ADR, glossaire, README — corrigés

- **Seuil de WER strict (point 6).** Le seuil passe de `wer > 0,05` à `wer >= 0,05`, conformément à la spec (« inférieure à 5 points »). Un nouveau test vérifie qu'un WER d'exactement 5 % (1 erreur sur 20 mots) échoue. Le README indique désormais « < 5 % ».
- **ADR 0001 (point 3).** L'ordre de la Chain est corrigé (PSOLA, puis McAdams, puis VTLN). J'ai ajouté une section « Mise en œuvre » qui explique pourquoi le PSOLA passe en premier et pourquoi c'est une implémentation maison plutôt que le « Change gender » de Praat, chiffres à l'appui.
- **Glossaire (point 4).**
  - La définition de l'ASV attacker ne dit plus qu'il « ne connaît pas l'Anonymizer » : elle renvoie aux conditions Ignorant et Lazy-informed.
  - Nouvelle entrée **Baseline** (condition de référence). La clé JSON correspondante de `evaluate-corpus` passe de `original` à `baseline`. ⚠️ Ce renommage casse le format de sortie, sans conséquence puisque rien n'est encore publié.
- **README (point 5).** Le vocabulaire suit maintenant le glossaire : Params, Anonymization, Pseudo-voice, Safe range, Stage, Chain. Le terme « déformation » a disparu.
- **Au passage.** `requires-python` passe de `>=3.11` à `>=3.12`, en cohérence avec le README et mypy.

### Tickets créés pour la suite

Les trois tickets portent le label `ready-for-agent` et sont rattachés comme sous-issues à la spec #1 :

- [#12](https://github.com/Nurtal/anodio/issues/12) : analyser les features sur une copie à 16 kHz. Aujourd'hui, les mesures dépendent de la fréquence d'échantillonnage de l'Original.
- [#13](https://github.com/Nurtal/anodio/issues/13) : afficher la forme d'onde de l'Original dès le chargement (user story 5).
- [#14](https://github.com/Nurtal/anodio/issues/14) : nettoyage des code smells : registre de Stages, seuils typés, résultat de Corpus evaluation typé, dédoublonnage, nommage.

### État à la fin de la session

- Les 207 tests passent, y compris les tests lents avec les vrais modèles ECAPA et Whisper. mypy est propre.
- Les commits sont sur `main` en local. **Rien n'est poussé** : `origin/main` ne contient que le commit de documentation du début.

## Questions ouvertes (tes décisions)

1. **WER : un ou deux ?** La user story 25 demande « le WER de l'Original, celui de l'Anonymized et leur écart », alors que la spec dit aussi « la transcription de l'Original sert de référence, donc pas de vérité terrain ». Sans vérité terrain, le WER de l'Original est par définition 0. Aujourd'hui, on calcule un seul WER (Anonymized contre Original). Pour avoir les deux, il faudrait pouvoir fournir une transcription de référence, par exemple un `.txt` à côté du `.wav`. Est-ce utile ?
2. **Les pauses doivent-elles être jugées ?** La spec disait « les statistiques de pauses sont aussi rapportées ». Le code fait aussi échouer la Prosody si les pauses bougent de plus de 20 ms ou si leur nombre change. C'est plus strict que la spec, mais cohérent avec l'Invariant « timing ». On garde ?
3. **Bas de la Safe range de McAdams.** Avec α ≈ 0,5, j'ai observé une fois un jitter à +77 % et un HNR à −8 dB. Faut-il remonter le plancher (à 0,65 par exemple) au prix d'un peu de Privacy ?
4. **Validation sur de vrais enregistrements.** Tout a été validé sur des voix synthétiques. Sur ces voix, ECAPA reconnaît encore les Speakers après anonymisation (cosinus de 0,5 à 0,8, EER ignorant de 0 %), ce qui ne dit rien de la Privacy réelle. As-tu un petit corpus français de plusieurs locuteurs pour lancer `anodio evaluate-corpus` ?
5. **Push.** Veux-tu que je pousse `main` sur GitHub ?
