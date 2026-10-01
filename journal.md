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

