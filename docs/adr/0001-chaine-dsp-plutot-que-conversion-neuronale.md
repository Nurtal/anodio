# Chaîne DSP plutôt que conversion de voix neuronale

L'Anonymizer de la v1 est une Chain DSP (décalage du F0 et des formants par PSOLA, puis McAdams, puis VTLN). Nous n'utilisons pas de conversion de voix neuronale (kNN-VC, etc.), alors que ces méthodes anonymisent nettement mieux face à un ASV attacker. La raison : la re-synthèse neuronale détruit les Clinical features (jitter, shimmer, HNR), qui doivent survivre pour l'usage en recherche clinique. Les transformations DSP conservent la structure source-filtre, et donc ces mesures relatives.

## Consequences

- La Privacy attendue est modérée : un EER d'environ 20 à 40 % face à l'ASV attacker, et non 50 %. L'outil doit l'afficher honnêtement plutôt que la masquer.
- Les Anonymizers neuronaux restent possibles derrière l'interface Anonymizer, mais uniquement pour les usages où les Clinical features n'importent pas. Ils ne doivent pas remplacer la Chain par défaut.

## Mise en œuvre (ajouté le 2026-10-01)

- **Le PSOLA passe en premier.** Il détecte les pulsations glottiques, ce qui n'est fiable que sur le signal original : placé après McAdams, il faisait chuter la corrélation de F0 à 0,73 et monter f0_cv de 26 %.
- **Le PSOLA est une implémentation maison, pas le « Change gender » de Praat.** Praat resynthétise les périodes à partir d'une courbe de pitch lissée, ce qui divisait le jitter par deux (1,92 % → 0,85 %). Notre PSOLA reporte, période par période, des déviations de période et d'amplitude tirées parmi celles de l'original : le jitter est conservé (1,92 % → 1,9 %).
