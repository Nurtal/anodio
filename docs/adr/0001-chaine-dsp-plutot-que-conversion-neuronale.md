# Chaîne DSP plutôt que conversion de voix neuronale

L'Anonymizer de la v1 est une Chain DSP (McAdams, puis décalage du F0 et des formants par PSOLA, puis VTLN). Nous n'utilisons pas de conversion de voix neuronale (kNN-VC, etc.), alors que ces méthodes anonymisent nettement mieux face à un ASV attacker. La raison : la re-synthèse neuronale détruit les Clinical features (jitter, shimmer, HNR), qui doivent survivre pour l'usage en recherche clinique. Les transformations DSP conservent la structure source-filtre, et donc ces mesures relatives.

## Consequences

- La Privacy attendue est modérée : un EER d'environ 20 à 40 % face à l'ASV attacker, et non 50 %. L'outil doit l'afficher honnêtement plutôt que la masquer.
- Les Anonymizers neuronaux restent possibles derrière l'interface Anonymizer, mais uniquement pour les usages où les Clinical features n'importent pas. Ils ne doivent pas remplacer la Chain par défaut.
