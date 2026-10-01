# anodio

anodio rend un enregistrement de parole non réidentifiable par un système de vérification du locuteur, tout en préservant l'information utile à la recherche clinique, à l'ASR et à l'analyse prosodique. Chaque anonymisation est accompagnée d'une évaluation chiffrée qui montre ce qui a changé et ce qui a été conservé.

## Enregistrements et locuteurs

**Speaker** (locuteur) :
La personne dont la voix figure dans un enregistrement. Un enregistrement contient un seul Speaker.
_Avoid_ : sujet, patient, utilisateur

**Original** :
L'enregistrement fourni en entrée, avant toute transformation.
_Avoid_ : source, input, brut

**Anonymized** (version anonymisée) :
L'enregistrement produit par une Anonymization à partir d'un Original.
_Avoid_ : déformé, distordu, converti

**Pseudo-voice** (pseudo-voix) :
La voix perçue dans une version Anonymized. Une même Seed donne la même Pseudo-voice.
_Avoid_ : voix cible, fausse voix

## Anonymisation

**Anonymization** (anonymisation) :
La transformation d'un Original en Anonymized, qui retire l'identité du Speaker et conserve les Invariants.
_Avoid_ : déformation, distorsion, voice conversion

**Anonymizer** :
Une méthode d'Anonymization interchangeable. Une Chain est un Anonymizer.
_Avoid_ : algorithme, filtre

**Chain** (chaîne) :
Un Anonymizer composé de plusieurs Stages appliqués successivement.
_Avoid_ : pipeline

**Stage** (étape) :
Une transformation élémentaire d'une Chain, réglée par ses propres Params.

**Params** :
L'ensemble des valeurs qui déterminent entièrement une Anonymization. Les Params sont dérivés d'une Seed, sauf ceux fixés par un Override.

**Seed** (graine) :
La valeur à partir de laquelle les Params sont tirés dans leurs Safe ranges. Une même Seed donne les mêmes Params et donc la même Pseudo-voice.

**Safe range** (plage sûre) :
L'intervalle dans lequel un Param est tiré. Il exclut les valeurs trop proches de l'identité, qui n'anonymisent pas, et les valeurs trop extrêmes, qui détruisent les Invariants.

**Override** :
Une valeur de Param imposée par l'utilisateur à la place de celle tirée de la Seed.

## Ce qui doit survivre

**Invariant** :
Une propriété de l'Original qui doit se retrouver dans la version Anonymized : timing, contenu verbal, forme de la prosodie, Clinical features.
_Avoid_ : information portée, signal utile

**Clinical feature** (feature clinique) :
Une mesure de qualité vocale ou de débit exprimée sous forme relative ou normalisée, par exemple le jitter en %, la variabilité du F0 en demi-tons ou la durée des pauses. Une mesure absolue comme le F0 moyen en Hz n'est jamais une Clinical feature, car l'Anonymization la modifie volontairement.
_Avoid_ : biomarqueur (pour la mesure elle-même)

**Prosody** (prosodie) :
La forme du contour d'intonation normalisé, le contour d'énergie et le rythme. La hauteur absolue de la voix n'en fait pas partie.

## Évaluation

**Report** (rapport) :
Le résultat de l'évaluation d'une paire Original/Anonymized : les Params, la Seed, les mesures de chaque Axis et leurs Verdicts.

**Axis** (axe) :
Une dimension de l'évaluation. Il y a quatre Axes jugés (Privacy, Content, Prosody, Clinical) et un axe Informational, qui n'est jamais jugé.

**Privacy** :
L'Axis qui mesure à quel point un ASV attacker échoue à relier la version Anonymized au Speaker.
_Avoid_ : sécurité, confidentialité

**Content** :
L'Axis qui mesure la préservation du contenu verbal, c'est-à-dire ce qui est dit.
_Avoid_ : intelligibilité (qui n'en est qu'une approximation)

**Informational** :
Les distances entre distributions de features (KL, JS, Wasserstein). Elles sont affichées pour indiquer à quel point le signal a changé, mais elles ne prouvent ni la non-réidentifiabilité ni la préservation des Invariants.

**Threshold** (seuil) :
La valeur, modifiable par l'utilisateur, qui sépare la réussite de l'échec sur un Axis.

**Verdict** :
La réussite ou l'échec d'un Axis, obtenu en comparant ses mesures à son Threshold.

## Menace

**ASV attacker** :
L'adversaire visé : un système automatique de vérification du locuteur, pré-entraîné sur des voix non anonymisées. Selon la condition, il ignore l'Anonymization (Ignorant attack) ou connaît l'Anonymizer sans connaître la Seed (Lazy-informed attack). Un auditeur humain et un attaquant entraîné sur des voix anonymisées sont hors périmètre.
_Avoid_ : pirate, adversaire (sans préciser)

**Baseline** (condition de référence) :
Une condition de Corpus evaluation où l'ASV attacker compare des Originaux entre eux. Elle mesure ce que l'ASV attacker sait faire sans Anonymization ; les autres conditions se lisent par rapport à elle.
_Avoid_ : condition originale

**Ignorant attack** :
Une condition de Corpus evaluation où l'ASV attacker enrôle des Originaux et les compare à des versions Anonymized d'*autres* enregistrements : il ignore l'Anonymization.

**Lazy-informed attack** :
Une condition de Corpus evaluation où l'ASV attacker connaît l'Anonymizer mais pas la Seed : il anonymise lui-même ses enregistrements d'enrôlement avec sa propre Seed, puis les compare aux versions Anonymized d'*autres* enregistrements.

**Linkability** (reliabilité) :
La possibilité de déterminer que deux versions Anonymized proviennent du même Speaker. Une Seed partagée rend la Linkability volontaire.

**Corpus evaluation** :
Une évaluation de Privacy sur un ensemble d'enregistrements de plusieurs Speakers, qui produit un taux d'erreur global. Elle s'oppose à l'évaluation d'une seule paire.
