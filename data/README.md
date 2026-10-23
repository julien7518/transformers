# Corpus v2 — français standard ↔ français Internet / Gen Z

Le corpus contient **1 346 paires synthétiques**, dont **771 exemples rédigés individuellement** et **575 exemples compositionnels** issus de 36 patrons grammaticaux. Il couvre **31 thèmes** et **89 expressions ou références** inventoriées. Ces nombres décrivent le corpus, pas 1 346 situations indépendantes ni 89 mèmes distincts : l’inventaire inclut aussi de l’argot et du vocabulaire gaming.

Cette version remplace la v1 de 1 000 lignes, qui répétait 250 phrases sous quatre suffixes. **L’expansion par suffixes de remplissage a été supprimée.** Le contenu rédigé comprend maintenant humour, réactions, conversations, culture Internet, références francophones, gaming, relations, études, questions et négations.

## Références et sens

Exemples d’entrées couvertes : aura, PNJ, lore, side quest, boss final, rizz, delulu, red flag, ghoster, laissé en vu, remis, cringe, « laisse-le cook », lock in, hype, banger, rage quit, skill issue, POV, « this is fine », « c’est pas faux », « j’ai glissé chef », « les calculs sont pas bons », ratio, quoicoubeh, feur et skibidi.

`references.tsv` décrit les interprétations retenues et leurs limites. `reference_tags` est un repérage lexical destiné à l’audit ; il ne constitue pas une annotation sémantique parfaite. Un même mot peut être polysémique et un tag peut apparaître dans une négation.

Le registre est plus expressif que dans la v1, mais chaque phrase ne doit pas devenir un catalogue de références. Les paires conservent autant que possible action, sujet, objet, négation et intensité. Exemples originaux :

| Standard | Internet / Gen Z |
| --- | --- |
| Je ne connais pas cette référence et je préfère le dire. | J’ai pas la ref, je vais pas mytho. |
| Je ne suis pas amoureux, je la trouve simplement sympathique. | J’ai pas de crush, je la trouve juste sympa. |
| Le problème vient de ma connexion, pas de mon niveau. | C’est la co, pas un skill issue. |
| Je ne vais pas prétexter un accident, j’ai choisi de le faire. | Je vais pas sortir « j’ai glissé chef », j’ai choisi de le faire. |

Quoicoubeh et feur figurent dans des phrases qui explicitent le jeu de mots ; aucune règle `quoi → quoicoubeh` n’est introduite. Skibidi est cité comme référence absurde, pas associé à une fausse définition stable. Les termes de détresse figurée ne sont pas des diagnostics. Une déconnexion ne devient pas automatiquement un rage quit ; une amitié ne devient pas automatiquement un crush.

## Origine, droits et limites éditoriales

Les exemples sont rédigés par l’agent pour ce projet, sans collecte de conversations réelles, messages privés ou publications personnelles. Les noms des exemples sont fictifs. Les références culturelles ne sont pas des créations du projet. Le texte original des exemples est proposé sous CC0-1.0 dans la mesure des droits applicables ; cette indication ne revendique aucun droit sur des œuvres, personnages, marques ou références préexistantes.

Les sources de contexte sont documentées dans [SOURCES.md](SOURCES.md). Elles servent à comprendre certains usages, pas de corpus copié. Aucune représentativité du langage des jeunes en 2026 n’est revendiquée : les références couvrent plusieurs périodes et des usages différents.

**Aucune relecture humaine n’a encore été effectuée.** `review_status` reste `pending_human_review`. La relecture éditoriale par l’agent et les contrôles automatiques ne garantissent pas la fidélité parfaite de chaque exemple. Il faut encore vérifier le naturel, la précision et les emplois ironiques avec des locuteurs.

## Construction et séparation

Entrées versionnées :

- `seeds.txt` : 250 phrases de base conservées sans leurs anciens suffixes ;
- `meme_seeds.txt` : 521 exemples contextualisés supplémentaires ;
- `compositions.json` : 36 patrons avec 16 remplissages chacun, pour travailler aussi la composition ;
- `references.tsv` : inventaire lexical et notes de sens.

Après déduplication : 771 paires rédigées et 575 paires compositionnelles, 788 groupes. Les compositions ne sont pas présentées comme des situations indépendantes ; leur provenance les identifie. Les patrons mettent en jeu des actions, objets et contextes contrôlés. Ils ne remplacent pas un corpus naturel diversifié.

Tous les exemples d’un patron restent dans une seule partition. On fusionne aussi les composantes dont un côté normalisé est identique ou atteint un ratio `SequenceMatcher >= 0.88`. La comparaison normalise NFC, casse, apostrophes et ponctuation ; les textes livrés gardent leur graphie. Un groupe ne traverse jamais deux partitions, même s’il couvre plusieurs thèmes.

Le découpage déterministe utilise la graine 42 et une stratification par thème. On réserve environ 10 % des groupes à chaque ensemble de contrôle ; les groupes n’ont pas la même taille et ne sont jamais coupés. Les références lexicales présentes sont toutes représentées dans le train. Les références très rares sont donc gardées dans le train : ce test évalue de **nouveaux contextes**, pas la découverte de mots totalement inconnus.

| Partition | Paires |
| --- | ---: |
| Train | 976 |
| Validation | 201 |
| Test | 169 |

Les proportions en lignes diffèrent volontairement de 80/10/10 pour préserver les groupes entiers. Le plus grand groupe contient 34 lignes. Le regroupement reste heuristique : certaines paraphrases sémantiques peuvent lui échapper. Deux ambiguïtés de surface restantes sont des paraphrases compatibles, conservées dans un même groupe et comptées dans le rapport.

La version du corpus a changé : les scores de la v1 et de la v2 ne sont pas directement comparables. Repartir de poids et d’un tokenizer neufs ; ne pas reprendre un ancien checkpoint avec `--resume`. Les anciennes évaluations sont historiques, pas un résultat sur ce corpus.

## Contrôles reproductibles

```sh
uv run python scripts/prepare_dataset.py
uv run python scripts/audit_dataset.py
uv run pytest -q
```

La préparation n’utilise que la bibliothèque standard. `--output-dir /tmp/attention-data-check` reconstruit les partitions ailleurs ; `--seed` permet de modifier la graine. Les contrôles vérifient schéma, provenance, identifiants, normalisation NFC, doublons normalisés, conservation des nombres écrits en chiffres, taille maximale de 500 caractères, couverture du corpus et absence de groupes partagés.

`audit_dataset.py` entraîne un BPE de 1 024 tokens sur le train uniquement et exerce le pipeline réel sur les trois ensembles. Aucun texte n’est tronqué. Les séquences livrées font au plus **33 tokens**, tokens de contrôle compris, pour une limite de 64. La restitution Unicode est vérifiée, accents et emojis compris. `tokenization_report.json` conserve les empreintes des fichiers audités. Ces longueurs dépendent du tokenizer choisi.

`report.json` donne les tailles, groupes, thèmes, modes de construction, couverture lexicale par partition, doublons, ambiguïtés de surface, conservation des chiffres et empreinte SHA-256. Une vérification numérique ne détecte ni toutes les erreurs d’entités ni une négation mal reformulée : ces aspects demandent une relecture sémantique.

## Relecture et entraînement

`review_sample.jsonl` propose **100 paires du train**, avec couverture des thèmes et de l’inventaire lexical. Vérifier les deux directions : sens conservé, français standard naturel, registre Internet plausible, absence de fait inventé, préservation des noms/nombres/négations et intensité. Corriger les fichiers d’entrée, reconstruire et versionner toute modification. Ne pas utiliser le test pour choisir les corrections ou les hyperparamètres.

La présence de plus de références ne garantit pas un bon modèle. Le corpus reste petit, synthétique, déséquilibré et partiellement composé ; certaines expressions restent rares. Il faudra des exemples indépendants relus, davantage de paraphrases et une évaluation humaine pour obtenir une traduction robuste. Les sorties inventées par le modèle ne deviennent jamais automatiquement de nouvelles références d’apprentissage.
