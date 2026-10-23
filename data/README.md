# Corpus pédagogique français standard ↔ français Gen Z

Ce corpus contient **1 000 paires synthétiques**, issues de **250 noyaux
sémantiques rédigés par l'agent IA**, chacun décliné en quatre contextes
pragmatiques : phrase seule, annonce à l'interlocuteur, proposition d'en reparler
demain, souhait de tenir l'interlocuteur informé. Ce ne sont donc **pas 1 000
situations indépendantes**. Les suffixes apportent un acte de communication
supplémentaire, sans substitutions de noms ou de nombres pour gonfler le volume.
Ce premier jeu sert surtout à vérifier et explorer le pipeline ; sa diversité et
sa taille ne permettent pas de promettre une généralisation linguistique.

## Origine et droits

`seeds.txt` contient les 250 paires originales et les dix thèmes. Aucun réseau
social, corpus tiers, message privé ou texte personnel n'a été collecté. Emma et
Lyon sont des références fictive/géographique génériques, sans information sur
une personne réelle. Les textes synthétiques produits pour ce projet sont mis à
disposition sous **CC0-1.0**, dans la mesure des droits applicables. Les champs
`provenance` précisent cette origine ; `synthetic` ne signifie pas témoignage réel
ni validation humaine. Les phrases ne constituent pas des citations.

## Registre

Français informel écrit compréhensible : suppression occasionnelle de « ne »,
« capter », « bosser », « seum », « dinguerie », « grave », « chill », « checker ».
Les intensités sont conservées autant que possible et aucun emoji n'est ajouté.
Certaines phrases restent identiques lorsque la formulation convient aux deux
registres. Des noms, horaires, nombres, accents, apostrophes et négations sont
présents. Le style est volontairement modéré ; il représente une convention
pédagogique et non une définition universelle du langage d'une génération.

## Fichiers et reproductibilité

```sh
python3 scripts/prepare_dataset.py
python3 -m unittest discover -s tests -p test_dataset.py -v
```

Le script utilise uniquement la bibliothèque standard. Depuis un autre dossier,
utiliser son chemin absolu. `--output-dir /tmp/attention-data-check` permet de
reconstruire ailleurs ; `--seed` vaut 42 par défaut. Le corpus est normalisé NFC,
sans suppression des accents ou des majuscules. Tous les JSONL contiennent le
même schéma : `id`, `standard`, `genz`, `group_id`, `topic`, `provenance`.
`pairs.jsonl` est la collection complète ; `train.jsonl`, `validation.jsonl` et
`test.jsonl` contiennent les partitions. `report.json` décrit les effectifs,
thèmes, groupes, longueurs et l'empreinte SHA-256 du corpus.

Avant découpage, les noyaux dont un côté normalisé présente un ratio
`difflib.SequenceMatcher >= 0.88` sont fusionnés par composantes connexes.
La normalisation de comparaison applique NFC, minuscules, uniformisation des
apostrophes et retrait de la ponctuation. Les quatre variantes de chaque noyau
héritent du même groupe. Les groupes sont mélangés avec la graine 42 par thème,
puis répartis environ 80/10/10. Un groupe couvrant plusieurs thèmes est affecté
au premier thème pour la stratification et reste indivisible. L'arrondi des
groupes explique les écarts aux proportions exactes. Aucune paire inverse n'est
créée ici : le pipeline modèle la créera dans la partition de la paire originale.

Les contrôles vérifient schéma, provenance, identifiants, doublons de paires
normalisées, textes identiques dans des groupes différents, longueur maximale
500 points de code Unicode, couverture des partitions et étanchéité des groupes.
Le regroupement lexical reste une heuristique : il ne détecte pas toutes les
paraphrases sémantiques. Les suffixes de contexte sont volontairement partagés
entre partitions ; les noyaux et leurs variantes proches sont regroupés, mais les
tests ne mesurent pas la généralisation à de nouveaux patrons de contexte.
La borne de tokens reste à mesurer avec le tokenizer appris sur le train.

## Relecture et limites

`review_sample.jsonl` fournit **50 paires du train**, cinq par thème, à relire
manuellement dans les deux sens. Aucune relecture humaine n'a été effectuée.
Les vérifications réalisées sont programmatiques et une inspection éditoriale
par l'agent auteur ; elles ne prouvent pas la fidélité sémantique de chaque ligne.
La relecture doit notamment vérifier intensité, négation et adéquation sociale de
« seum », « chaud » ou « éclaté », dont le sens dépend du contexte. Le corpus
contient surtout des phrases courtes au présent, peu de questions, de syntaxe
complexe ou d'argot rare ; les suffixes répétitifs favorisent la mémorisation.

Utiliser la validation pour les réglages. Garder le test pour l'évaluation finale,
sans l'utiliser pour sélectionner le checkpoint. Après relecture du train, toute
correction des noyaux nécessite une reconstruction et un nouveau rapport ;
consigner les changements avant de comparer les expériences.
