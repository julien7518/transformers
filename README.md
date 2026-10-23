# Attention — français standard ↔ Gen Z

Transformer encodeur-décodeur recodé avec PyTorch, entraîné depuis zéro, avec une API FastAPI et une interface React/Next.js. Le travail est intégré sur la branche `dev`.

Chaque paire du dataset sert dans les deux sens : français standard → Gen Z et Gen Z → français standard. Un token spécial indique au modèle le registre demandé.

**Le pipeline fonctionne, mais la qualité des traductions reste insuffisante.** Le modèle mémorise le train sans bien généraliser aux phrases inédites. Augmenter uniquement le nombre d’époques ne résout pas ce problème : la priorité reste la diversité des exemples et leur relecture humaine.

## Repères rapides

- [Installer les dépendances](#installation)
- [Préparer et vérifier le dataset](#préparer-et-vérifier-le-dataset)
- [Entraîner](#entraîner-un-modèle)
- [Reprendre un entraînement](#reprendre-un-entraînement)
- [Tester une traduction](#tester-une-traduction-dans-le-terminal)
- [Évaluer le modèle](#évaluer-le-modèle)
- [Lancer et tester l’API](#lancer-et-tester-lapi)
- [Lancer les tests techniques](#tests-techniques)

Toutes les commandes Python ci-dessous se lancent **depuis la racine du projet**. Les commandes frontend utilisent explicitement `npm --prefix web` pour rester dans ce même dossier.

## Installation

Prérequis : Python 3.14+, [uv](https://docs.astral.sh/uv/) et Node.js 26 recommandé pour le frontend.

```sh
uv sync
npm --prefix web ci
```

`uv sync` installe les dépendances Python dans `.venv`. Les checkpoints sont locaux et exclus de Git : après un clonage, il faut entraîner un modèle ou récupérer un dossier de checkpoint compatible.

## Préparer et vérifier le dataset

Le corpus v2 comprend 1 346 paires et un inventaire de 89 expressions et références. Il est synthétique et attend une relecture humaine. Voir [la documentation du corpus](data/README.md).

Les partitions sont déjà livrées. Après modification des exemples sources :

```sh
uv run python scripts/prepare_dataset.py
uv run python scripts/audit_dataset.py
```

La préparation reconstruit `data/pairs.jsonl`, `train.jsonl`, `validation.jsonl`, `test.jsonl` et les rapports. Les fichiers à éditer sont `data/seeds.txt`, `data/meme_seeds.txt` et `data/compositions.json` ; `data/references.tsv` décrit les expressions.

L’audit apprend un tokenizer BPE sur le train uniquement et vérifie les longueurs et la restitution du texte. Il ne tronque pas les phrases. Le découpage garde ensemble les variantes proches et les exemples d’un même patron pour limiter les fuites.

| Partition | Rôle |
| --- | --- |
| Train | Apprendre le tokenizer et les poids du modèle |
| Validation | Choisir le meilleur checkpoint et les réglages |
| Test | Évaluer une fois les réglages terminés |

## Entraîner un modèle

Utiliser un nouveau dossier de sortie pour chaque expérience :

```sh
uv run python -m attention.train \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --output checkpoints/mon-modele \
  --config configs/default.json \
  --epochs 100 \
  --batch-size 32 \
  --lr .0005 \
  --patience 15 \
  --vocab-size 1024 \
  --threads 4
```

Le pipeline apprend d’abord un tokenizer partagé. Pendant l’entraînement, le décodeur reçoit le début de la phrase cible et apprend à prédire le token suivant : c’est le *teacher forcing*. La perte ignore le padding. À chaque époque, le programme affiche les pertes train et validation ; le meilleur checkpoint est choisi selon la perte de validation.

| Paramètre | Rôle |
| --- | --- |
| `--output` | Dossier où sauvegarder tokenizer, poids, configuration et métriques |
| `--config` | Architecture du Transformer, définie dans un fichier JSON |
| `--epochs` | Nombre maximal de passages sur les données |
| `--batch-size` | Nombre d’exemples bidirectionnels traités ensemble |
| `--lr` | Vitesse d’apprentissage de l’optimiseur |
| `--patience` | Arrêt après ce nombre d’époques sans amélioration de la validation ; `0` désactive cet arrêt |
| `--vocab-size` | Taille cible du vocabulaire BPE |
| `--threads` | Nombre de threads PyTorch |
| `--device` | `auto` par défaut : CUDA, puis MPS, puis CPU selon disponibilité ; `cpu` force le CPU |
| `--seed` | Graine aléatoire, `42` par défaut |

`configs/default.json` définit un modèle de dimension 128, quatre têtes, deux blocs encodeur, deux blocs décodeur et une longueur maximale de 64 tokens.

Le dossier de sortie contient notamment :

| Fichier | Contenu |
| --- | --- |
| `best.pt` | Meilleur checkpoint selon la validation, utilisé pour l’inférence |
| `last.pt` | Dernier état complet, utilisé pour reprendre l’entraînement |
| `tokenizer.json`, `special_tokens.json` | Tokenizer et identifiants des tokens spéciaux |
| `config.json` | Configuration complète du modèle |
| `run.json` | Paramètres du run, environnement et empreintes des données |
| `metrics.json` | Historique des pertes |

Conserver ensemble les poids, le tokenizer et les fichiers de configuration. Un tokenizer d’un autre run n’est pas interchangeable avec celui du checkpoint.

## Reprendre un entraînement

Relancer la même commande, avec le même dossier, les mêmes données et la même configuration, en ajoutant `--resume` :

```sh
uv run python -m attention.train \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --output checkpoints/mon-modele \
  --config configs/default.json \
  --epochs 150 \
  --batch-size 32 \
  --lr .0005 \
  --patience 15 \
  --vocab-size 1024 \
  --threads 4 \
  --resume
```

`--epochs 150` signifie 150 époques **au total**, pas 150 supplémentaires. La reprise charge `last.pt`, l’optimiseur et la progression. Elle conserve aussi la patience déjà consommée ; elle n’annule pas un arrêt anticipé.

Si le dataset ou l’architecture change, repartir de zéro avec un nouveau dossier. Ne pas reprendre un checkpoint v1 sur les données v2.

## Tester une traduction dans le terminal

Français standard → Gen Z :

```sh
uv run python -m attention.inference \
  --model-dir checkpoints/mon-modele \
  --direction to_genz \
  "Cette soirée était excellente."
```

Gen Z → français standard :

```sh
uv run python -m attention.inference \
  --model-dir checkpoints/mon-modele \
  --direction to_standard \
  "J'ai trop le seum."
```

La réponse JSON contient `translation`, `direction`, `truncated` et `model_id`. `truncated: true` signifie que la génération a atteint sa limite sans produire de token de fin.

## Évaluer le modèle

Une fois les réglages terminés, lancer l’évaluation sur le test :

```sh
uv run python -m attention.evaluate \
  --model-dir checkpoints/mon-modele \
  --data data/test.jsonl \
  --output checkpoints/mon-modele/evaluation.json
```

Le rapport présente les résultats séparément pour les deux directions : correspondances exactes avec la référence, copies de l’entrée, générations limitées, temps et sorties détaillées. Une référence unique ne couvre pas toutes les reformulations valides : lire aussi les sorties pour juger le sens, le style et la grammaire.

Ne pas utiliser le test pour choisir le checkpoint ou les réglages. Si ses erreurs servent à orienter de nouvelles modifications, prévoir ensuite une nouvelle évaluation indépendante.

## Lancer et tester l’API

Dans un terminal, charger le dossier du modèle souhaité :

```sh
ATTENTION_MODEL_DIR=checkpoints/mon-modele \
uv run uvicorn attention.api:app --host 127.0.0.1 --port 8000
```

Pour essayer le checkpoint v2 déjà entraîné sur cette machine, remplacer `checkpoints/mon-modele` par `checkpoints/memes-v2`. Ce checkpoint reste expérimental et n’est pas inclus dans Git.

Le modèle est chargé au démarrage : **redémarrer l’API après un changement de checkpoint**. Sans checkpoint valide, elle démarre en état `unavailable` et refuse les traductions avec une erreur explicite.

Dans un autre terminal, vérifier son état :

```sh
curl http://127.0.0.1:8000/api/health
```

Si le modèle est chargé, `status` vaut `ready`. Pour tester une traduction :

```sh
curl http://127.0.0.1:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Cette soirée était excellente.","direction":"to_genz"}'
```

Pour le sens inverse, utiliser `"direction":"to_standard"`. Les entrées vides ou trop longues sont refusées, sans troncature silencieuse.

L’API propose également une [documentation interactive locale](http://127.0.0.1:8000/docs) lorsqu’elle est démarrée.

### Connexion à l’interface web

Dans un second terminal :

```sh
npm --prefix web run dev
```

Ouvrir [localhost:3000](http://localhost:3000). Next.js relaie les requêtes vers `http://127.0.0.1:8000` par défaut. Pour changer cette adresse, définir `ATTENTION_API_URL` dans `web/.env.local`, puis redémarrer Next.js. Le mode mock doit rester désactivé pour tester les vraies sorties du modèle. Voir [la documentation frontend](web/README.md).

## Tests techniques

Tests Python du dataset, du modèle et de l’API :

```sh
uv run pytest -q
```

Vérifications frontend depuis la racine :

```sh
npm --prefix web run typecheck
npm --prefix web run lint
npm --prefix web test
npm --prefix web run build
```

Ces tests vérifient le fonctionnement technique ; leur réussite ne garantit pas la qualité linguistique.

## Résultats actuels et documentation

Le premier entraînement v2 surapprend : la perte train baisse, tandis que la validation plafonne. Les traductions inédites présentent encore des erreurs de sens et des mots inventés. Il faut enrichir et relire les données avant de considérer le modèle comme fiable.

- [Corpus, provenance et règles de relecture](data/README.md)
- [Résultats de l’entraînement v2](data/TRAINING_V2.md)
- [Expériences historiques et test de mémorisation](configs/EXPERIMENTS.md)
- [Spécification du projet](SPEC.md)
