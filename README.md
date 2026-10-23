# Attention — français standard ↔ Gen Z

Transformer encodeur-décodeur recodé avec PyTorch, API FastAPI et interface React/Next.js. Les trois lots sont intégrés sur la branche `dev`.

Le premier checkpoint valide le fonctionnement technique, mais ses traductions inédites restent de mauvaise qualité. Les données sont synthétiques et demandent une relecture humaine. Voir [les expériences](configs/EXPERIMENTS.md) et [le corpus](data/README.md).

## Installation

Python 3.14+, uv et Node.js 26 recommandé.

```sh
uv sync
cd web
npm ci
```

## Lancement local

Depuis la racine, démarrer l’API dans un terminal :

```sh
ATTENTION_MODEL_DIR=checkpoints/local-bpe512 uv run uvicorn attention.api:app --host 127.0.0.1 --port 8000
```

Dans un second terminal :

```sh
cd web
npm run dev
```

Ouvrir http://localhost:3000. Le proxy utilise `http://127.0.0.1:8000` par défaut. Le mode mock est désactivé. Les checkpoints sont locaux et exclus de Git : sur une nouvelle installation, entraîner le modèle avant de lancer l’API avec ce dossier. Sans checkpoint, l’interface affiche son indisponibilité.

## Données et entraînement

```sh
uv run python scripts/prepare_dataset.py
uv run python -m attention.train --train data/train.jsonl --validation data/validation.jsonl --output checkpoints/local-bpe512 --config configs/default.json --epochs 50 --batch-size 32 --lr .0005 --patience 10 --vocab-size 512 --threads 4
```

Les commandes d’évaluation, de reprise et le test de mémorisation sont détaillés dans `configs/EXPERIMENTS.md`.

## Vérifications

```sh
uv run pytest -q
cd web
npm run typecheck
npm run lint
npm test
npm run build
```

La [spécification](SPEC.md) décrit l’architecture et les contrats ; [web/README.md](web/README.md) décrit le frontend.
