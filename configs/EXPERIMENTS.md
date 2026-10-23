# Expériences Python — 23 octobre 2026

## Exécution et limites

Exécution locale avec Python 3.14.9, PyTorch 2.14.1, tokenizers 0.23.3,
FastAPI 0.161.1, NumPy 2.5.4. CPU uniquement : CUDA indisponible, backend
MPS compilé mais indisponible dans cet environnement. Les latences ci-dessous
sont mesurées dans ce contexte, sans garantie sur un autre matériel.

Le pipeline est fonctionnel ; **la qualité linguistique sur des phrases inédites
n'est pas validée**. Les sorties du premier corpus contiennent des changements de
sens, des répétitions et des mots mal formés. Ne pas présenter le checkpoint
comme un traducteur fiable. Aucun poids préentraîné ni service de génération
externe n'est utilisé.

## Installation et commandes

Depuis la racine du projet :

```sh
uv sync
uv run pytest -q
uv run python -m attention.smoke
uv run python -m attention.train --train checkpoints/smoke-fixtures/train.jsonl --validation checkpoints/smoke-fixtures/validation.jsonl --output checkpoints/smoke --config configs/smoke.json --epochs 180 --batch-size 8 --lr .003 --patience 0 --vocab-size 400 --threads 2
uv run python -m attention.evaluate --model-dir checkpoints/smoke --data checkpoints/smoke-fixtures/train.jsonl --output checkpoints/smoke/memorization.json
uv run python -m attention.train --train data/train.jsonl --validation data/validation.jsonl --output checkpoints/local-v1 --config configs/default.json --epochs 35 --batch-size 32 --lr .0008 --patience 8 --threads 4
uv run python -m attention.train --train data/train.jsonl --validation data/validation.jsonl --output checkpoints/local-bpe512 --config configs/default.json --epochs 50 --batch-size 32 --lr .0005 --patience 10 --vocab-size 512 --threads 4
uv run python -m attention.evaluate --model-dir checkpoints/local-bpe512 --data data/test.jsonl --output checkpoints/local-bpe512/test-evaluation.json
uv run python -m attention.inference --model-dir checkpoints/local-bpe512 --direction to_genz 'Bonjour.'
ATTENTION_MODEL_DIR=checkpoints/local-bpe512 uv run uvicorn attention.api:app --host 127.0.0.1 --port 8000
```

Les exécutions initiales lisaient les fichiers du worktree données via leurs
chemins absolus ; les hashes et identifiants sont conservés dans chaque run.
Les fichiers sont ensuite les mêmes après intégration. Une entrée dépassant la
limite de tokens provoque une erreur explicite, jamais une troncature silencieuse.

Sans `ATTENTION_MODEL_DIR`, l'API démarre en état `unavailable`. Le checkpoint
n'est chargé qu'au démarrage ; redémarrer l'API après remplacement des artefacts.
Lancer un seul worker pour une seule copie mémoire du modèle. L'inférence est
sérialisée dans un worker ; une requête valide sans modèle reçoit HTTP 503.

Pour reprendre le dernier état (pas le meilleur checkpoint), répéter la commande
d'entraînement avec `--resume` et un nombre total d'époques supérieur. Les mêmes
configurations, tokenizer, contenus de splits et identifiants sont exigés ; leurs
chemins peuvent changer. L'optimiseur, les RNG et l'historique sont restaurés.
La reproductibilité numérique entre matériels différents n'est pas promise.

Chaque run contient `best.pt`, `last.pt`, `tokenizer.json`, `special_tokens.json`,
`config.json`, `run.json`, `metrics.json` et éventuellement les évaluations. Le
checkpoint lie le tokenizer à ses poids par SHA-256. Les poids restent hors Git.

## Validation technique

`uv run pytest -q` : **8 tests réussis**. Vérification du masque causal, du padding,
des longueurs distinctes en cross-attention, de l'enregistrement et des gradients
finis de toutes les têtes, d'une optimisation, du rechargement des logits, du BPE
NFC/accents/emojis, du décalage teacher forcing, d'EOS immédiat, de la génération
bornée, de la reprise, de l'incompatibilité du tokenizer et du contrat HTTP.

Surapprentissage : 4 paires, 8 directions, 180 époques, configuration 48/4/1/1,
78 561 paramètres. Temps boucle : 2,52 s (CPU, 2 threads). Perte finale en
évaluation : 0,00330 par token. **8/8 sorties gloutonnes exactes** ; aucun arrêt
par limite. Les phrases de validation sont volontairement les mêmes dans ce
seul test de mémorisation, avec des identifiants distincts. Ce score ne mesure
aucune généralisation. Exemples reproduits exactement :

| Entrée | Direction | Sortie |
| --- | --- | --- |
| Ce film est excellent. | to_genz | Ce film est une dinguerie. |
| Je suis KO. | to_standard | Je suis fatigué. |
| Je ne viens pas. | to_genz | Je viens pas. |

## Premier corpus

808 paires train, 96 validation, 96 test ; expansion bidirectionnelle après
séparation. Groupes train/validation disjoints vérifiés par le pipeline. Consulter
`data/README.md` pour la provenance synthétique et les limites de diversité :
1 000 paires ne correspondent pas à 1 000 situations indépendantes.

Deux réglages exploratoires, poids choisis selon le minimum de perte validation
**à l'intérieur de chaque run** :

| Run | BPE cible | Paramètres | Meilleure époque | Perte train / validation | Arrêt | Temps boucle |
| --- | --- | --- | --- | --- | --- | --- |
| local-v1 | 2 000 | 1 316 850 | 2 | 1,730 / 4,047 | 10 | 15,80 s |
| local-bpe512 | 512 | 1 057 280 | 7 | 0,995 / 2,500 | 17 | 30,85 s |

Les pertes par token **ne sont pas directement comparables entre vocabulaires**.
Le second run explore un vocabulaire plus petit, sans amélioration de qualité
linguistique démontrée. Les deux ont 0 % de correspondance exacte sur validation.
Aucun réglage supplémentaire n'a été fait après consultation du test.

Évaluation finale descriptive du run exploratoire `local-bpe512` sur le test :

| Sens | Exemples | Sortie exacte | Copie source | Source inchangée exacte (référence simple) | Génération limitée | Latence moyenne |
| --- | --- | --- | --- | --- | --- | --- |
| to_genz | 96 | 0 % | 0 % | 6,25 % | 0/96 | 17,7 ms |
| to_standard | 96 | 0 % | 0 % | 6,25 % | 1/96 | 17,6 ms |

Une référence unique n'épuise pas toutes les reformulations correctes ; toutefois
les erreurs observées confirment ici l'insuffisance du modèle. Le modèle fait
moins bien que la copie inchangée en correspondance exacte. Ces temps mesurent
la fonction locale à deux threads, sans transport HTTP ni navigateur.

| Entrée inédite | Sens | Sortie observée | Problème |
| --- | --- | --- | --- |
| J'ai oublié mon cahier. | to_genz | J'ai zappé cat. | Objet perdu et mot mal formé |
| J'ai zappé mon cahier. | to_standard | J'ai oublié mon méthode. | Objet remplacé, accord incorrect |

Les rapports JSON complets conservent les 192 sorties, références, temps et
indicateurs d'arrêt. Un travail futur doit enrichir le corpus en situations et
variantes indépendantes, puis réévaluer sur un nouveau test réservé. Il n'y a
pas de seuil de qualité ou de latence annoncé comme atteint.
