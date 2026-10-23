# Essai après refonte du corpus — 23 octobre 2026

Le corpus a été figé avant cet essai. Entraînement depuis zéro : aucun poids, tokenizer ou état d’optimiseur de la v1 réutilisé.

```sh
uv run python -m attention.train --train data/train.jsonl --validation data/validation.jsonl --output checkpoints/memes-v2 --config configs/default.json --epochs 100 --batch-size 32 --lr .0005 --patience 15 --vocab-size 1024 --threads 4
uv run python -m attention.evaluate --model-dir checkpoints/memes-v2 --data data/test.jsonl --output checkpoints/memes-v2/test-evaluation.json
```

Configuration : Transformer 128 dimensions, quatre têtes, deux blocs encodeur et deux décodeur, vocabulaire BPE de 1 024 tokens appris sur le train. Exécution CPU, quatre threads pour l’entraînement et deux pour l’évaluation. Train 976 paires, validation 201, test 169 ; chaque paire est exploitée dans les deux sens.

Arrêt anticipé après 36 époques, environ 84,5 secondes de boucle d’entraînement. Meilleur checkpoint sélectionné à l’époque 21 sur la validation : perte train 1,2534, validation 3,0269. Le train continue de baisser tandis que la validation plafonne : surapprentissage manifeste. Ces pertes ne sont pas comparables à celles de la v1, dont le vocabulaire et les données étaient différents.

Évaluation descriptive finale, sans réglage supplémentaire après lecture du test :

| Sens | Exemples | Sortie exacte | Copie de l’entrée | Génération limitée |
| --- | ---: | ---: | ---: | ---: |
| Standard → Gen Z | 169 | 0 % | 0 % | 2 / 169 |
| Gen Z → standard | 169 | 0 % | 0 % | 0 / 169 |

Une seule référence ne couvre pas toutes les reformulations valides, mais les sorties contiennent aussi de vraies erreurs de sens et des mots mal formés. Par exemple, « La bibliothèque est déjà pleine. » devient « La réunion est débutant. ». Ce résultat ne permet pas de présenter le modèle comme un traducteur fiable, ni de conclure à une amélioration linguistique par rapport à la v1.

Le checkpoint `checkpoints/memes-v2` reste local, hors Git. Il n’a pas été activé automatiquement dans l’interface. `training_v2_summary.json` conserve configuration, résultats et empreintes des données ; le rapport complet des sorties se trouve dans le dossier de checkpoint.

La refonte apporte plus de diversité éditoriale, des références explicites et de meilleurs contrôles, mais le jeu reste petit et synthétique. Il faut poursuivre par une relecture humaine, des exemples indépendants plus nombreux et des paraphrases de qualité. Les compositions ne doivent pas être multipliées artificiellement pour atteindre un nombre arbitraire. Il faudra réserver une nouvelle évaluation indépendante avant de mener une nouvelle série de réglages fondée sur les erreurs observées ici.
