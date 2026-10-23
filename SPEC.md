# Spécification — Traduction français standard ↔ français Gen Z

Statut : spécification révisée, en attente de confirmation avant lancement des trois agents. Cette phase ne crée aucune branche et ne lance aucune implémentation.

## 1. Objectif

Construire un Transformer encodeur-décodeur en Python avec PyTorch, entraîné depuis zéro, puis proposer une interface web permettant de reformuler une phrase dans les deux sens :

- français standard → français Gen Z ;
- français Gen Z → français standard.

Il s’agit d’un transfert de style au sein du français. Le résultat doit conserver le sens, les noms, les nombres et la négation, tout en adaptant le registre. Le style Gen Z visé est un français informel écrit, défini par le corpus ; il ne doit pas ajouter systématiquement de l’argot ou des emojis.

Exemples indicatifs, à harmoniser lors de la création du corpus :

| Sens | Entrée | Sortie possible |
| --- | --- | --- |
| Standard → Gen Z | Ce film est vraiment excellent. | Ce film est une dinguerie. |
| Gen Z → standard | J’ai trop le seum. | Je suis vraiment déçu. |

La priorité est pédagogique : comprendre et vérifier chaque brique. Une qualité comparable à celle d’un grand modèle préentraîné n’est pas un objectif de la première version.

## 2. Périmètre et choix proposés

- Un seul modèle et un vocabulaire partagé pour les deux directions.
- Un marqueur source `<to_genz>` ou `<to_standard>` sélectionne la direction.
- Architecture Transformer encodeur-décodeur avec attention multi-têtes, encodage positionnel sinusoïdal, connexions résiduelles, normalisation et dropout.
- Implémentation manuelle de l’attention et des blocs. Ne pas utiliser `nn.Transformer`, `nn.MultiheadAttention` ou une fonction d’attention complète comme raccourci.
- Utilisation autorisée des primitives PyTorch : tenseurs, autograd, couches linéaires, embeddings, `LayerNorm`, dropout, fonctions d’activation, pertes et optimiseurs.
- Aucun poids préentraîné pour le traducteur. Le tokenizer peut utiliser une bibliothèque dédiée.
- Interface React avec Next.js App Router et TypeScript, dans `web/`, reliée à une API Python FastAPI.
- Next.js assure l’interface et un proxy serveur vers FastAPI ; PyTorch et le chargement du modèle restent dans le processus Python.
- Première version limitée aux phrases courtes ; longueur maximale explicite, sans troncature silencieuse.

Hors périmètre initial : déploiement public, comptes utilisateurs, conversation avec historique, traduction de documents et apprentissage en ligne.

## 3. État du code existant

`src/attention/transformers.py` contient un tokenizer tiktoken, des embeddings, de l’attention, un feed-forward et des blocs encodeur/décodeur.

Points à traiter pendant l’implémentation :

- Enregistrer les têtes dans `nn.ModuleList` pour que leurs paramètres soient entraînés et sauvegardés ; utiliser une dimension de tête entière et vérifier la divisibilité.
- Stocker et appliquer réellement les masques ; le retour de `masked_fill` doit être utilisé.
- Séparer masque causal, padding source et padding cible, avec des formes compatibles avec les batches et la cross-attention.
- Tester `query is not None` plutôt que la valeur booléenne d’un tenseur.
- Remplacer `torch.norm` par des couches `nn.LayerNorm` apprenables, qui préservent les dimensions.
- Utiliser la sortie de la première sous-couche du décodeur comme requête et comme connexion résiduelle de la cross-attention.
- Enregistrer l’encodage positionnel comme buffer ; sa longueur dépend de la longueur maximale des séquences, pas de la taille du vocabulaire.
- Ajouter embeddings au modèle complet, empilement des blocs, dropout et projection finale vers le vocabulaire.
- Clarifier ou supprimer les paramètres de configuration actuellement inutilisés, notamment `masked_h`.

Ce document ne modifie pas encore le code.

## 4. Données et tokenisation

### Corpus parallèle

Chaque ligne du corpus contient `id`, `standard`, `genz`, `group_id`, `topic` et `provenance`, au format JSONL UTF-8. `provenance` est un objet contenant `kind` (`synthetic` ou `sourced`), `source` et `usage_rights`. Ne pas inclure de données personnelles. `group_id` regroupe les variantes issues d’une même phrase ou d’un même patron de génération.

Les paires doivent conserver le sens et couvrir différentes situations : études, sorties, émotions, travail et échanges courants. Inclure négations, nombres, noms propres et phrases qui demandent peu de changement. Une sortie ne doit pas introduire d’insulte, de jugement ou de fait absent de l’entrée.

Livrer un corpus initial d’au moins 1 000 paires distinctes, avec au moins huit thèmes et sans gonfler ce nombre par simple substitution de noms ou de nombres. Documenter les limites de diversité : ce volume constitue un point de départ pédagogique, pas une garantie de généralisation. Les exemples synthétiques sont autorisés et doivent être identifiés ; ne pas présenter une relecture automatique comme une validation humaine. Fournir un échantillon de 50 paires à relire manuellement et distinguer les vérifications réellement effectuées de celles restant à faire.

Le registre initial est informel et compréhensible : argot courant, quelques abréviations et anglicismes, emojis occasionnels seulement. Pas d’ajout systématique d’une expression à chaque phrase. Documenter les choix dans `data/README.md`.

### Découpage

- Répartir les paires en train/validation/test, à proportions approximatives 80/10/10, avec la graine 42 et sans séparer un même `group_id`.
- Dédupliquer et regrouper les variantes proches avant la séparation pour limiter les fuites.
- Séparer les paires avant de créer leurs deux directions : une paire et son inverse restent dans le même sous-ensemble.
- Réserver le test à l’évaluation finale ; utiliser la validation pour choisir les réglages et le checkpoint.

### Tokenizer

Utiliser un tokenizer BPE partagé via la bibliothèque `tokenizers`, appris uniquement sur le train, avec un pré-découpage ByteLevel et son décodeur associé pour préserver les caractères français et emojis. Normalisation Unicode NFC, sans suppression des accents ni passage systématique en minuscules. Vocabulaire cible initial de 2 000 tokens, configurable et adapté si le corpus le nécessite. Ce tokenizer remplace l’usage de `o200k_base` dans le pipeline final.

Tokens réservés : `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<to_genz>` et `<to_standard>`. Leurs identifiants sont sauvegardés avec le tokenizer. Préserver accents, apostrophes, ponctuation et emojis ; documenter la normalisation Unicode.

Pour une paire, créer deux exemples d’apprentissage :

```text
source : <to_genz> phrase standard <eos>
cible  : <bos> phrase Gen Z <eos>

source : <to_standard> phrase Gen Z <eos>
cible  : <bos> phrase standard <eos>
```

## 5. Contrat du modèle

Notations : `B` batch, `S` longueur source, `T` longueur cible, `D` dimension du modèle, `V` taille du vocabulaire.

| Élément | Forme / comportement |
| --- | --- |
| Identifiants source | `(B, S)`, entiers |
| Identifiants cible en entrée du décodeur | `(B, T)`, entiers |
| Embeddings et états cachés | `(B, longueur, D)` |
| Logits en sortie | `(B, T, V)` ; pas de softmax avant la perte |
| Dimension d’une tête | `D / nombre_de_têtes`, entier |
| Attention | `softmax(QKᵀ / sqrt(d_head) + masque) V` |

L’encodeur utilise une self-attention non causale et ignore le padding source. Le décodeur utilise une self-attention causale qui ignore le padding cible, puis une cross-attention sur la sortie de l’encodeur qui ignore le padding source. La normalisation est définie explicitement, par exemple en post-norm : `LayerNorm(x + Dropout(sous_couche(x)))`.

Configuration de départ indicative, à ajuster aux ressources : `D=128`, 4 têtes, 2 blocs encodeur, 2 blocs décodeur, feed-forward de dimension 512, dropout 0,1 et longueur maximale de 64 tokens, tokens spéciaux compris. La taille du vocabulaire doit rester adaptée au corpus. Enregistrer tous les hyperparamètres.

## 6. Entraînement

- Préparer les batches avec padding et masques associés.
- Utiliser le teacher forcing : entrée décodeur = cible sans son dernier token ; labels = cible sans son premier token.
- Calculer une cross-entropy qui ignore `<pad>` ; agréger les métriques par token utile.
- Utiliser AdamW, un learning rate configurable et un clipping des gradients.
- Fixer et sauvegarder la graine, les paramètres du run et les versions de l’environnement ; vérifier le matériel disponible et sélectionner CPU, CUDA ou MPS selon disponibilité.
- Mesurer la perte train et validation à chaque époque ; validation sans gradients et en mode évaluation.
- Sauvegarder le meilleur checkpoint selon la perte de validation ; prévoir une limite d’époques et un arrêt anticipé configurable.
- Sauvegarder poids, configuration, tokenizer, découpage des données et métriques. Pour reprendre un entraînement, sauvegarder également état de l’optimiseur et progression.

Avant tout entraînement significatif, vérifier que le modèle peut surapprendre un tout petit lot de paires dans les deux sens. Cet essai valide le pipeline, pas la généralisation.

## 7. Inférence, API et interface web

Une fonction commune `translate(text, direction)` effectue la validation, la tokenisation, l’encodage de la source et la génération autorégressive. Commencer par un décodage glouton, depuis `<bos>`, jusqu’à `<eos>` ou une limite explicite. Utiliser `eval()` et désactiver les gradients. Retirer les tokens de contrôle du texte affiché.

### Contrat HTTP partagé — version 1

L’agent modèle implémente FastAPI ; l’agent interface consomme ce contrat sans modifier le backend. Les directions ont exactement les valeurs `to_genz` et `to_standard`.

`POST /api/translate` reçoit :

```json
{"text": "Ce film est vraiment excellent.", "direction": "to_genz"}
```

Réponse HTTP 200 :

```json
{
  "translation": "Ce film est une dinguerie.",
  "direction": "to_genz",
  "truncated": false,
  "model_id": "local-v1"
}
```

`truncated` indique que la génération a atteint sa limite sans EOS. `model_id` identifie le checkpoint chargé. L’exemple de réponse illustre le format et ne promet pas cette sortie exacte.

Toutes les erreurs applicatives adoptent ce format :

```json
{"error": {"code": "MODEL_UNAVAILABLE", "message": "Aucun modèle entraîné n’est disponible."}}
```

| Statut | Code | Cas |
| --- | --- | --- |
| 400 | `EMPTY_TEXT` | Texte vide après suppression des espaces extérieurs |
| 400 | `INPUT_TOO_LONG` | Plus de 500 caractères Unicode ou limite source de tokens dépassée |
| 422 | `INVALID_REQUEST` | JSON invalide, type incorrect ou direction inconnue |
| 503 | `MODEL_UNAVAILABLE` | Checkpoint ou tokenizer absent, incompatible ou non chargé |
| 500 | `INTERNAL_ERROR` | Échec inattendu, sans exposer les détails internes au navigateur |
| 504 | `BACKEND_TIMEOUT` | Délai du proxy Next.js dépassé |

La validation Python doit aussi normaliser les erreurs FastAPI dans cette enveloppe. Pour les requêtes invalides, la validation prime sur la disponibilité du modèle. La longueur en caractères utilise des points de code Unicode, y compris côté interface ; la limite en tokens est vérifiée par Python, marqueur de direction et EOS compris, sans troncature silencieuse.

`GET /api/health` renvoie HTTP 200 avec `status` (`ready` ou `unavailable`), `model_id` (chaîne ou `null`) et `limits` : `max_input_chars=500`, `max_source_tokens=64`, `max_new_tokens=63`. Les limites de tokens sont issues de la configuration du checkpoint lorsqu’il est chargé. L’absence de checkpoint laisse l’API démarrer et signaler son indisponibilité.

Next.js expose les mêmes routes au navigateur et relaie les requêtes vers FastAPI. `ATTENTION_API_URL`, variable uniquement serveur, configure l’adresse Python, par défaut `http://127.0.0.1:8000`. Délai du proxy : 60 secondes. Une connexion impossible produit HTTP 503 avec `BACKEND_UNAVAILABLE`. Les erreurs du backend sont conservées. Les requêtes restent de même origine côté navigateur.

### Expérience et direction visuelle

L’interface est entièrement en français, soignée et utilisable sur mobile comme sur ordinateur. Prévoir une composition claire en deux panneaux sur grand écran et empilée sur mobile : saisie et résultat. Une palette sobre, un accent coloré, une typographie lisible, des espacements réguliers et des transitions discrètes doivent donner un résultat fini. Éviter les éléments décoratifs qui gênent la lecture.

L’interface comprend :

- un champ texte source et un sélecteur de direction ;
- un bouton de traduction et une zone de résultat copiable ;
- quelques exemples pour chaque direction ;
- un état de traitement et des erreurs lisibles pour entrée vide, entrée trop longue ou checkpoint absent ;
- une indication si la génération atteint sa limite sans produire `<eos>` ;
- une mention claire du caractère expérimental du modèle.

Ajouter un compteur de caractères, une action d’inversion de direction, un bouton pour effacer et un retour de confirmation après copie. Si une traduction réussie est présente, l’inversion peut la transférer dans le champ source. Une modification du texte ou de la direction invalide le résultat affiché. Éviter les doubles soumissions et ignorer toute réponse devenue obsolète.

Prévoir explicitement les états initial, saisie, chargement, succès, génération limitée, backend indisponible et erreur. Les champs ont des labels, les actions sont accessibles au clavier, le focus est visible et les messages dynamiques sont annoncés par une région accessible. Le résultat est rendu comme du texte, sans interprétation HTML.

Le frontend peut être développé avec des fixtures conformes au contrat, dans un mode mock activé explicitement par configuration et signalé visuellement. Ce mode est désactivé par défaut ; ne jamais afficher une traduction simulée comme une sortie réelle du modèle.

Charger le modèle une seule fois au démarrage et partager exactement le tokenizer, les paramètres et le prétraitement de l’entraînement. L’interface doit fonctionner avec un checkpoint local, sans service de génération externe.

## 8. Étapes et critères de validation

| Étape | Livrable | Critère pour passer à la suite |
| --- | --- | --- |
| 1. Cadrage du corpus | Règles de style et paires représentatives relues | Les deux directions et la conservation du sens sont définies |
| 2. Données et tokenizer | Chargement, découpage, tokenizer sauvegardé, batches | Tokens spéciaux stables, décodage vérifié, absence de fuite entre sous-ensembles |
| 3. Briques du Transformer | Attention, embeddings, masques, blocs | Formes correctes, paramètres enregistrés, gradients finis |
| 4. Modèle complet | Encodeur-décodeur et projection vocabulaire | Un forward et un backward fonctionnent sur un batch avec padding |
| 5. Pipeline d’entraînement | Boucle, métriques et checkpoints | Le modèle surapprend un minuscule corpus dans les deux sens |
| 6. Génération | Traduction locale depuis un checkpoint | Génération bornée, arrêt sur EOS, rechargement reproductible |
| 7. Expérience sur le corpus | Run documenté et évaluation | Résultats séparés par direction et limites identifiées |
| 8. Interface web | Application locale | Les deux directions fonctionnent ; erreurs d’entrée et de chargement gérées |
| 9. Livraison pédagogique | README et exemples | Installation, entraînement et lancement reproductibles depuis les instructions |

### Vérifications techniques ciblées

- Le masque causal empêche une position de dépendre des tokens futurs.
- Modifier les valeurs des clés/valeurs aux positions de padding ne modifie pas les sorties utiles en évaluation.
- La cross-attention accepte des longueurs source et cible différentes.
- Tous les paramètres des têtes apparaissent dans `model.parameters()` et reçoivent des gradients.
- Une étape d’optimisation produit une perte et des gradients finis.
- Sauvegarder puis recharger conserve les logits en mode évaluation, à la tolérance numérique près.
- La génération gère EOS immédiat et limite de longueur sans boucle infinie.

### Évaluation de la qualité

Évaluer sur des phrases inédites, séparément pour chaque direction : conservation du sens, respect du style demandé et lisibilité. Conserver un petit tableau d’exemples avec réussites et échecs. Les métriques automatiques comme BLEU ou chrF peuvent compléter l’analyse, mais une référence unique ne décrit pas toutes les reformulations valides.

Comparer également au texte source inchangé pour détecter un modèle qui copie systématiquement. Fixer les objectifs chiffrés de qualité et de latence après un premier run sur le corpus et le matériel retenus.

## 9. Organisation proposée

Conserver le code du modèle dans `src/attention/transformers.py` au début pour faciliter sa lecture. Extraire ensuite les responsabilités si sa taille le justifie.

```text
SPEC.md
README.md
src/attention/
  transformers.py      # Architecture du modèle
  tokenizer.py         # Tokenizer et tokens spéciaux
  data.py              # Corpus, découpage et batches
  train.py             # Entraînement et validation
  inference.py         # Chargement et génération
  api.py               # API FastAPI et contrat HTTP
scripts/
  prepare_dataset.py   # Validation et découpage reproductible du corpus
web/                   # React, Next.js App Router, TypeScript, proxy API
configs/               # Hyperparamètres reproductibles
data/                  # Corpus et découpages, selon droits de diffusion
checkpoints/           # Artefacts locaux, exclus de Git
tests/                 # Vérifications techniques ciblées
```

Le README documentera les commandes réelles une fois implémentées. Ne pas ajouter de dépendances ni modifier l’environnement pendant la phase de spécification.

## 10. Trois agents, trois branches

Après confirmation explicite de l’utilisateur, lancer exactement trois agents dans trois worktrees isolés, issus du même commit contenant cette spécification. Ne pas lancer les agents ni créer leurs branches pendant la rédaction de la spec. Les agents suivent ce document comme contrat commun ; tout changement de contrat est coordonné avant application.

| Agent / branche | Responsabilité | Fichiers possédés |
| --- | --- | --- |
| Données — `codex/dataset` | Créer, documenter, valider et découper le corpus | `data/**`, `scripts/prepare_dataset.py`, `tests/test_dataset.py` |
| Modèle — `codex/model-inference` | Corriger le Transformer, construire tokenizer, batches, entraînement, inférence et API Python | `src/attention/**`, `configs/**`, tests Python hors dataset, `pyproject.toml`, `uv.lock` |
| Interface — `codex/web-interface` | Réaliser l’interface React/Next.js, le proxy et les états d’interaction | `web/**`, dont dépendances, lockfile, `.env.example`, README et tests frontend |

Chaque agent effectue des commits sur sa branche et fournit un compte rendu : fichiers changés, commandes de validation, résultats, limites et éventuels blocages. Aucun ne fusionne les branches des autres. `SPEC.md`, le README racine et `.gitignore` sont coordonnés par l’agent principal pour éviter les conflits.

### Agent données

- Livrer `data/pairs.jsonl`, `data/train.jsonl`, `data/validation.jsonl`, `data/test.jsonl`, `data/review_sample.jsonl` et `data/README.md`.
- Conserver le même schéma de paire dans tous ces fichiers ; ne pas produire les exemples bidirectionnels ni le tokenizer, qui appartiennent au pipeline modèle.
- Fournir un script relançable avec graine 42, rapport de taille, répartition des thèmes, doublons et contrôle d’étanchéité des groupes.
- Vérifier champs obligatoires, identifiants uniques, textes non vides, provenance et absence de doublons exacts. Documenter les règles employées pour repérer les variantes proches.
- Respecter la borne de 500 caractères ; mesurer les longueurs, sans prétendre garantir la limite de tokens avant l’apprentissage du tokenizer.

### Agent modèle et inférence

- Prendre en charge toute la chaîne Python, y compris l’entraînement : une inférence réelle nécessite un checkpoint entraîné.
- Lire directement les fichiers JSONL convenus ; pouvoir avancer d’abord avec un minuscule jeu de test interne, sans dépendre de la livraison du corpus complet.
- Livrer les vérifications techniques de la section 8, un essai de surapprentissage, la génération et le contrat HTTP de la section 7.
- Conserver le pipeline utilisable sans checkpoint avec une erreur explicite ; ne pas remplacer le modèle par des réponses codées en dur.
- Sauvegarder dans un dossier de run les poids, `tokenizer.json`, configuration et métriques. Le dossier est sélectionné via `ATTENTION_MODEL_DIR` pour l’API ; les poids restent hors Git.
- Après intégration du corpus, lancer un premier entraînement adapté aux ressources, documenter les résultats et distinguer validation technique et qualité linguistique. Si le calcul disponible limite l’expérience, fournir le checkpoint obtenu et les commandes de reprise, sans annoncer une qualité non mesurée.

### Agent interface web

- Implémenter Next.js App Router avec React et TypeScript, une feuille de style cohérente et des composants réutilisables à la taille du projet.
- Développer contre le contrat HTTP figé ; les fixtures ne sont que des outils de développement et de test.
- Fournir les scripts de développement, vérification TypeScript, lint et build, ainsi qu’un lockfile et les instructions de connexion au backend.
- Vérifier saisie, inversion, copie, chargement, erreurs, réponses obsolètes, mode mock explicite et affichage mobile/ordinateur.
- Livrer un frontend fonctionnel même si le backend est temporairement indisponible : message utile et possibilité de réessayer.

## 11. Intégration et définition de terminé

L’agent principal inspecte les trois livraisons puis les intègre dans l’ordre données, modèle, interface. Le travail peut être parallèle ; l’évaluation finale du modèle dépend du corpus et le contrôle de bout en bout dépend des trois lots.

La livraison complète exige :

1. Corpus validé, provenance documentée et découpages reproductibles sans groupe partagé.
2. Transformer manuel vérifié, pipeline d’entraînement exécutable et checkpoint rechargeable.
3. Traduction réelle depuis le navigateur dans les deux directions, avec le mode mock désactivé.
4. Gestion vérifiée d’un backend arrêté, d’un checkpoint absent, d’une entrée invalide et d’une génération limitée.
5. Build Next.js, vérification TypeScript, lint et tests ciblés réussis ; interface inspectée sur mobile et ordinateur.
6. README racine avec installation, préparation des données, entraînement, démarrage des deux services et résultats observés.

Les versions des bibliothèques seront vérifiées au moment de l’implémentation, en tenant compte du Python déclaré par le projet. Aucun changement d’environnement n’est nécessaire pendant cette phase de spécification.

Prochaine action : attendre la confirmation de l’utilisateur, puis préparer les trois branches et lancer les agents selon cette répartition.
