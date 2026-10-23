# Entre nous — interface React / Next.js

Application locale en français. Prérequis : Node.js 26 recommandé (ou 22.22.2+ / 24.15+), et npm. Ces versions couvrent aussi les outils de test.

```sh
cd web
npm ci
cp .env.example .env.local
npm run dev
```

Ouvrir http://localhost:3000. Démarrer l’API Python séparément selon le README racine. `ATTENTION_API_URL` configure son origine (défaut `http://127.0.0.1:8000`), sans suffixe `/api`. Cette variable reste côté serveur. Redémarrer Next.js après modification de l’environnement.

Les routes Next.js `POST /api/translate` et `GET /api/health` relaient le contrat de SPEC.md, sans appel direct du navigateur à Python. Délai maximal : 60 secondes ; backend injoignable : `503 BACKEND_UNAVAILABLE` ; délai dépassé : `504 BACKEND_TIMEOUT`. Les erreurs et statuts Python sont préservés. Aucun checkpoint n’est requis pour afficher l’interface : les erreurs de disponibilité sont lisibles, avec possibilité de réessayer.

## Démonstration explicite

`ATTENTION_MOCK=true` active uniquement les trois paires de fixtures listées dans l’interface, identifiées comme sorties simulées par une bannière et `model_id=fixture-demo`. Les autres phrases sont refusées en mode mock. Ce mode serveur est désactivé par défaut et doit être désactivé pour toute vérification réelle du modèle. Il ne simule aucune qualité linguistique.

## Validation et production

```sh
npm run typecheck
npm run lint
npm test
npm run build
npm start
```

Les tests couvrent validation Unicode, erreurs, inversion, copie, annulation des résultats obsolètes, double soumission, enveloppe des erreurs proxy et délai. La mise en page s’adapte au mobile et à l’ordinateur ; saisie avec label, actions clavier, focus visible, messages annoncés et mouvements réduits sont prévus. Aucun texte de résultat n’est interprété comme HTML. Aucune police ou ressource distante n’est nécessaire.
