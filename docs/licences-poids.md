# Licences et provenance des poids

Dossier pour les checkpoints livrés ou cités par **akasha-model**. À faire relire
par Gabriel avant toute livraison commerciale / module payant (issue #60).

Statut : **brouillon technique** — notices à confirmer sur les textes de licence
amont (fichiers LICENSE des dépôts / cartes HF), pas seulement les métadonnées.

## Encodeurs de base (Path B / MASK)

| Encodeur | Rôle | Licence (métadonnée / source lue) | Notice à reproduire | Confirmation |
|----------|------|-----------------------------------|---------------------|--------------|
| Tiny transformer maison (`MultiQuestionTinyScorer` / MASK tiny) | Démo CPU, tests, `examples/gate/checkpoints/gate-tiny.pt` | **MIT** (ce dépôt, [`LICENSE`](../LICENSE)) | Copyright Minimal Labs ; MIT | OK (dépôt) |
| [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) | Encodeur RLCD / décision optionnel | **Apache-2.0** (métadonnée HF) | NOTICE Apache-2.0 du checkpoint amont | **À confirmer** sur le texte LICENSE du repo amont |
| [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) | Encodeur optionnel | **Apache-2.0** (métadonnée HF) | NOTICE Apache-2.0 du checkpoint amont | **À confirmer** sur le texte LICENSE du repo amont |

Les poids téléchargés depuis le Hub **gardent leurs propres termes** (README
du package). Ce dépôt ne redistribue pas BERT / ModernBERT dans le wheel PyPI.

## Checkpoints de démonstration (dans le checkout Git)

| Artefact | Contenu | Licence des poids | Données d’entraînement | Statut licence données |
|----------|---------|-------------------|------------------------|------------------------|
| `examples/gate/checkpoints/gate-tiny.pt` | Tiny multitask scorer (démo gate) | MIT (dérivé du code maison) | Données **synthétiques** générées localement (scénarios scriptés gate / multitask) | Synthétique — **pas** un corpus tiers ; pas de claim « données libres redistribuables » au-delà du générateur maison |
| `examples/checkpoints/*.pt` (Doom / chess / joint) | Scorer vision + table d’options | MIT (code) + poids entraînés ici | Trajectoires VizDoom / parties chess générées localement ; Stockfish pour labels chess | **Environnements et moteurs** : respecter les licences VizDoom / Stockfish / FreeDoom ; **ne pas** redistribuer captures ou films avec audio copyrighté (`AGENTS.md`) |
| Jeu mix synthétique (`scripts/generate_mix_dataset.py`) | N/A (pas de poids) | — | Descripteurs + labels **Option A** (règles écrites), `CC0-1.0` déclaré par ligne | Synthétique ; mesure = accord aux règles ≠ qualité de mix ([`docs/mix-dataset.md`](mix-dataset.md)) |
| Mix MASK T8 (`runs/mix-mask-tiny.pt`, hors git) | Tiny MASK (`DecisionModel`) | MIT (code maison) si entraîné sur Option A CC0 | Option A synthétique, recette [`docs/mix-mask-train.md`](mix-mask-train.md) | **Pas publiable** tant que T4 ne bat pas le *majority prior*. `gate-tiny.pt` **n’est pas** un checkpoint mix. |
| Mix Option B (labels humains) | N/A tant qu’aucun corpus n’est importé | — | Schéma + import [`docs/contracts/mix-option-b.md`](contracts/mix-option-b.md) ; **pas** de labels inventés dans le dépôt | Renseigner `source` / `label_licence` **avant** tout poids ; fixtures `format_only` interdits à l’entraînement |

## Jeux de données tiers (non commités)

Les archives usuelles de mixage (MedleyDB, MUSDB18, Cambridge-MT, MoisesDB,
ENST-Drums, etc.) portent souvent des **restrictions non commerciales ou
éducatives**. Elles **ne sont pas** utilisées par le générateur Option A actuel
et **ne doivent pas** être commités (cohérent avec `AGENTS.md` / `.gitignore`
`data/`).

Si un futur Option B (annotations humaines) ou un corpus tiers est branché :

1. Ajouter une ligne ici : encodeur / données / licence / notice.
2. Renseigner `source` + `licence` par ligne JSONL.
3. **Faire relire Gabriel** avant livraison.

## Notices pratiques pour un hôte (Tauri / hors ligne)

- Embarquer uniquement des poids dont la licence autorise la redistribution
  prévue (MIT maison, ou Apache-2.0 amont **avec** NOTICE).
- Ne pas bundler de checkpoint BERT/ModernBERT sans recopier la NOTICE Apache.
- Afficher un message d’état clair si le modèle est absent (exigence produit
  song-maker) — sans clé technique brute.

## Revue

| Rôle | Action | Date |
|------|--------|------|
| Agent | Brouillon initial (#60) | 2026-10-02 |
| Gabriel | Relecture avant livraison | **pending** |
