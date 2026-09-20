# Akasha OS v2 — dataset de décisions JevLike

## Résumé

Le dataset a été généré depuis un checkout local d’Akasha OS, et non depuis
une liste d’actions inventée. Le générateur extrait les littéraux d’intents,
les noms d’outils des manifests de modules et les routes utilisées par les
handlers Rust. L’inventaire complet contient 316 identifiants extraits ; 160
actions réparties sur les préfixes `agent`, `canvas`, `mem`, `model`, `notes`,
`tasks`, `device`, `media`, `module`, `cap`, `schedule`, `web`, etc. sont
utilisées dans le dataset pour garder les menus de décision de taille réaliste.

Chaque ligne conserve le schéma JevLike : `context`, `options`, `label`.
`__abstain__` est une option normale et est la cible pour les demandes
ambiguës, contradictoires, non autorisées, dangereuses ou hors distribution.

## Volumétrie et qualité

| Split | Exemples | Familles | Abstention |
|---|---:|---:|---:|
| train | 21 000 | 219 | 26,97 % |
| validation | 4 500 | 216 | 24,27 % |
| test | 4 500 | 205 | 23,47 % |
| stress | 3 000 |  — | 34,07 % |

Les contextes sont générés avec plusieurs formulations, surfaces, ressources,
niveaux de confiance, états (`ready`, `busy`, `offline`, `missing`, `stale`,
`degraded`, `failed`) et signaux opérationnels. Il n’y a aucun doublon exact ou
normalisé entre train, validation et test, et aucune famille train/test
partagée. Les familles sont réparties avant la génération des lignes.

## Couverture et limites

Les scénarios couvrent les conversations, agents, mémoire, notes, tâches,
skills, outils réseau, modèles, média, Canvas, périphériques, USB, filesystem,
capabilities, modules, santé runtime, diagnostics, mises à jour, MCP, harness
de code et feedback.

L’inventaire est une extraction statique : les descriptions, préconditions,
permissions et effets de bord restent des annotations prudentes générées à
partir du fichier source. Ils doivent être relus par un annotateur Akasha OS
avant un déploiement de sécurité. Les identifiants extraits des tests ou de
routes UI peuvent également être valides comme intents mais ne sont pas tous
des actions de production exécutables.

## Résultats de validation

Les modèles ont été entraînés deux epochs sur CPU avec `context_tokens=256`,
`option_tokens=32`, largeur/rang 32. Le modèle calibration-aware utilise une
perte Brier avec poids `0.1`. L’ensemble moyenne les probabilités de la
baseline et du modèle calibration-aware.

| Modèle / split | Accuracy | Macro-F1 | NLL | Brier | ECE | Couverture | Accuracy conditionnelle |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline / test | 0.546 | 0.326 | 1.148 | 0.554 | 0.059 | 0.765 | 0.406 |
| calibration-aware / test | 0.603 | 0.390 | 1.048 | 0.504 | 0.022 | 0.765 | 0.482 |
| ensemble / test | 0.650 | 0.437 | 0.989 | 0.473 | 0.101 | 0.765 | 0.542 |
| baseline / stress | 0.621 | 0.394 | 0.963 | 0.465 | 0.038 | 0.659 | 0.426 |
| calibration-aware / stress | 0.663 | 0.462 | 0.887 | 0.430 | 0.035 | 0.659 | 0.489 |
| ensemble / stress | 0.710 | 0.536 | 0.833 | 0.401 | 0.099 | 0.659 | 0.560 |

Le taux d’erreurs dangereuses mesuré par l’audit est nul sur cette génération,
mais ce résultat ne constitue pas une garantie de sécurité : les situations
réelles et les effets de bord doivent être annotés à partir de traces Akasha
OS et testés avec revue humaine.

## Artefacts

- `data/akasha_os_v2/action_inventory.json` : inventaire extrait du code.
- `data/akasha_os_v2/{train,validation,test,stress}.jsonl` : données.
- `data/akasha_os_v2/audit.json` : audit automatique.
- `scripts/generate_akasha_dataset.py` : génération reproductible.
- `scripts/audit_akasha_dataset.py` : validation des splits et du schéma.
- `scripts/evaluate_akasha_models.py` : métriques individuelles et ensemble.
- `reports/akasha_model_metrics.json` : métriques détaillées.

## Recommandation

Utiliser le modèle comme routeur rapide avec abstention obligatoire dès que la
probabilité est basse, que l’ensemble n’est pas d’accord ou qu’une capability
est inconnue. La prochaine itération devrait remplacer les annotations
synthétiques par des décisions humaines et des traces d’autorisation réelles,
en conservant les familles et empreintes de contexte pour éviter les fuites.
