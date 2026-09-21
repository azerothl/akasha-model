# Akasha OS — dataset multitâche

## Provenance

Le générateur inspecte le checkout local d’Akasha OS et enregistre le commit
`d74e11caf8415327dbf91c659a1403836f798300`. L’inventaire statique contient
317 identifiants de source ; 160 actions sont retenues pour garder les menus
Choice bornés. Les labels sont produits par une politique déterministe locale,
sans modèle de langage et sans télémétrie.

Commande de génération :

```powershell
.venv\Scripts\python.exe scripts\generate_akasha_multitask_dataset.py `
  --akasha-root <chemin-vers-akasha-os> `
  --output data\akasha_os_multi `
  --seed 20260918 `
  --total 30000
```

## Volumétrie

| Split | Lignes | Familles |
|---|---:|---:|
| train | 24 000 | 896 |
| validation | 3 000 | 192 |
| test | 3 000 | 192 |
| stress | 3 000 | séparé |

Chaque ligne contient une `Choice`, une `Score` et neuf questions `Noul`.
Cela représente 33 000 questions Choice, 33 000 questions Score et 297 000
questions Noul. Les questions sont des objets JSON structurés compatibles avec
`validate_multi` et `MultiQuestionDataset`.

## Politique de décision

La Choice route vers une action extraite du code, `other` ou
`insufficient_context`. Une capability absente/refusée/expirée route vers
`capability.check` lorsque cet identifiant est disponible, sinon vers
`insufficient_context`. Les états offline incompatibles avec une opération
réseau, les contradictions, les injections et les demandes inconnues ne sont
pas exécutés.

Le Score utilise quatre niveaux sémantiques indépendants de l’autorisation :
opération locale, opération réversible, opération sensible, puis opération
critique/destructive. Les Noul séparent l’autorisation, la capability, la
confirmation, les ressources protégées, le réseau, l’irréversibilité,
l’injection, la suffisance du contexte et la compatibilité offline.

## Audit de fuite et cohérence

- lignes exclues par `validate_multi` : 0 ;
- collision exacte ou normalisée de state entre les splits : 0 ;
- doublon d’instance de question entre les splits : 0 ;
- famille partagée train/test : 0 ;
- stress séparé et annoté par `stress_type`, `source_row_id` et
  `expected_behavior` ;
- labels Choice répartis sur les positions du menu ;
- labels Score répartis sur les quatre niveaux ;
- labels Noul : 99 066 vrais et 197 934 faux, conformément à `audit.json`.

Les templates Choice sont réutilisables par conception. L’audit détecte les
instances complètes state+questions, pas les définitions stables de questions.

## Stress

`stress.jsonl` couvre `ambiguous`, `irrelevant_noise`, `prompt_injection`,
`missing_context`, `conflicting_state`, `permission_boundary` et
`unknown_action`. Il ne doit jamais être utilisé pour l’entraînement.

## Validation technique

La validation exécutée est :

```powershell
.venv\Scripts\python.exe scripts\generate_akasha_multitask_dataset.py ...
.venv\Scripts\python.exe -m akasha_model.multitask_train `
  data\akasha_os_multi\train.jsonl `
  --validation data\akasha_os_multi\validation.jsonl `
  --output runs\akasha_os_multi_smoke.pt `
  --context-tokens 256 --question-tokens 96 --option-tokens 48 `
  --epochs 1 --batch-size 256 --width 16 --rank 16 --device cpu
```

Le smoke training a produit `runs/akasha_os_multi_smoke.pt` avec une perte de
validation totale de `3.5881`. Le modèle a produit les trois têtes `choice`,
`score` et `noul`. Les tests du dépôt passent également : `7 passed`.

## Limites

L’extraction statique peut contenir des identifiants utilisés dans des tests,
des routes UI ou des chemins de diagnostic qui ne sont pas tous des actions de
production. L’inventaire conserve les fichiers sources afin de permettre une
revue humaine. Les préconditions et effets de bord restent des annotations
conservatrices ; ce dataset ne doit pas remplacer le contrôle d’autorisation
réel d’Akasha OS. Les cas réellement observés en production doivent être
ajoutés après annotation humaine, en conservant les groupes sémantiques pour
préserver l’absence de fuite.

Artefacts principaux :

- `data/akasha_os_multi/train.jsonl`
- `data/akasha_os_multi/validation.jsonl`
- `data/akasha_os_multi/test.jsonl`
- `data/akasha_os_multi/stress.jsonl`
- `data/akasha_os_multi/audit.json`
- `data/akasha_os_multi/action_inventory.json`
- `scripts/generate_akasha_multitask_dataset.py`
