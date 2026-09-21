# Akasha OS — dataset multitâche v2

## Provenance et reproductibilité

- Dépôt inspecté : checkout local d’Akasha OS.
- Commit source : `d74e11caf8415327dbf91c659a1403836f798300`.
- Seed : `20260918`.
- Inventaire statique : 317 identifiants ; 160 actions utilisées ; 16 actions réservées OOD.
- Les labels sont produits par une politique déterministe locale. Aucun modèle de
  langage, secret ou télémétrie n’intervient.
- Les noms d’action et les références source sont conservés dans l’inventaire et
  les sidecars, jamais dans le `context` d’entrée du modèle.

Commande exacte :

```powershell
.venv\Scripts\python.exe scripts\generate_akasha_multitask_dataset_v2.py `
  --akasha-root E:\akasha-os `
  --output data\akasha_os_multi_v2 `
  --seed 20260918 `
  --total 30000 `
  --ood-total 1000
.venv\Scripts\python.exe scripts\audit_akasha_multitask_dataset_v2.py `
  --input data\akasha_os_multi_v2 `
  --akasha-root E:\akasha-os
```

## Volumétrie et organisation

| Split | Lignes | Familles |
|---|---:|---:|
| train | 24 000 | 2 048 |
| validation | 3 000 | 260 |
| test | 3 000 | 252 |
| stress | 3 000 | 160 |
| ood | 1 000 | 48 |

Les groupes sont construits par combinaison action/scénario. Les familles sont
disjointes entre les splits ; les actions OOD sont réservées hors des splits
ordinaires. Chaque ligne normale contient `route`, `operation_risk` et les neuf
questions Noul pertinentes : `authorized`, `capability_present`,
`confirmation_needed`, `protected_resource`, `requires_network`, `irreversible`,
`prompt_injection`, `sufficient_context` et `offline_compatible`.

## Distributions

Le dataset contient 34 000 questions Choice, 34 000 Score et 306 000 Noul.

| Question | Distribution des labels |
|---|---|
| `route` | positions 0–9, de 1 956 à 3 577 |
| `operation_risk` | niveau 0: 15 039 ; 1: 7 338 ; 2: 7 430 ; 3: 4 193 |
| Noul global | vrai: 123 525 ; faux: 182 475 |

Distributions Noul par identifiant :

| ID | Vrai | Faux |
|---|---:|---:|
| authorized | 15 568 | 18 432 |
| capability_present | 29 650 | 4 350 |
| confirmation_needed | 12 839 | 21 161 |
| protected_resource | 4 744 | 29 256 |
| requires_network | 1 233 | 32 767 |
| irreversible | 4 193 | 29 807 |
| prompt_injection | 2 345 | 31 655 |
| sufficient_context | 19 455 | 14 545 |
| offline_compatible | 33 498 | 502 |

## Audit anti-fuite

- violations anti-fuite : 0 ;
- lignes exclues par `validate_multi` : 0 ;
- noms d’action dans les contexts : 0 ;
- champs interdits (`target_action`, `oracle_action`, `variant`, `label`, etc.) : 0 ;
- références source dans les contexts : 0 ;
- collisions exactes ou normalisées entre tous les splits : 0 ;
- doublons d’instances de questions entre splits : 0 ;
- familles partagées entre splits comparés : 0 ;
- cas ambigus ou nécessitant une abstention : 7 767 et 14 545 routes fallback ;
- `stress` et `ood` ne sont jamais utilisés pour l’entraînement.

Les oracle sont uniquement dans `*.metadata.jsonl`, notamment `oracle_action`,
`source_refs`, `variant`, `stress_type` et `scenario_id`.

## Stress et OOD

Chaque type de stress contient 300 lignes : `ambiguous`, `irrelevant_noise`,
`prompt_injection`, `missing_context`, `conflicting_state`,
`permission_boundary`, `unknown_action`, `offline_network`,
`expired_capability` et `unsigned_install`.

Le bruit non pertinent est placé dans des champs neutres et réordonnés. Les
injections sont placées dans une note externe, jamais dans un indicateur explicite.
Les 1 000 lignes OOD utilisent des actions réservées et des familles distinctes.

## Longueurs et risque de troncature

Mesure UTF-8 correspondant au texte sérialisé par `MultiQuestionCollator` :

| Élément | Min | Moyenne | Max | Limite |
|---|---:|---:|---:|---:|
| context | 502 | 587,30 | 787 | 768 |
| question | 88 | 192,45 | 315 | 512 |
| option/level | 43 | 190,71 | 362 | 384 |

73 contexts dépassent 768 octets et peuvent donc être tronqués avec la
configuration demandée. Les questions et options ne dépassent pas leurs limites.
Le risque est enregistré dans `audit.json` sous `truncation`.

## Validation technique

Scripts reproductibles :

- `scripts/generate_akasha_multitask_dataset_v2.py`
- `scripts/audit_akasha_multitask_dataset_v2.py`

Artefacts :

- `data/akasha_os_multi_v2/{train,validation,test,stress,ood}.jsonl` ;
- `data/akasha_os_multi_v2/{train,validation,test,stress,ood}.metadata.jsonl` ;
- `data/akasha_os_multi_v2/source_manifest.json` ;
- `data/akasha_os_multi_v2/action_inventory.json` ;
- `data/akasha_os_multi_v2/audit.json`.

La validation attendue avec Akasha Model est :

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m py_compile `
  scripts\generate_akasha_multitask_dataset_v2.py `
  scripts\audit_akasha_multitask_dataset_v2.py
```

Un smoke pass avec `MultiQuestionDataset`, `MultiQuestionCollator(768, 512,
384, 10)` et `MultiQuestionTinyScorer(64, 64, 768, 512, 10)` doit produire les
trois sorties `choice`, `score` et `noul`.

## Limites connues

L’inventaire est extrait statiquement des littéraux d’identifiants et manifests
Akasha OS ; certains identifiants peuvent encore provenir de tests ou de routes
UI. Les préconditions et effets de bord sont des annotations déterministes et
conservatrices, pas une autorisation runtime. Une revue humaine des actions
réservées à la production reste recommandée avant déploiement.
