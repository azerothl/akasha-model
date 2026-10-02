# Path B hors ligne / embarquable (issue #62)

## Décision d’export

| Format | Décision | Motif |
|--------|----------|-------|
| Torch `.pt` local | **Retenu** pour l’embarqué v1 | Checkpoint démo `examples/gate/checkpoints/gate-tiny.pt` et futurs poids maison ; chargement via `akasha_model.offline` |
| ONNX / GGUF | **Pas d’export MASK maintenant** | Pas encore de checkpoint mix entraîné (T4 GPU/BERT bloqué) ; la tête multitask + collator byte n’a pas de graphe ONNX figé testé. Revoir après un checkpoint mix stable |

Cette décision est documentée pour l’acceptance « export **ou** décision de ne pas exporter ».

## Chargement sans réseau

```python
from akasha_model.offline import (
    ModelAbsentError,
    MODEL_ABSENT_STATUS,
    load_offline_torch_checkpoint,
    deny_network,
)

try:
    payload, meta = load_offline_torch_checkpoint("path/to/gate-tiny.pt")
except ModelAbsentError as exc:
    show_status(exc.status)  # message utilisateur, pas la traceback brute
```

`deny_network()` coupe les sockets le temps du `torch.load` et force
`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`. Les tests échouent si un accès
réseau est tenté pendant le chargement.

Tokenizer / encodeur : pour le tiny byte scorer, **aucun** tokenizer Hub — les
poids et le code d’embedding sont locaux. Un futur BERT offline devra embarquer
vocab + poids sur disque (jamais `from_pretrained` en ligne).

## Message d’absence

`MODEL_ABSENT_STATUS` (FR) / `MODEL_ABSENT_STATUS_EN` — texte produit, sans
chemin technique brut comme message principal (`ModelAbsentError.status`).

## Mesures (CPU, publiées telles quelles)

Relancer :

```sh
python scripts/bench_offline_path_b.py --checkpoint examples/gate/checkpoints/gate-tiny.pt
```

Le script écrit latence de chargement, RSS, et taille fichier dans
`reports/offline_path_b.json` (généré, non commité).
