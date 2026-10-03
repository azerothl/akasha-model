# Path B hors ligne / embarquable (issue #62)

## Décision d’export

| Format | Décision | Motif |
|--------|----------|-------|
| Torch `.pt` local | **Retenu** pour l’embarqué v1 | Checkpoint démo `examples/gate/checkpoints/gate-tiny.pt` et futurs poids maison ; chargement via `akasha_model.offline` |
| ONNX / GGUF | **Toujours bloqué sur T8 (#82)** | Recette MASK mix : [`mix-mask-train.md`](mix-mask-train.md). Script d’export [`scripts/export_mix_mask_onnx.py`](../scripts/export_mix_mask_onnx.py) refuse `gate-tiny.pt` et tout checkpoint qui n’a pas passé le juge T4. Le graphe tiny MASK **peut** s’exporter si le paquet optionnel `onnx` est installé ; ce n’est pas l’acceptance T10. GGUF : hors sujet (pas un LLM llama.cpp). Servir en `.pt` local via `akasha_model.offline` jusqu’à un mix MASK publiable. |

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
