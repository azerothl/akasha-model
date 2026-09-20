# Jevlike

Train a small model that chooses among a changing list of text options.

A Jev-like model takes a piece of text and a list of `N` text options. It returns one probability for each option. It does this in one pass instead of writing an answer word by word. [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is TypeSafe's commercial model for this kind of task. TypeSafe has not published its design. This repository is an independent starter model with the same input and output shape.

## Demo

The same option-attention head can score controller buttons from image patches. [This ten-second film](docs/jevre-demo-10s-bgm.mp4) joins two selected five-second windows: live `deadly_corridor` combat on the seven Doom buttons, then a chess controller walking to and playing moves with five keys. The diagram shows the tensors used for each decision. The Doom window came from the supplied joint checkpoint, which averaged 0.60 kills and -97.50 reward across its ten recorded episodes. The chess window came from the stronger chess-only checkpoint, which scored 4 wins, 46 draws and 0 losses in 50 sampled games against a random mover, but 0 wins, 2 draws and 48 losses against Stockfish level 0. The windows were selected for activity and are not typical-play or competence claims.

<video src="docs/jevre-demo-10s-bgm.mp4" controls width="960"></video>

Install the game extras and record a fresh 640 by 480 Doom trace from the released joint checkpoint:

```sh
uv pip install -e '.[games]'
python examples/doom/play.py examples/checkpoints/joint-imitation.pt --episodes 10 --game-seconds 35.3 --device cpu --capture-resolution 640x480 --output runs/doom.mp4 --trace runs/doom-trace.json
```

Render the trace in the same visual layout. This writes a silent film because the author-owned soundtrack source is not part of the repository.

```sh
(cd examples/film && npm install && npx playwright install chromium)
examples/film/make-film.sh runs/doom-trace.json runs/doom-film.mp4 10
```

The release includes the [Doom example](examples/doom/README.md), the [chess example](examples/chess/README.md), the single-game checkpoints and the shared 12-option checkpoint. Both games import the visual scorer from `jevlike.vision`; there is no second model copy in either example.

## Architecture

Each option becomes a query vector, which is a short list of numbers representing its text. The query assigns attention weights to the context tokens. Those weights make one context vector for that option. A shared dot product turns each option and context pair into one score. A softmax, which converts scores into probabilities that sum to one, runs across the options.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
  <img src="docs/architecture.svg" alt="Each option queries the context, receives an attended context vector, and becomes one probability.">
</picture>

The default encoder learns byte embeddings from scratch. An encoder is the part that turns text into vectors. The optional Hugging Face path uses a frozen pretrained encoder, whose existing weights stay fixed while the small scorer learns.

## Data format

Use one JSON object per line:

```json
{"context":"The customer needs a refund.","options":["refund","sales","technical support"],"label":0}
```

`label` is the zero-based index of the correct option. Each row may have a different number of options, with a minimum of two.

For Jev-like semantic criteria, an option can also be an object. The old
string format remains valid:

```json
{"context":"The user asks for a webpage.","options":[{"name":"tool.request","description":"Use the network tool for an external fetch."},{"name":"skill.invoke","description":"Use an installed local workflow."}],"label":0}
```

The package exposes three typed decision contracts in `jevlike.primitives`:

```python
from jevlike import ChoiceQuestion, NoulQuestion, OptionSpec, ScoreLevel, ScoreQuestion

choice = ChoiceQuestion(
    "route", "Choose the safest route.",
    (OptionSpec("tool.request", "Use the network tool."),
     OptionSpec("skill.invoke", "Use a local skill.")),
)
score = ScoreQuestion(
    "risk", "Rate the risk.",
    (ScoreLevel("low"), ScoreLevel("high")),
)
noul = NoulQuestion("authorized", "Is the operation authorized?")
```

These contracts are the first compatibility layer for the multi-primitive
model. They validate bounded options, ordered score levels, probability
distributions and explicit abstention thresholds. Authorization, side effects
and workflow decisions remain application code rather than model output.

### Tool calling: planification, pas exécution

Le modèle multitâche ne génère pas directement un appel de fonction JSON et
ne possède aucun exécuteur d’outil. `Choice` sélectionne un outil dans un
catalogue borné, `Score` estime un niveau de risque et `Noul` fournit des
signaux binaires comme l’autorisation, la présence d’une capability ou la
suffisance du contexte. L’application doit ensuite valider les arguments,
les permissions et la confirmation humaine avant tout effet de bord.

`jevlike.tool_calling.ToolCallPlanner` fournit cette frontière déterministe :
il retourne un plan `ready`, `abstain` ou `blocked`, valide un sous-ensemble du
schéma JSON des paramètres et ne lance jamais l’outil. Akasha-OS doit fournir
son propre exécuteur après un plan `ready` et refaire ses contrôles finaux.

```python
from jevlike import ToolCallPlanner, ToolSpec

spec = ToolSpec(
    "fs.read", "Lire un fichier",
    parameters={"type": "object", "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False},
    required_capability="workspace_access",
)
plan = ToolCallPlanner().plan(
    choice_result, {spec.name: spec}, arguments={"path": "notes.txt"},
    nouls={"authorized": authorized_result,
           "capability_present": capability_result,
           "sufficient_context": context_result},
)
if plan.status == "ready":
    akasha_executor.execute(plan.tool_name, plan.arguments)
```

### Multi-question training

The shared-context model accepts several atomic questions in one JSONL row.
The encoder is shared, while `Choice`, `Score` and `Noul` have independent
heads:

```json
{"context":{"request":"access a protected device","offline":true},"questions":[{"id":"route","type":"choice","instructions":"Choose the route.","options":[{"name":"deny","description":"Block the operation."},{"name":"allow","description":"Permit the operation."}],"label":0},{"id":"risk","type":"score","instructions":"Rate the risk.","levels":["low","high"],"label":1},{"id":"authorized","type":"noul","instructions":"Is it authorized?","criteria":{"true":"The user explicitly granted the capability.","false":"No explicit grant is present."},"label":0}]}
```

Train it with the separate multi-question command:

```sh
jevlike-multitask-train data/akasha_os_multi/train.jsonl \
  --validation data/akasha_os_multi/validation.jsonl \
  --output runs/akasha-os-multitask.pt
```

`label` is an option index for `Choice` and a level index for `Score`. Score
levels are ordered descriptions, numbered internally from 0; they are not
arbitrary numeric measurements. A `Noul` returns the probability that the
condition in its instructions is true, with values near 0.5 representing
uncertainty.

Evaluate the three heads separately on the held-out test and stress sets:

```powershell
jevlike-multitask-eval `
  runs\akasha_os_multi.pt `
  data\akasha_os_multi\test.jsonl `
  --stress data\akasha_os_multi\stress.jsonl `
  --output reports\akasha_os_multi_metrics.json
```

The report includes accuracy and calibration for `Choice`, ordinal error for
`Score`, AUROC/Brier/log loss for `Noul`, coverage-risk tables, and stress
results broken down by `stress_type`.

The multi-question defaults are sized for structured Akasha states:
`context_tokens=768`, `question_tokens=512` and `option_tokens=384`.
Lower values may silently truncate criteria and source-derived descriptions.
`Noul` uses `0` or `1`. Rows may contain any subset of the three question
types; absent heads are skipped for that batch.

## Quickstart

Run these commands from the repository root. They create local synthetic data, train on it, evaluate the saved model and score one new menu.

```sh
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'

jevlike-data synthetic --output data/synthetic
jevlike-train data/synthetic/train.jsonl \
  --validation data/synthetic/validation.jsonl \
  --output runs/synthetic.pt
jevlike-eval runs/synthetic.pt data/synthetic/test.jsonl
jevlike-predict runs/synthetic.pt \
  --context "Choose the exact badge amber badger. Badge: amber badger." \
  --option "azure crane" \
  --option "amber badger" \
   --option "gold heron"
```

At inference time, option descriptions can be supplied and the application can
request abstention below a confidence threshold:

```sh
jevlike-predict runs/synthetic.pt \
  --context "The user asks to fetch a public webpage." \
  --option tool.request \
  --option skill.invoke \
  --option-description "Use the network tool for an external fetch." \
  --option-description "Use an installed local workflow." \
  --min-confidence 0.75
```

The evaluation prints top-1 accuracy, which is the fraction of correct first choices. Top-3 accuracy is the fraction with the right answer among the three highest scores. It also reports negative log-likelihood (NLL), Brier score and expected calibration error (ECE). ECE compares confidence with observed accuracy. The command also prints a shuffled-context control, which pairs each menu with the wrong context. A useful model should beat that control.

## Calibrate a trained model

Use a separate calibration split that was not used to train or select the model. Temperature scaling changes the sharpness of the probabilities without changing the ranking of the options:

```sh
jevlike-calibrate runs/synthetic.pt data/synthetic/validation.jsonl \
  --output runs/synthetic-calibrated.pt
jevlike-eval runs/synthetic-calibrated.pt data/synthetic/test.jsonl
```

The fitted temperature is stored in the checkpoint and applied automatically by `jevlike-predict` and `jevlike-eval`. Compare NLL, Brier score and ECE on the held-out test set before deciding whether calibration helped. Calibration improves the reliability of probabilities; it does not necessarily improve top-1 accuracy.

Training can also include a calibration-aware objective. The default weight is zero, which keeps the original training behavior. A small positive weight adds the multiclass Brier score to the cross-entropy objective:

```sh
jevlike-train data/synthetic/train.jsonl \
  --validation data/synthetic/validation.jsonl \
  --output runs/synthetic-calibration-aware.pt \
  --calibration-weight 0.1
```

Tune this weight on a validation split. Do not assume that a lower calibration loss improves accuracy, and always compare the final held-out NLL, Brier score and ECE against the unmodified baseline.

## Leakage-resistant splits

When related rows share the same context, split by context rather than by row. The helper below combines JSONL files and keeps every identical context in exactly one split:

```sh
python scripts/split_by_context.py data/akasha_os data/akasha_os_grouped
```

Train and evaluate against `data/akasha_os_grouped`. Check that the reported context sets do not overlap before interpreting the test metrics.

To probe robustness without contaminating training, derive an evaluation-only stress set from the grouped test split:

```sh
python scripts/make_stress_eval.py \
  data/akasha_os_grouped/test.jsonl \
  data/akasha_os_grouped/stress.jsonl
jevlike-eval runs/akasha_os_grouped/baseline.pt \
  data/akasha_os_grouped/stress.jsonl
```

The stress set contains ambiguous-context and irrelevant-noise variants. Its labels inherit the original test labels, so treat it as a robustness probe rather than a replacement for a human-labelled benchmark.

## Ensemble abstention

For uncertainty-sensitive decisions, train several checkpoints with different seeds and evaluate their agreement:

```sh
jevlike-ensemble-eval \
  runs/akasha_os_ensemble/seed-11.pt \
  runs/akasha_os_ensemble/seed-23.pt \
  runs/akasha_os_ensemble/seed-37.pt \
  --data data/akasha_os_grouped/stress.jsonl
```

The ensemble averages probabilities and reports the fraction of examples on which every model agrees. A conservative application can abstain when the models disagree; `unanimous_accuracy` measures the accuracy of the remaining decisions and `unanimous_coverage` measures how often a decision is still returned.

Use the same policy at inference time:

```sh
jevlike-ensemble-predict \
  runs/akasha_os_ensemble/seed-11.pt \
  runs/akasha_os_ensemble/seed-23.pt \
  runs/akasha_os_ensemble/seed-37.pt \
  --context "User asks to fetch a public webpage." \
  --option tool.request \
  --option skill.invoke \
  --option capability.check
```

The command abstains when the ensemble disagrees or when its mean confidence is below the configured threshold.

## Use your own data

1. Export train, validation and test JSONL files in the format above.
2. Keep all options that the model will see at prediction time in each row.
3. Split related records together. For example, keep all records for one customer or one target page in one split. This prevents near-duplicates from leaking into the test set.
4. Run `jevlike-train` with your train and validation files.
5. Run `jevlike-eval` once on the held-out test file. Held-out means the file was never used for training or model selection.

The default byte encoder truncates context to 192 bytes and each option to 32 bytes. Raise `--context-tokens` or `--option-tokens` when your text needs more room. Training supports CPU, Apple MPS for a Mac GPU, and CUDA for an NVIDIA GPU through `--device`.

### Akasha-OS-style routing data

The repository can generate a deterministic, synthetic policy dataset inspired
by [azerothl/akasha-os](https://github.com/azerothl/akasha-os): the context
describes a task, surface, resource and trust state; the options are possible
OS-level actions; and `label` identifies the action selected by a small routing
policy. It covers sessions, memory, scheduled tasks, agents, skills, tools,
model packs and capability checks. It is a local synthetic benchmark, not a
dump of Akasha OS telemetry.

```sh
jevlike-data akasha-os --output data/akasha_os
jevlike-train data/akasha_os/train.jsonl \
  --validation data/akasha_os/validation.jsonl \
  --output runs/akasha-os.pt
jevlike-eval runs/akasha-os.pt data/akasha_os/test.jsonl
```

Les splits sont groupés par famille de scénario par défaut : une famille
(par exemple Canvas, mémoire ou périphérique) reste entièrement dans un seul
fichier. Pour un split plus fin, regroupe plutôt par famille et signal runtime
(GPU, permissions, réseau, audit, etc.) :

```sh
jevlike-data akasha-os --output data/akasha_os_context --split-by context
```

Cela évite qu'une même situation de contexte ou une variante quasi identique
se retrouve à la fois dans l'entraînement et dans l'évaluation.

Pour la version v2 dérivée du checkout source local, avec abstention et audit :

```powershell
.venv\Scripts\python.exe scripts\generate_akasha_dataset.py `
  --akasha-root <chemin-vers-akasha-os> `
  --output data\akasha_os_v2 `
  --seed 20260918 `
  --total 30000
.venv\Scripts\python.exe scripts\audit_akasha_dataset.py `
  --input data\akasha_os_v2
```

`stress.jsonl` est réservé à l'évaluation. Le rapport complet se trouve dans
`reports/akasha_dataset_report.md`.

### Akasha OS — dataset multitâche source-derived

Pour entraîner séparément les têtes `Choice`, `Score` et `Noul` sur des
scénarios dérivés des identifiants réellement présents dans le checkout local
d'Akasha OS :

```powershell
.venv\Scripts\python.exe scripts\generate_akasha_multitask_dataset.py `
  --akasha-root <chemin-vers-akasha-os> `
  --output data\akasha_os_multi `
  --seed 20260918 `
  --total 30000
.venv\Scripts\python.exe -m jevlike.multitask_train `
  data\akasha_os_multi\train.jsonl `
  --validation data\akasha_os_multi\validation.jsonl `
  --output runs\akasha_os_multi.pt
```

Cette version produit 24 000 lignes d'entraînement, 3 000 de validation,
3 000 de test et 3 000 lignes `stress` séparées. Le détail de la provenance,
des familles et de l'audit se trouve dans
`reports/akasha_multitask_dataset_report.md`.

## Use a frozen pretrained encoder

Install the optional dependency and name any compatible encoder from Hugging Face:

```sh
uv pip install -e '.[transformers]'
jevlike-train data/synthetic/train.jsonl \
  --validation data/synthetic/validation.jsonl \
  --output runs/qwen-head.pt \
  --encoder hf \
  --hf-model Qwen/Qwen2.5-0.5B \
  --rank 256 \
  --batch-size 8
```

The checkpoint stores the trained scorer head and the encoder name. It does not copy the frozen encoder weights. Loading the checkpoint therefore needs access to the same Hugging Face model.

`--rank` sets the width of the small scorer head. A wider head has more trainable weights and uses more memory.

## Wikispeedia example

[`scripts/get_wikispeedia.sh`](scripts/get_wikispeedia.sh) downloads the public SNAP archives and builds next-click JSONL files. The data stay outside this repository.

```sh
scripts/get_wikispeedia.sh
jevlike-train data/wikispeedia/jsonl/train.jsonl \
  --validation data/wikispeedia/jsonl/validation.jsonl \
  --output runs/wikispeedia.pt
```

Cite Robert West and Jure Leskovec, *Human Wayfinding in Information Networks*, WWW 2012. Review the source data terms on the [SNAP dataset page](https://snap.stanford.edu/data/wikispeedia.html).

## What to expect

In the experiments that led to this starter, the one-pass scorer reached about 98% accuracy on synthetic menus. On target-disjoint Wikispeedia next-click data, a frozen Qwen2.5-0.5B encoder plus the scorer reached 26%, against about 8% for shuffled and random-encoder controls. A small model trained from scratch on 40,000 clicks reached 29%. At eight options, one pass was about 100 times faster than a small decoder forced to write 400 tokens.

These numbers describe local experiments, not this quickstart run. We did not show equal quality with Jev or reproduce TypeSafe's private training method.

## Limitations

- This is a research starter, not a copy of Jev.
- Accuracy depends on data quality, split quality and the encoder.
- The byte encoder is cheap but weak on language meaning.
- The pretrained path may download a large model and needs more memory.
- One-pass scoring requires the complete option list before prediction.
- The speed comparison used a small local decoder rather than a large commercial model.

## Licence

Code is released under the [MIT License](LICENSE). Downloaded datasets and pretrained models keep their own terms.
