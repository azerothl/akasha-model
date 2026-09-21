import torch
from torch.nn import functional as F

from akasha_model.data import ByteCollator, synthetic_example
from akasha_model.model import TinyScorer
from akasha_model.calibration import fit_temperature
from akasha_model.train import brier_loss, training_loss
from akasha_model.data import validate
from akasha_model.decision import ByteTokenizer, DecisionModel, MaskCollator, TinyEncoder
from akasha_model.primitives import (
    ChoiceQuestion, NoulQuestion, OptionSpec, ScoreLevel, ScoreQuestion,
    choice_result, noul_result, score_result,
)
from akasha_model.multitask import (
    MultiQuestionCollator, MultiQuestionTinyScorer, validate_multi, multitask_loss,
)
from akasha_model.multitask_eval import _binary_auc, _ece
from akasha_model.rewards import proper_reward
from akasha_model.rlcd import grpo_loss
from akasha_model.sequence import QTYPES, build_sequence
from akasha_model.tool_calling import ToolCallPlanner, ToolSpec
from akasha_model.typed_decisions import convert_typed_row


def test_tiny_scorer_learns_and_normalises():
    torch.manual_seed(5)
    examples = [synthetic_example(1000 + index) for index in range(32)]
    batch = ByteCollator(128, 24)(examples)
    model = TinyScorer(width=32, rank=32, context_tokens=128)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = float(F.cross_entropy(model(batch), batch["labels"]).detach())
    for _ in range(30):
        loss = F.cross_entropy(model(batch), batch["labels"])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    logits = model(batch)
    final = float(F.cross_entropy(logits, batch["labels"]).detach())
    assert final < initial * 0.6
    assert torch.allclose(logits.softmax(-1).sum(-1), torch.ones(len(examples)), atol=1e-6)


def test_temperature_calibration_is_positive_and_reduces_nll():
    logits = torch.tensor([[3.0, 0.0], [2.0, 1.0], [0.2, 0.0], [0.1, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    before = float(F.cross_entropy(logits, labels))
    temperature = fit_temperature(logits, labels)
    after = float(F.cross_entropy(logits / temperature, labels))
    assert temperature > 0
    assert after < before


def test_calibration_weight_adds_brier_without_changing_baseline():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    baseline, calibration = training_loss(logits, labels, 0.0)
    weighted, _ = training_loss(logits, labels, 0.5)
    assert torch.allclose(calibration, brier_loss(logits, labels))
    assert torch.allclose(baseline, F.cross_entropy(logits, labels))
    assert weighted > baseline


def test_rich_choice_rows_preserve_option_descriptions():
    example = validate({
        "context": "The user asks for a webpage.",
        "options": [
            {"name": "tool.request", "description": "Use the network tool."},
            {"name": "skill.invoke", "description": "Use a local skill."},
        ],
        "label": 0,
    })
    assert example.option_texts()[0].startswith("tool.request")
    assert "network tool" in example.option_texts()[0]


def test_typed_primitives_validate_and_compute_results():
    choice = ChoiceQuestion(
        "route", "Choose the safest route.",
        (OptionSpec("allow", "The capability is granted."),
         OptionSpec("deny", "The capability is absent.")),
    )
    result = choice_result(choice, {"allow": 0.8, "deny": 0.2}, threshold=0.9)
    assert result.selected is None
    assert result.abstained

    score = ScoreQuestion(
        "risk", "Rate risk.",
        (ScoreLevel("low"), ScoreLevel("high")),
    )
    score_value = score_result(score, {0: 0.25, 1: 0.75})
    assert score_value.score == 0.75
    assert score_value.legend == {0: "low", 1: "high"}

    noul = noul_result(NoulQuestion("authorized", "Is this authorized?"), 0.5)
    assert noul.probability == 0.5


def test_multi_question_model_has_three_typed_heads():
    example = validate_multi({
        "context": "The user asks to access a protected device.",
        "questions": [
            {"id": "route", "type": "choice", "instructions": "Choose the route.",
             "options": [{"name": "deny", "description": "Block the operation."},
                         {"name": "allow", "description": "Permit the operation."}], "label": 0},
            {"id": "risk", "type": "score", "instructions": "Rate risk.",
             "levels": ["low", "high"], "label": 1},
            {"id": "authorized", "type": "noul", "instructions": "Is it authorized?", "label": 0},
        ],
    })
    batch = MultiQuestionCollator(64, 32, 24)([example])
    model = MultiQuestionTinyScorer(width=16, rank=16, context_tokens=64,
                                    question_tokens=32, max_score_levels=2)
    outputs = model(batch)
    losses = multitask_loss(outputs, batch, score_levels=2)
    assert set(outputs) == {"choice", "score", "noul"}
    assert all(torch.isfinite(value) for value in losses.values())


def test_multi_question_model_encodes_state_once():
    example = validate_multi({
        "context": {"request": "read USB", "offline": True},
        "questions": [
            {"id": "route", "type": "choice", "instructions": "Choose.",
             "options": ["deny", "allow"], "label": 0},
            {"id": "authorized", "type": "noul", "instructions": "Is it authorized?", "label": 0},
        ],
    })
    batch = MultiQuestionCollator(64, 32, 24)([example])
    model = MultiQuestionTinyScorer(width=16, rank=16, context_tokens=64,
                                    question_tokens=32, max_score_levels=2)
    calls = []
    handle = model.context_position.register_forward_hook(lambda *_: calls.append(1))
    model(batch)
    handle.remove()
    assert len(calls) == 1


def test_multi_question_local_encoder_supports_order_sensitive_pooling():
    example = validate_multi({
        "context": {"request": "resume a model", "mode": "offline"},
        "questions": [
            {"id": "route", "type": "choice", "instructions": "Choose.",
             "options": ["resume", "stop"], "label": 0},
            {"id": "risk", "type": "score", "instructions": "Rate risk.",
             "levels": ["low", "high"], "label": 0},
        ],
    })
    batch = MultiQuestionCollator(64, 32, 24)([example])
    model = MultiQuestionTinyScorer(width=16, rank=16, context_tokens=64,
                                    question_tokens=32, max_score_levels=2,
                                    encoder="local")
    outputs = model(batch)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_multi_question_lexical_encoder_supports_unseen_option_matching():
    example = validate_multi({
        "context": {"request": "resume a model", "mode": "offline"},
        "questions": [
            {"id": "route", "type": "choice", "instructions": "Choose.",
             "options": [
                 {"name": "resume", "description": "resume a model"},
                 {"name": "stop", "description": "stop a model"},
             ], "label": 0},
        ],
    })
    batch = MultiQuestionCollator(64, 32, 48)([example])
    model = MultiQuestionTinyScorer(width=16, rank=16, context_tokens=64,
                                    question_tokens=32, max_score_levels=2,
                                    encoder="lexical")
    outputs = model(batch)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_multitask_eval_helpers_are_well_defined():
    assert _binary_auc([0.1, 0.9, 0.8, 0.2], [0, 1, 1, 0]) == 1.0
    assert abs(_ece([0.9, 0.1], [1.0, 0.0]) - 0.1) < 1e-8


def _tool_choice(selected: str = "fs.read"):
    question = ChoiceQuestion(
        "route", "Choose a tool.",
        (OptionSpec("fs.read"), OptionSpec("fs.delete")),
    )
    return choice_result(
        question, {"fs.read": 0.9 if selected == "fs.read" else 0.1,
                    "fs.delete": 0.1 if selected == "fs.read" else 0.9},
    )


def test_tool_call_planner_returns_a_non_executing_ready_plan():
    spec = ToolSpec(
        "fs.read", "Read a file.",
        parameters={"type": "object", "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                    "additionalProperties": False},
        required_capability="workspace_access",
    )
    plan = ToolCallPlanner().plan(
        _tool_choice(), {spec.name: spec}, arguments={"path": "notes.txt"},
        nouls={"authorized": 0.95, "sufficient_context": 0.90,
               "capability_present": 0.92, "confirmation_needed": 0.05},
    )
    assert plan.status == "ready"
    assert plan.executable
    assert plan.arguments == {"path": "notes.txt"}


def test_tool_call_planner_blocks_missing_capability_and_confirmation():
    spec = ToolSpec("fs.delete", irreversible=True, required_capability="workspace_access")
    plan = ToolCallPlanner().plan(
        _tool_choice("fs.delete"), {spec.name: spec},
        nouls={"authorized": 0.95, "sufficient_context": 0.95,
               "capability_present": 0.20},
        confirmation_given=True,
    )
    assert plan.status == "blocked"
    assert "capability" in plan.reason

    plan = ToolCallPlanner().plan(
        _tool_choice("fs.delete"), {spec.name: spec},
        nouls={"authorized": 0.95, "sufficient_context": 0.95,
               "capability_present": 0.95},
    )
    assert plan.status == "blocked"
    assert "confirmation" in plan.reason


def test_tool_call_planner_abstains_on_weak_choice():
    spec = ToolSpec("fs.read")
    choice = choice_result(
        ChoiceQuestion("route", "Choose.", (OptionSpec("fs.read"), OptionSpec("fs.delete"))),
        {"fs.read": 0.51, "fs.delete": 0.49},
    )
    plan = ToolCallPlanner(min_choice_probability=0.60).plan(
        choice, {spec.name: spec},
        nouls={"authorized": 1.0, "sufficient_context": 1.0},
    )
    assert plan.status == "abstain"


def test_mask_sequence_places_a_marker_per_option():
    tokenizer = ByteTokenizer()
    question = {
        "t": "choice", "ins": "Choose the route.",
        "crit": {"deny": "Block the operation.", "allow": "Permit the operation."},
    }
    ids, markers = build_sequence(
        tokenizer, "offline protected device", question,
        max_len=128, head_max_len=64, option_max_len=16,
    )
    assert len(markers) == 2
    assert all(ids[position] == tokenizer.mask_token_id for position in markers)


def test_proper_reward_prefers_the_true_distribution():
    mask = torch.ones(1, 2, dtype=torch.bool)
    target = torch.tensor([[1.0, 0.0]])
    qtype = torch.tensor([QTYPES["choice"]])
    better = proper_reward(torch.tensor([[0.9, 0.1]]), target, qtype, mask)
    worse = proper_reward(torch.tensor([[0.1, 0.9]]), target, qtype, mask)
    assert float(better) > float(worse)


def test_mask_rlcd_learns_on_a_tiny_encoder():
    example = validate_multi({
        "context": "Permit the operation on this protected device.",
        "questions": [
            {"id": "route", "type": "choice", "instructions": "Choose the route.",
             "options": [{"name": "deny", "description": "Block the operation."},
                         {"name": "allow", "description": "Permit the operation."}],
             "label": 1},
            {"id": "authorized", "type": "noul",
             "instructions": "Is the operation authorized?", "label": 1},
        ],
    })
    model = DecisionModel(
        TinyEncoder(hidden_size=32, layers=1, heads=4, max_len=128, dropout=0.0),
        head_layers=1, dropout=0.0,
    )
    batch = MaskCollator(ByteTokenizer(), 128, 64, 16, group_size=2)([example, example])
    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)

    def current_loss():
        return grpo_loss(
            model(
                batch["input_ids"], batch["attention_mask"],
                batch["marker_pos"], batch["marker_mask"], batch["qtype"],
            ),
            batch, 0.5, 1.0, 1.0,
        )

    initial = float(current_loss()["total"].detach())
    losses = None
    for _ in range(25):
        losses = current_loss()
        optimiser.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimiser.step()
    assert losses is not None
    assert torch.isfinite(losses["total"])
    assert float(losses["total"].detach()) < initial


def test_type_balance_does_not_let_noul_drown_choice():
    mask = torch.ones(10, 2, dtype=torch.bool)
    target = torch.tensor([[1.0, 0.0]] * 10)
    qtype = torch.tensor([0] + [2] * 9)
    reported = torch.tensor([[0.1, 0.9]] + [[0.9, 0.1]] * 9)
    logits = torch.log(reported)
    batch = {
        "marker_mask": mask,
        "target": target,
        "qtype": qtype,
        "group_size": torch.tensor(1),
    }
    meaned = grpo_loss(logits, batch, 0.5, 1.0, 0.0, type_balance=False)
    balanced = grpo_loss(logits, batch, 0.5, 1.0, 0.0, type_balance=True)
    assert float(balanced["supervised"]) > float(meaned["supervised"])
    assert float(balanced["reward_choice"]) < float(balanced["reward_noul"])


def test_typed_decisions_row_keeps_soft_targets_and_official_keys():
    row = convert_typed_row({
        "id": "cs_000",
        "workflow": "customer_service",
        "state": {"message": "Please refund the duplicate charge."},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": {"billing": "invoices", "technical": "bugs"},
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is this urgent?",
            },
            "severity": {
                "type": "score",
                "instructions": "How severe?",
                "criteria": ["low", "high"],
            },
        },
        "gold": {
            "department": {
                "label": "billing",
                "probabilities": {"billing": 0.8, "technical": 0.2},
            },
            "urgent": {"noul": 0.9, "label": "true"},
            "severity": {"label": 1, "probabilities": {"0": 0.1, "1": 0.9}},
        },
    })
    parsed = validate_multi(row)
    assert parsed.questions[0]["label"] == 0
    assert parsed.questions[0]["target"][0] == 0.8
    assert parsed.questions[1]["target"][1] == 0.9
    assert parsed.questions[2]["label"] == 1
    assert row["workflow"] == "customer_service"
