"""Safe tool-call planning for the Choice/Score/Noul scorer.

The multitask model is a bounded decision scorer, not a generative function
caller. This module is the deterministic boundary around it: it checks the
selected tool, authorization/capability/context signals and a shallow JSON
schema contract, then returns a plan. It deliberately has no executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .primitives import ChoiceResult, NoulResult, ScoreResult


PlanStatus = Literal["ready", "abstain", "blocked"]


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ToolSpec:
    """Description of one callable tool known by the host application.

    Prefer ``required_capabilities`` (zero or more tokens). The singular
    ``required_capability`` field remains as a backward-compatible alias for
    the first listed capability (or ``None`` when the tuple is empty).
    """

    name: str
    description: str = ""
    parameters: Mapping[str, Any] | None = None
    required_capability: str | None = None
    required_capabilities: tuple[str, ...] = ()
    requires_confirmation: bool = False
    irreversible: bool = False
    placement_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.name, "tool name")
        caps: list[str] = []
        if self.required_capabilities:
            for cap in self.required_capabilities:
                _non_empty(cap, "required_capabilities item")
                if cap not in caps:
                    caps.append(cap)
        if self.required_capability is not None:
            _non_empty(self.required_capability, "required_capability")
            if self.required_capability not in caps:
                caps.insert(0, self.required_capability)
        # Normalise both fields so hosts can read either shape.
        object.__setattr__(self, "required_capabilities", tuple(caps))
        object.__setattr__(
            self, "required_capability", caps[0] if caps else None,
        )
        for tag in self.placement_tags:
            _non_empty(tag, "placement_tags item")
        if self.parameters is not None:
            if not isinstance(self.parameters, Mapping):
                raise ValueError("tool parameters must be a JSON schema object")
            if self.parameters.get("type", "object") != "object":
                raise ValueError("tool parameters schema must describe an object")


@dataclass(frozen=True)
class ToolCallPlan:
    """A non-executing decision returned by :class:`ToolCallPlanner`."""

    status: PlanStatus
    tool_name: str | None
    arguments: dict[str, Any] | None
    reason: str
    choice_probability: float = 0.0
    choice_confidence: float = 0.0
    risk_score: float | None = None

    @property
    def executable(self) -> bool:
        """Whether the host may proceed to its own final execution checks."""
        return self.status == "ready"


def _probability(
    nouls: Mapping[str, float | NoulResult], question_id: str,
) -> float | None:
    value = nouls.get(question_id)
    if value is None:
        return None
    probability = value.probability if isinstance(value, NoulResult) else float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Noul probability for {question_id!r} must be in [0, 1]")
    return probability


def _matches_type(value: Any, expected: str) -> bool:
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "null": lambda item: item is None,
    }
    return expected not in checks or checks[expected](value)


def validate_tool_arguments(
    spec: ToolSpec, arguments: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Validate the common JSON-schema subset used by Akasha tool specs."""
    values = dict(arguments or {})
    schema = spec.parameters
    if schema is None:
        return True, ""
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return False, "invalid tool parameter schema"
    missing = [name for name in required if name not in values]
    if missing:
        return False, "missing required argument(s): " + ", ".join(map(str, missing))
    if schema.get("additionalProperties", True) is False:
        unknown = sorted(set(values) - set(properties))
        if unknown:
            return False, "unknown argument(s): " + ", ".join(unknown)
    for name, value in values.items():
        rule = properties.get(name)
        if not isinstance(rule, Mapping):
            continue
        expected = rule.get("type")
        if isinstance(expected, str) and not _matches_type(value, expected):
            return False, f"argument {name!r} must be {expected}"
        enum = rule.get("enum")
        if isinstance(enum, list) and value not in enum:
            return False, f"argument {name!r} is outside its enum"
    return True, ""


class ToolCallPlanner:
    """Turn typed model results into a safe, non-executing tool-call plan."""

    def __init__(
        self,
        *,
        min_choice_probability: float = 0.55,
        min_choice_confidence: float = 0.50,
        noul_threshold: float = 0.70,
        max_risk_score: float | None = None,
    ) -> None:
        if not 0.0 <= min_choice_probability <= 1.0:
            raise ValueError("min_choice_probability must be in [0, 1]")
        if not 0.0 <= min_choice_confidence <= 1.0:
            raise ValueError("min_choice_confidence must be in [0, 1]")
        if not 0.5 <= noul_threshold <= 1.0:
            raise ValueError("noul_threshold must be in [0.5, 1]")
        self.min_choice_probability = min_choice_probability
        self.min_choice_confidence = min_choice_confidence
        self.noul_threshold = noul_threshold
        self.max_risk_score = max_risk_score

    def plan(
        self,
        choice: ChoiceResult,
        tools: Mapping[str, ToolSpec],
        *,
        arguments: Mapping[str, Any] | None = None,
        nouls: Mapping[str, float | NoulResult] | None = None,
        score: ScoreResult | None = None,
        confirmation_given: bool = False,
    ) -> ToolCallPlan:
        """Return ``ready``, ``abstain`` or ``blocked`` without executing."""
        nouls = nouls or {}
        if choice.abstained or choice.selected is None:
            return ToolCallPlan(
                "abstain", None, None, "Choice abstained", 0.0,
                choice.confidence, score.score if score else None,
            )
        probability = float(choice.probabilities.get(choice.selected, 0.0))
        if (choice.confidence < self.min_choice_confidence or
                probability < self.min_choice_probability):
            return ToolCallPlan(
                "abstain", choice.selected, None,
                "Choice confidence or probability is below the execution threshold",
                probability, choice.confidence, score.score if score else None,
            )
        spec = tools.get(choice.selected)
        if spec is None:
            return ToolCallPlan(
                "blocked", choice.selected, None,
                "Selected tool is not in the host tool catalog",
                probability, choice.confidence, score.score if score else None,
            )
        if self.max_risk_score is not None and score is not None and score.score > self.max_risk_score:
            return ToolCallPlan(
                "blocked", spec.name, None,
                "Risk score exceeds the planner limit", probability,
                choice.confidence, score.score,
            )

        for question_id, message in (
            ("authorized", "authorization gate is not positive"),
            ("sufficient_context", "context sufficiency gate is not positive"),
        ):
            gate = _probability(nouls, question_id)
            if gate is None or gate < self.noul_threshold:
                return ToolCallPlan(
                    "blocked", spec.name, None, message, probability,
                    choice.confidence, score.score if score else None,
                )
        if spec.required_capabilities:
            # Hosts may AND all caps into ``capability_present``, and/or supply
            # per-capability nouls named ``capability:<token>``.
            missing: list[str] = []
            for cap in spec.required_capabilities:
                per_cap = _probability(nouls, f"capability:{cap}")
                if per_cap is not None:
                    if per_cap < self.noul_threshold:
                        missing.append(cap)
                    continue
                shared = _probability(nouls, "capability_present")
                if shared is None or shared < self.noul_threshold:
                    missing.append(cap)
            if missing:
                return ToolCallPlan(
                    "blocked", spec.name, None,
                    "required capability is not positive: " + ", ".join(missing),
                    probability, choice.confidence, score.score if score else None,
                )
        confirmation_needed = spec.requires_confirmation or spec.irreversible
        model_confirmation = _probability(nouls, "confirmation_needed")
        if model_confirmation is not None:
            confirmation_needed = confirmation_needed or model_confirmation >= self.noul_threshold
        if confirmation_needed and not confirmation_given:
            return ToolCallPlan(
                "blocked", spec.name, None,
                "explicit human confirmation is required", probability,
                choice.confidence, score.score if score else None,
            )
        valid, reason = validate_tool_arguments(spec, arguments)
        if not valid:
            return ToolCallPlan(
                "blocked", spec.name, None, reason, probability,
                choice.confidence, score.score if score else None,
            )
        return ToolCallPlan(
            "ready", spec.name, dict(arguments or {}), "all planner gates passed",
            probability, choice.confidence, score.score if score else None,
        )


__all__ = ["PlanStatus", "ToolCallPlan", "ToolCallPlanner", "ToolSpec", "validate_tool_arguments"]
