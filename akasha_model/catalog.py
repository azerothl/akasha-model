"""Optional phase-3 catalog constraints for Path A.

Layering (outer → inner):

1. **CatalogPolicy** (this module) — which tools may be proposed on a surface
   (allowlist / denylist / required placement tags).
2. **ToolSpec** — per-tool schema, capabilities, confirmation, irreversible.
3. **OS ``check_permissions``** — final ACL / capability grant on the host.

No execution here. Outcomes suggestions (#8) never auto-write these rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .tool_calling import ToolCallPlan, ToolSpec


@dataclass(frozen=True)
class CatalogPolicy:
    """Optional constraints applied before planner thresholds.

    - ``allowlist``: if set, proposal tool must be in the set.
    - ``denylist``: proposal tool must not be in the set.
    - ``required_placement``: every listed tag must appear on the tool's
      ``placement_tags`` (e.g. ``offline``, ``preview``).
    """

    allowlist: frozenset[str] | None = None
    denylist: frozenset[str] = field(default_factory=frozenset)
    required_placement: frozenset[str] = field(default_factory=frozenset)


def check_catalog_policy(
    tools: Mapping[str, ToolSpec],
    tool_name: str,
    policy: CatalogPolicy | None,
) -> ToolCallPlan | None:
    """Return a blocked plan when policy rejects the proposal, else ``None``."""
    if policy is None:
        return None
    if policy.allowlist is not None and tool_name not in policy.allowlist:
        return ToolCallPlan(
            "blocked",
            tool_name,
            None,
            f"catalog_allowlist: tool not permitted: {tool_name}",
        )
    if tool_name in policy.denylist:
        return ToolCallPlan(
            "blocked",
            tool_name,
            None,
            f"catalog_denylist: tool denied: {tool_name}",
        )
    if policy.required_placement:
        spec = tools.get(tool_name)
        tags = frozenset(getattr(spec, "placement_tags", ()) or ()) if spec else frozenset()
        missing = sorted(policy.required_placement - tags)
        if missing:
            return ToolCallPlan(
                "blocked",
                tool_name,
                None,
                "catalog_placement: missing tag(s): " + ", ".join(missing),
            )
    return None


__all__ = ["CatalogPolicy", "check_catalog_policy"]
