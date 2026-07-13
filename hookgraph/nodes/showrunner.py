"""Showrunner node — the hierarchical supervisor that runs the newsroom.

Every routing decision in the graph flows through this node. Instead of a
static conditional edge, the Showrunner returns LangGraph ``Command`` objects,
which lets it combine *control flow* (where the pipeline goes next) with
*state writes* (what the workers should do when they get there) in a single
atomic super-step.

Its three responsibilities:

1. **Kickoff** — dispatch the HookExtractor on a fresh transcript.
2. **Adaptive repair planning** — when QualityControl rejects the batch, pick
   the next repair strategy for every ``(hook, rule)`` failure from an
   escalation ladder, consulting the ``repair_memory`` stream so a strategy
   that already failed is never prescribed twice. This is what makes the
   corrective loop *monotonic*: each retry attacks the problem differently.
3. **Graceful degradation + human gate** — if the retry budget is spent or
   every ladder is exhausted, release the batch to compilation flagged for
   human review. With the review gate enabled (``--review``), execution
   pauses on a durable ``interrupt()`` so an operator can attach a note
   before the degraded batch ships.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from ..state import QCViolation, HookGraphState

# Ordered escalation ladders: the Showrunner walks each ladder left to right,
# one rung per retry, and never repeats a rung that already failed.
STRATEGY_LADDERS: dict[str, tuple[str, ...]] = {
    "exactly_three_hooks": ("tight_reextract",),
    "duration_under_60s": ("trim_weak_edges", "tight_rewindow", "reseed_peak"),
    "punchy_opening_line": ("reanchor_start", "widen_anchor_search", "reseed_peak"),
    "non_overlapping_times": (
        "shrink_from_collision",
        "shift_after_collision",
        "reseed_peak",
    ),
    "valid_timestamps": ("resnap_boundaries", "reseed_peak"),
    "metadata_completeness": ("refresh_artifacts",),
}


def _plan_repairs(
    blockers: list[QCViolation], memory: dict[str, list[str]]
) -> tuple[dict[str, str], list[str]]:
    """Choose the next untried strategy per failure; report exhausted ladders."""
    directives: dict[str, str] = {}
    exhausted: list[str] = []
    for violation in blockers:
        key = f"{violation.hook_id}:{violation.rule}"
        if key in directives:
            continue
        ladder = STRATEGY_LADDERS.get(violation.rule, ("reseed_peak",))
        tried = memory.get(key, [])
        next_strategy = next((rung for rung in ladder if rung not in tried), None)
        if next_strategy is None:
            exhausted.append(key)
        else:
            directives[key] = next_strategy
    return directives, exhausted


def showrunner_node(
    state: HookGraphState, config: RunnableConfig
) -> Command[Literal["hook_extractor", "package_compiler"]]:
    """LangGraph supervisor: kickoff, adaptive repair planning, or release."""
    reports = state["qc_reports"]

    # --- Kickoff: nothing evaluated yet, send the extractor in fresh. -------
    if not reports:
        return Command(
            goto="hook_extractor",
            update={
                "pipeline_events": [
                    "[Showrunner] Kickoff: dispatching HookExtractor for a fresh "
                    f"top-3 extraction (retry budget {state['max_extraction_attempts']})."
                ]
            },
        )

    last_report = reports[-1]

    # --- Rubric passed: release the batch to compilation. -------------------
    if last_report.passed:
        return Command(
            goto="package_compiler",
            update={
                "pipeline_events": [
                    f"[Showrunner] Rubric PASSED on attempt {last_report.attempt} — "
                    "releasing the batch to PackageCompiler."
                ]
            },
        )

    # --- Rubric failed: plan the next corrective pass or degrade. -----------
    blockers = state["active_violations"]
    directives, exhausted = _plan_repairs(blockers, state["repair_memory"])
    budget_spent = state["extraction_attempts"] >= state["max_extraction_attempts"]

    if budget_spent or exhausted or not directives:
        if budget_spent:
            reason = (
                f"retry budget spent ({state['extraction_attempts']}/"
                f"{state['max_extraction_attempts']} attempts)"
            )
        elif exhausted:
            reason = (
                "every repair strategy exhausted for: " + ", ".join(sorted(exhausted))
            )
        else:
            reason = "no applicable repair strategy for the reported violations"

        events = [
            f"[Showrunner] Degrading gracefully — {reason}. Batch will ship "
            "flagged for human review instead of looping forever."
        ]

        reviewer_note = state.get("reviewer_note", "")
        if config and config.get("configurable", {}).get("review_gate"):
            # Durable pause: the checkpoint persists here, and the run resumes
            # (on this thread id) only when an operator supplies a note.
            note = interrupt(
                {
                    "reason": reason,
                    "open_violations": [
                        f"[{violation.hook_id}] {violation.rule}: {violation.message}"
                        for violation in blockers
                    ],
                    "question": (
                        "QC could not fully converge. Add a reviewer note to "
                        "attach to the degraded packages, then resume."
                    ),
                }
            )
            reviewer_note = str(note).strip() or "approved without comment"
            events.append(
                f"[Showrunner] Human review gate: operator note recorded — "
                f"'{reviewer_note}'"
            )

        return Command(
            goto="package_compiler",
            update={"pipeline_events": events, "reviewer_note": reviewer_note},
        )

    # Record the prescribed strategies into memory *now*, so the next planning
    # pass escalates even if this repair round fails to clear the rule.
    memory_update = {key: [strategy] for key, strategy in directives.items()}
    plan_text = "; ".join(
        f"{key} -> {strategy}" for key, strategy in sorted(directives.items())
    )
    return Command(
        goto="hook_extractor",
        update={
            "repair_directives": directives,
            "repair_memory": memory_update,
            "pipeline_events": [
                f"[Showrunner] Rubric FAILED on attempt {last_report.attempt} with "
                f"{len(blockers)} blocker(s). Repair plan: {plan_text}."
            ],
        },
    )
