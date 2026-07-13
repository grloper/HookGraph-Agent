"""Unit tests for the state reducers — the merge semantics the whole
corrective loop and the parallel Scriptwriter fan-out depend on."""

from hookgraph.state import (
    HookCandidate,
    ScoreBreakdown,
    merge_hooks,
    merge_repair_memory,
)


def make_hook(hook_id: str, rank: int, revision: int = 0, start: float = 0.0) -> HookCandidate:
    return HookCandidate(
        hook_id=hook_id,
        rank=rank,
        hook_title=f"Title for {hook_id}",
        virality_score=80.0,
        virality_justification="A substantive justification of the retention composite.",
        peak_type="emotional_spike",
        start_seconds=start,
        end_seconds=start + 30.0,
        segment_ids=[0, 1],
        opening_line="Stop scrolling!",
        score_breakdown=ScoreBreakdown(
            semantic_density=0.5, emotional_intensity=0.5, topic_novelty=0.5, opening_punch=0.5
        ),
        revision=revision,
    )


class TestUpsertReducer:
    def test_partial_update_never_clobbers_siblings(self):
        existing = [make_hook("hook-1", 1), make_hook("hook-2", 2), make_hook("hook-3", 3)]
        repaired = [make_hook("hook-2", 2, revision=1)]
        merged = merge_hooks(existing, repaired)
        assert len(merged) == 3
        by_id = {hook.hook_id: hook for hook in merged}
        assert by_id["hook-2"].revision == 1
        assert by_id["hook-1"].revision == 0 and by_id["hook-3"].revision == 0

    def test_deterministic_ordering(self):
        merged = merge_hooks([], [make_hook("hook-3", 3), make_hook("hook-1", 1)])
        assert [hook.hook_id for hook in merged] == ["hook-1", "hook-3"]


class TestRepairMemoryReducer:
    def test_strategies_accumulate_in_order(self):
        memory = merge_repair_memory({}, {"hook-1:duration_under_60s": ["trim_weak_edges"]})
        memory = merge_repair_memory(memory, {"hook-1:duration_under_60s": ["tight_rewindow"]})
        assert memory["hook-1:duration_under_60s"] == ["trim_weak_edges", "tight_rewindow"]

    def test_duplicate_strategies_never_re_recorded(self):
        memory = merge_repair_memory(
            {"k": ["a", "b"]}, {"k": ["a"], "other": ["x"]}
        )
        assert memory["k"] == ["a", "b"]
        assert memory["other"] == ["x"]

    def test_merge_does_not_mutate_inputs(self):
        existing = {"k": ["a"]}
        merge_repair_memory(existing, {"k": ["b"]})
        assert existing == {"k": ["a"]}
