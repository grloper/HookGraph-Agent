"""End-to-end tests for the compiled graph: convergence, parallel fan-out,
graceful degradation, the repair-memory stream, and the human-review gate."""

import pytest
from langgraph.types import Command

from hookgraph.graph import build_graph
from hookgraph.sample_data import load_sample_transcript
from hookgraph.state import SourceVideo, TranscriptSegment, initial_state

MAX_CLIP_SECONDS = 60.0
MIN_CLIP_SECONDS = 8.0


def run_graph(source, segments, *, max_attempts=4, review_gate=False, thread_id="test"):
    app = build_graph()
    config = {
        "configurable": {"thread_id": thread_id, "review_gate": review_gate},
        "recursion_limit": 60,
    }
    state = initial_state(source, segments, max_extraction_attempts=max_attempts)
    app.invoke(state, config=config)
    return app, config


def tiny_transcript() -> tuple[SourceVideo, list[TranscriptSegment]]:
    """Two segments — structurally impossible to cut three non-overlapping clips."""
    segments = [
        TranscriptSegment(segment_id=0, start=0.0, end=10.0, speaker="A",
                          text="Stop! This shocking collapse ruined everything overnight."),
        TranscriptSegment(segment_id=1, start=10.0, end=20.0, speaker="B",
                          text="Why did the dopamine prediction error wreck the streaks?"),
    ]
    source = SourceVideo(video_id="tiny-ep", title="Tiny Episode", duration_seconds=24.0)
    return source, segments


@pytest.fixture(scope="module")
def final_state():
    source, segments = load_sample_transcript()
    app, config = run_graph(source, segments, thread_id="happy")
    return app.get_state(config).values


class TestHappyPath:

    def test_exactly_three_qc_approved_packages(self, final_state):
        assert len(final_state["final_packages"]) == 3
        assert not final_state["pipeline_degraded"]
        assert all(not p.requires_human_review for p in final_state["final_packages"])

    def test_every_clip_respects_platform_duration_limits(self, final_state):
        for package in final_state["final_packages"]:
            assert MIN_CLIP_SECONDS <= package.hook.duration_seconds < MAX_CLIP_SECONDS

    def test_clips_never_share_source_footage(self, final_state):
        windows = sorted(
            (p.hook.start_seconds, p.hook.end_seconds)
            for p in final_state["final_packages"]
        )
        for (_, earlier_end), (later_start, _) in zip(windows, windows[1:]):
            assert later_start >= earlier_end - 0.011

    def test_artifacts_in_sync_with_final_hook_revisions(self, final_state):
        for package in final_state["final_packages"]:
            assert package.captions.hook_revision == package.hook.revision
            assert package.metadata.hook_revision == package.hook.revision
            platforms = {v.platform for v in package.metadata.variants}
            assert platforms == {"youtube_shorts", "tiktok", "instagram_reels"}

    def test_srt_documents_are_well_formed(self, final_state):
        for package in final_state["final_packages"]:
            srt = package.captions.srt
            assert srt.startswith("1\n")
            assert "-->" in srt
            first_cue = package.captions.cues[0]
            assert first_cue.start_seconds >= 0.0  # clip-relative time

    def test_corrective_loop_actually_fired_and_was_remembered(self, final_state):
        assert len(final_state["qc_reports"]) >= 2
        assert not final_state["qc_reports"][0].passed
        assert final_state["qc_reports"][-1].passed
        assert final_state["repair_memory"], "Showrunner should have recorded strategies"

    def test_parallel_workers_emitted_per_hook_events(self, final_state):
        worker_events = [
            event for event in final_state["pipeline_events"]
            if event.startswith("[Scriptwriter:")
        ]
        # 3 hooks on the fresh pass + one per repaired hook on retries.
        assert len(worker_events) >= 3

    def test_render_manifests_are_executable_recipes(self, final_state):
        for package in final_state["final_packages"]:
            assert package.render.ffmpeg_command.startswith("ffmpeg ")
            assert package.render.srt_filename.endswith(".srt")
            assert package.render.clip_out > package.render.clip_in


class TestGracefulDegradation:
    def test_impossible_transcript_terminates_flagged_not_looping(self):
        source, segments = tiny_transcript()
        app, config = run_graph(source, segments, thread_id="degraded")
        final = app.get_state(config).values
        assert final["pipeline_degraded"]
        assert all(p.requires_human_review for p in final["final_packages"])
        # The Showrunner degraded via ladder exhaustion, not by burning the
        # whole retry budget on a structurally impossible repair.
        assert final["extraction_attempts"] <= final["max_extraction_attempts"]

    def test_strategy_ladder_escalates_across_attempts(self):
        source, segments = tiny_transcript()
        app, config = run_graph(source, segments, thread_id="ladder")
        memory = app.get_state(config).values["repair_memory"]
        assert "package:exactly_three_hooks" in memory


class TestHumanReviewGate:
    def test_degraded_run_pauses_then_resumes_with_note(self):
        source, segments = tiny_transcript()
        app = build_graph()
        config = {
            "configurable": {"thread_id": "review", "review_gate": True},
            "recursion_limit": 60,
        }
        state = initial_state(source, segments, max_extraction_attempts=2)
        app.invoke(state, config=config)

        snapshot = app.get_state(config)
        assert snapshot.next, "graph should be paused on the review interrupt"
        interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        assert interrupts and "reason" in interrupts[0].value

        app.invoke(Command(resume="shipped by test reviewer"), config=config)
        final = app.get_state(config).values
        assert final["reviewer_note"] == "shipped by test reviewer"
        assert final["final_packages"], "resume must complete compilation"
        assert not app.get_state(config).next


class TestDurableExecution:
    def test_every_super_step_is_checkpointed(self):
        source, segments = load_sample_transcript()
        app, config = run_graph(source, segments, thread_id="checkpoints")
        history = list(app.get_state_history(config))
        assert len(history) >= 6  # kickoff, extract, fan-out, QC, repair, compile...
