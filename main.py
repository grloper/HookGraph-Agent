"""HookGraph entry point.

Compiles the LangGraph multi-agent ecosystem and runs a full production pass:
a long-form transcript goes in, three QC-approved vertical clip packages come
out. Runs completely offline by default — the analytical engines are
deterministic, so no API keys are required.

Usage::

    python main.py                            # run on the bundled demo episode
    python main.py --transcript episode.json  # run on your own transcript
    python main.py --output-dir dist          # choose the export directory
    python main.py --llm                      # Claude-powered titling (needs ANTHROPIC_API_KEY)
    python main.py --review                   # pause for a human note if QC degrades

Custom transcript JSON shape::

    {
      "source_video": {"video_id": "...", "title": "...", "duration_seconds": 900.0},
      "segments": [
        {"segment_id": 0, "start": 0.0, "end": 8.2, "speaker": "Host", "text": "..."},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from langgraph.types import Command

from hookgraph.engines import build_creative_engine
from hookgraph.graph import build_graph
from hookgraph.sample_data import load_sample_transcript
from hookgraph.state import (
    ClipPackage,
    SourceVideo,
    TranscriptSegment,
    HookGraphState,
    initial_state,
)

DIVIDER = "=" * 78


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hookgraph",
        description="Repurpose a long-form video transcript into 3 vertical clip packages.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Path to a transcript JSON file (defaults to the bundled demo episode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to export SRT files and package JSON into (default: ./output).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="QC retry budget for the corrective loop before degrading (default: 4).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Use the Claude creative engine for hook titling (requires the "
            "'anthropic' package and ANTHROPIC_API_KEY; falls back to the "
            "deterministic engine on any failure). Also enabled by HOOKGRAPH_LLM=1."
        ),
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help=(
            "Enable the human-review gate: if QC cannot fully converge, pause "
            "on a durable interrupt and ask for a reviewer note before shipping "
            "the degraded batch."
        ),
    )
    parser.add_argument(
        "--thread-id",
        default="hookgraph-demo-run",
        help="Checkpointer thread id for this run (default: hookgraph-demo-run).",
    )
    return parser.parse_args(argv)


def load_transcript(path: Path) -> tuple[SourceVideo, list[TranscriptSegment]]:
    """Load and validate an external transcript JSON document."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = SourceVideo.model_validate(payload["source_video"])
    segments = [TranscriptSegment.model_validate(item) for item in payload["segments"]]
    if not segments:
        raise ValueError(f"{path} contains no transcript segments")
    return source, segments


def print_final_report(state: HookGraphState) -> None:
    """Human-readable summary of the compiled clip packages and QC audit trail."""
    print(f"\n{DIVIDER}\nFINAL CONTENT PACKAGES\n{DIVIDER}")
    for package in state["final_packages"]:
        hook = package.hook
        review_flag = "  [NEEDS HUMAN REVIEW]" if package.requires_human_review else ""
        print(
            f"\n#{hook.rank}  {hook.hook_title}{review_flag}\n"
            f"    id/revision : {hook.hook_id} (rev {hook.revision})\n"
            f"    source cut  : {hook.start_seconds:.1f}s -> {hook.end_seconds:.1f}s "
            f"({hook.duration_seconds:.1f}s, {hook.peak_type.replace('_', ' ')})\n"
            f"    virality    : {hook.virality_score}/100\n"
            f"    rationale   : {hook.virality_justification}\n"
            f"    opening     : {hook.opening_line}\n"
            f"    captions    : {len(package.captions.cues)} cues "
            f"(SRT ready, {len(package.captions.srt)} chars)"
        )
        for variant in package.metadata.variants:
            tags = " ".join(variant.hashtags[:5])
            print(f"    {variant.platform:<16}: {variant.title}  |  {tags}")

    print(f"\n{DIVIDER}\nQUALITY-CONTROL AUDIT TRAIL\n{DIVIDER}")
    for report in state["qc_reports"]:
        print(f"\n  {report.summary}")
        for violation in report.violations:
            print(f"    - [{violation.severity}] {violation.rule}: {violation.message}")

    if state["repair_memory"]:
        print(f"\n{DIVIDER}\nSHOWRUNNER REPAIR MEMORY\n{DIVIDER}")
        for key, strategies in sorted(state["repair_memory"].items()):
            print(f"  {key}: {' -> '.join(strategies)}")

    print(f"\n{DIVIDER}\nPIPELINE EVENT LOG\n{DIVIDER}")
    for event in state["pipeline_events"]:
        print(f"  * {event}")


def export_packages(
    state: HookGraphState, source: SourceVideo, output_dir: Path
) -> list[Path]:
    """Write SRT files plus a machine-readable package manifest to disk."""
    run_dir = output_dir / source.video_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for package in state["final_packages"]:
        srt_path = run_dir / package.render.srt_filename
        srt_path.write_text(package.captions.srt, encoding="utf-8")
        written.append(srt_path)

    manifest = {
        "source_video": source.model_dump(),
        "pipeline_degraded": state["pipeline_degraded"],
        "qc_attempts": state["extraction_attempts"],
        "reviewer_note": state.get("reviewer_note", ""),
        "repair_memory": state.get("repair_memory", {}),
        "packages": [package.model_dump() for package in state["final_packages"]],
        "qc_reports": [report.model_dump() for report in state["qc_reports"]],
        "pipeline_events": state["pipeline_events"],
    }
    manifest_path = run_dir / "packages.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written


def _stream_run(app, run_input, config) -> None:
    """Stream one graph execution segment, printing node-attributed events."""
    for chunk in app.stream(run_input, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if not isinstance(update, dict):
                continue  # e.g. the __interrupt__ marker chunk
            for event in update.get("pipeline_events", []):
                print(f"  [{node_name}] {event.split('] ', 1)[-1]}")


def _resolve_interrupts(app, config) -> bool:
    """If the run paused on a human-review interrupt, collect a note and resume.

    Returns True when an interrupt was handled and the caller should stream
    the continuation; False when the run has genuinely finished.
    """
    snapshot = app.get_state(config)
    if not snapshot.next:
        return False
    pending = [
        intr for task in snapshot.tasks for intr in getattr(task, "interrupts", ())
    ]
    if not pending:
        return False

    payload = pending[0].value
    print(f"\n{DIVIDER}\nHUMAN REVIEW GATE — pipeline paused (checkpoint persisted)\n{DIVIDER}")
    print(f"  Reason: {payload.get('reason', 'unknown')}")
    for line in payload.get("open_violations", []):
        print(f"    - {line}")
    print(f"\n  {payload.get('question', '')}")
    if sys.stdin.isatty():
        note = input("  Reviewer note (enter to approve): ").strip()
    else:
        note = "auto-approved (non-interactive run)"
        print(f"  Non-interactive session -> resuming with note: '{note}'")
    print()
    _stream_run(app, Command(resume=note or "approved without comment"), config)
    return True


def run(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.transcript is not None:
        source, segments = load_transcript(args.transcript)
    else:
        source, segments = load_sample_transcript()

    use_llm = args.llm or os.environ.get("HOOKGRAPH_LLM", "") in ("1", "true", "anthropic")
    engine, engine_note = build_creative_engine(use_llm)

    print(f"{DIVIDER}\nHookGraph — supervisor-orchestrated short-form repurposing ecosystem\n{DIVIDER}")
    print(
        f"Source : '{source.title}' ({source.duration_seconds:.0f}s, "
        f"{len(segments)} transcript segments)"
    )
    print(f"Engine : {engine_note}")
    print(f"Review : {'human gate armed (--review)' if args.review else 'autonomous (degraded batches ship flagged)'}\n")

    app = build_graph()
    config = {
        "configurable": {
            "thread_id": args.thread_id,
            "creative_engine": engine,
            "review_gate": args.review,
        },
        "recursion_limit": 60,
    }
    state = initial_state(source, segments, max_extraction_attempts=args.max_attempts)

    print("Executing graph (streaming node updates):\n")
    _stream_run(app, state, config)
    while _resolve_interrupts(app, config):
        pass

    snapshot = app.get_state(config)
    final_state: HookGraphState = snapshot.values

    print_final_report(final_state)
    written = export_packages(final_state, source, args.output_dir)

    print(f"\n{DIVIDER}\nEXPORTED ARTIFACTS\n{DIVIDER}")
    for path in written:
        print(f"  -> {path}")

    checkpoints = sum(1 for _ in app.get_state_history(config))
    print(
        f"\nDurable execution: {checkpoints} checkpoints recorded for thread "
        f"'{args.thread_id}' (every super-step, including each QC retry and any "
        f"review pause, is resumable)."
    )

    if final_state["pipeline_degraded"]:
        print("\nRESULT: pipeline completed DEGRADED — packages flagged for human review.")
        return 1
    print("\nRESULT: pipeline completed with all rubric rules satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
