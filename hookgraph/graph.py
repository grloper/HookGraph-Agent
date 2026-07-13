"""Graph compilation block — wires state, nodes, and routing into a runnable app.

Topology (hierarchical supervisor + parallel map-reduce workers)::

                       START
                         |
                         v
                  +-------------+   Command(goto=...)
        +-------->| showrunner  |-------------------------+
        |         +-------------+                         |
        |                | kickoff / repair plan          | release /
        |                v                                | degrade (+ human
        |         +---------------+                       |  review interrupt)
        |         | hook_extractor|                       |
        |         +---------------+                       |
        |                | Send() fan-out (1 worker       |
        |                v         per stale hook)        v
        |     [scriptwriter] x N  --barrier-->   +------------------+
        |                |                       | package_compiler |--> END
        |                v                       +------------------+
        |         +-----------------+
        +---------| quality_control |
                  +-----------------+

The Showrunner owns every routing decision (Command-based supervision); the
only conditional edge left in the graph is the Scriptwriter fan-out, which
maps stale hooks onto parallel workers via the Send API and reduces their
writes through the upsert reducers declared in state.py.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    hook_extractor_node,
    package_compiler_node,
    quality_control_node,
    scriptwriter_node,
    showrunner_node,
)
from .routing import dispatch_scriptwriters
from .state import (
    CaptionCue,
    CaptionTrack,
    ClipPackage,
    HookCandidate,
    MetadataPackage,
    PlatformVariant,
    QCReport,
    QCViolation,
    RenderManifest,
    ScoreBreakdown,
    SourceVideo,
    TranscriptSegment,
    HookGraphState,
)

# Explicit serializer allowlist: every Pydantic model that can appear in a
# checkpoint is registered, so snapshots round-trip without trust-on-first-use
# deserialization warnings (and anything unexpected is loudly blocked).
_STATE_MODELS = (
    CaptionCue,
    CaptionTrack,
    ClipPackage,
    HookCandidate,
    MetadataPackage,
    PlatformVariant,
    QCReport,
    QCViolation,
    RenderManifest,
    ScoreBreakdown,
    SourceVideo,
    TranscriptSegment,
)


def _default_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_STATE_MODELS))


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Compile the HookGraph pipeline into an executable LangGraph app.

    A checkpointer is always attached (in-memory by default) so every
    super-step — including each QC retry and any human-review interrupt — is
    durably snapshotted and the run can be inspected, resumed, or time-traveled
    by thread id.
    """
    builder = StateGraph(HookGraphState)

    builder.add_node(
        "showrunner",
        showrunner_node,
        destinations=("hook_extractor", "package_compiler"),
    )
    builder.add_node("hook_extractor", hook_extractor_node)
    builder.add_node("scriptwriter", scriptwriter_node)
    builder.add_node("quality_control", quality_control_node)
    builder.add_node("package_compiler", package_compiler_node)

    builder.add_edge(START, "showrunner")
    # showrunner routes itself via Command(goto=...) — no static edge needed.
    builder.add_conditional_edges(
        "hook_extractor",
        dispatch_scriptwriters,
        ["scriptwriter", "quality_control"],
    )
    builder.add_edge("scriptwriter", "quality_control")
    builder.add_edge("quality_control", "showrunner")
    builder.add_edge("package_compiler", END)

    return builder.compile(checkpointer=checkpointer or _default_checkpointer())
