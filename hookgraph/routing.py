"""Routing rules — the map-reduce dispatch edge of the HookGraph graph.

Supervisor routing lives in the Showrunner node (Command-based); this module
holds the one remaining conditional edge: the fan-out that turns "some hooks
have stale artifacts" into N parallel Scriptwriter workers via the LangGraph
``Send`` API. Kept separate from node handlers so dispatch policy can evolve
without touching agent logic.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Send

from .state import HookGraphState


def dispatch_scriptwriters(
    state: HookGraphState,
) -> list[Send] | Literal["quality_control"]:
    """Fan out one Scriptwriter worker per hook whose artifacts are stale.

    Staleness is (hook_id, revision) based: a hook the HookExtractor just
    repaired carries a bumped revision, so exactly the repaired hooks get
    fresh workers while untouched hooks keep their existing caption tracks
    and metadata. If everything is already in sync, skip the Scriptwriter
    layer entirely and go straight to QualityControl.
    """
    current_revisions = {
        track.hook_id: track.hook_revision for track in state["caption_tracks"]
    }
    stale = [
        hook
        for hook in state["hooks"]
        if current_revisions.get(hook.hook_id) != hook.revision
    ]
    if not stale:
        return "quality_control"
    return [
        Send(
            "scriptwriter",
            {
                "hook": hook,
                "transcript": state["transcript"],
                "source_video": state["source_video"],
            },
        )
        for hook in stale
    ]
