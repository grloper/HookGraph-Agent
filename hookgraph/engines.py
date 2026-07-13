"""Creative-engine layer — pluggable title/copy generation for the pipeline.

The analytical core of HookGraph (retention scoring, the QC rubric, timestamp
math) is deterministic on purpose: it is the safety rail that makes an
LLM-in-the-loop architecture convergent instead of chaotic. The *creative*
surface — hook titles today, descriptions/CTAs tomorrow — is where a language
model earns its keep, so it is isolated behind the small ``CreativeEngine``
protocol defined here.

Two engines ship:

- ``DeterministicCreativeEngine`` — template-based, dependency-free, always
  available. The default. Guarantees the whole pipeline runs offline with
  zero API keys.
- ``ClaudeCreativeEngine`` — calls the Anthropic API (official ``anthropic``
  SDK, structured outputs) to write scroll-stopping titles from the actual
  clip transcript. Selected with ``--llm`` / ``HOOKGRAPH_LLM=1``. Every call
  is guarded: on a missing SDK, missing key, refusal, or any API error the
  engine falls back to the deterministic template and records why, so an LLM
  outage can never take the pipeline down.

Engines travel to the nodes via ``RunnableConfig["configurable"]`` — they are
runtime collaborators, not state, so they are never checkpointed.
"""

from __future__ import annotations

import os
from typing import Optional, Protocol

from langchain_core.runnables import RunnableConfig

from .state import PeakType

MAX_TITLE_CHARS = 80

_TITLE_TEMPLATES: dict[PeakType, tuple[str, ...]] = {
    "emotional_spike": (
        "The {kw} Moment Nobody Saw Coming",
        "This {kw} Confession Changes Everything",
        "Why {kw} Nearly Broke Him",
    ),
    "semantic_density": (
        "The {kw} Framework Explained in 60 Seconds",
        "Steal This {kw} Playbook",
        "How {kw} Actually Works",
    ),
    "topic_transition": (
        "Wait — {kw} Is Not What You Think",
        "The {kw} Plot Twist",
        "From Zero to {kw}: The Pivot",
    ),
}


class CreativeEngine(Protocol):
    """The seam between the deterministic pipeline and generative creativity."""

    name: str

    def craft_title(
        self, excerpt: str, keyword: str, peak_type: PeakType, rank: int
    ) -> str:
        """Return an attention-grabbing clip title, at most 80 characters."""
        ...


class DeterministicCreativeEngine:
    """Template-based titling — zero dependencies, fully offline, always safe."""

    name = "deterministic"

    def craft_title(
        self, excerpt: str, keyword: str, peak_type: PeakType, rank: int
    ) -> str:
        templates = _TITLE_TEMPLATES[peak_type]
        keyword_title = (keyword or "this").replace("-", " ").title()
        return templates[(rank - 1) % len(templates)].format(kw=keyword_title)[
            :MAX_TITLE_CHARS
        ]


class ClaudeCreativeEngine:
    """Claude-powered titling with a deterministic safety net.

    Uses structured outputs so the title always arrives as validated JSON,
    and clamps to the 80-char schema limit so a long response can never
    trip the ``HookCandidate`` validator downstream.
    """

    name = "claude"

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # deferred: optional dependency
        from pydantic import BaseModel, Field

        class _TitleDraft(BaseModel):
            title: str = Field(
                description="A scroll-stopping short-form clip title, under 80 characters."
            )

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("HOOKGRAPH_MODEL", "claude-opus-4-8")
        self._schema = _TitleDraft
        self._fallback = DeterministicCreativeEngine()
        self.last_error: Optional[str] = None

    def craft_title(
        self, excerpt: str, keyword: str, peak_type: PeakType, rank: int
    ) -> str:
        prompt = (
            "You title vertical short-form clips (TikTok / Shorts / Reels). "
            "Write ONE title for the clip below. Hard requirements: under 80 "
            "characters, no hashtags, no emoji, no quotation marks around the "
            "title, and it must be specific to the excerpt (never generic "
            "clickbait).\n\n"
            f"Retention driver: {peak_type.replace('_', ' ')}\n"
            f"Primary keyword: {keyword}\n"
            f"Clip transcript excerpt:\n{excerpt[:1200]}"
        )
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
                output_format=self._schema,
            )
            if response.stop_reason == "refusal":
                raise RuntimeError("model declined the request (stop_reason=refusal)")
            title = response.parsed_output.title.strip().strip('"')
            if not title:
                raise ValueError("model returned an empty title")
            self.last_error = None
            return title[:MAX_TITLE_CHARS]
        except Exception as error:  # noqa: BLE001 — any failure degrades gracefully
            self.last_error = f"{type(error).__name__}: {error}"
            return self._fallback.craft_title(excerpt, keyword, peak_type, rank)


def build_creative_engine(use_llm: bool) -> tuple[CreativeEngine, str]:
    """Select an engine and explain the choice (for the pipeline event log)."""
    if not use_llm:
        return DeterministicCreativeEngine(), "deterministic engine (offline mode)"
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return (
            DeterministicCreativeEngine(),
            "LLM requested but the 'anthropic' package is not installed "
            "(pip install anthropic) — using deterministic engine",
        )
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return (
            DeterministicCreativeEngine(),
            "LLM requested but no ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN is set "
            "— using deterministic engine",
        )
    engine = ClaudeCreativeEngine()
    return engine, f"Claude creative engine active (model {engine._model})"


def engine_from_config(config: RunnableConfig | None) -> CreativeEngine:
    """Pull the creative engine out of a node's RunnableConfig (with default)."""
    if config:
        engine = config.get("configurable", {}).get("creative_engine")
        if engine is not None:
            return engine
    return DeterministicCreativeEngine()
