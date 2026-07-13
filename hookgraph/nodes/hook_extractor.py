"""HookExtractor node — finds and repairs the top 3 highest-retention windows.

Fresh pass: score every transcript segment, pick the strongest non-adjacent
retention peaks, and expand each peak outward to its *natural narrative arc*
(neighbors keep getting absorbed while they stay retention-relevant). The
extractor deliberately optimizes for story completeness, not platform limits —
enforcing the 60-second ceiling is QualityControl's job, which is what makes
the corrective feedback loop meaningful.

Repair pass: the Showrunner routes back with ``repair_directives`` — an
explicit ``{hook_id:rule -> strategy}`` plan drawn from per-failure escalation
ladders. The extractor executes exactly that plan, touching only the flagged
hooks (the upsert reducer keeps passing hooks untouched). Strategies range
from surgical (trim the weakest edge segments) through structural (re-window
tightly around the peak) to radical (abandon the window and reseed on the
best unclaimed retention peak elsewhere in the episode).
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ..analysis import (
    SegmentScore,
    opening_punchiness,
    score_segments,
    top_keywords,
)
from ..engines import engine_from_config
from ..state import (
    HookCandidate,
    ScoreBreakdown,
    TranscriptSegment,
    HookGraphState,
)

TOP_HOOK_COUNT = 3
NARRATIVE_ARC_CAP_SECONDS = 95.0   # fresh-pass cap: full story beat
PLATFORM_CAP_SECONDS = 58.0        # repair-pass cap: safely under the 60s rubric
MIN_CLIP_SECONDS = 8.0             # rubric floor: a clip must carry a story beat
EXPANSION_KEEP_RATIO = 0.55        # neighbor must retain >=55% of peak retention
MIN_PEAK_GAP_SEGMENTS = 4          # candidate peaks must be spread apart
ANCHOR_SEARCH_SPAN = 3             # default punchy-start search width
WIDE_ANCHOR_SEARCH_SPAN = 6        # escalated punchy-start search width


# ---------------------------------------------------------------------------
# Window geometry helpers
# ---------------------------------------------------------------------------


def _segment_index(transcript: list[TranscriptSegment]) -> dict[int, int]:
    """Map segment_id -> position in the transcript list."""
    return {segment.segment_id: position for position, segment in enumerate(transcript)}


def _window_duration(transcript: list[TranscriptSegment], lo: int, hi: int) -> float:
    return transcript[hi].end - transcript[lo].start


def _claimed_positions(
    transcript: list[TranscriptSegment],
    windows: dict[str, tuple[float, float]],
) -> set[int]:
    """Every transcript position covered by any of the given time windows."""
    claimed: set[int] = set()
    for start, end in windows.values():
        for position, segment in enumerate(transcript):
            if segment.start < end and segment.end > start:
                claimed.add(position)
    return claimed


def _expand_window(
    transcript: list[TranscriptSegment],
    scores: list[SegmentScore],
    peak_pos: int,
    max_duration: float,
    claimed: set[int],
) -> tuple[int, int]:
    """Grow [lo, hi] around a peak while neighbors stay retention-relevant.

    Greedy symmetric expansion: at each step absorb whichever unclaimed
    neighbor has the higher retention, stopping when both fall below the keep
    ratio, the duration budget is spent, or a neighbor already belongs to
    another hook.
    """
    peak_retention = scores[peak_pos].retention
    floor = peak_retention * EXPANSION_KEEP_RATIO
    lo = hi = peak_pos
    while True:
        left = lo - 1 if lo - 1 >= 0 and (lo - 1) not in claimed else None
        right = hi + 1 if hi + 1 < len(transcript) and (hi + 1) not in claimed else None
        left_score = scores[left].retention if left is not None else -1.0
        right_score = scores[right].retention if right is not None else -1.0

        candidates: list[tuple[float, str]] = []
        if left is not None and left_score >= floor:
            candidates.append((left_score, "left"))
        if right is not None and right_score >= floor:
            candidates.append((right_score, "right"))
        if not candidates:
            break

        best_score, direction = max(candidates)
        new_lo, new_hi = (lo - 1, hi) if direction == "left" else (lo, hi + 1)
        if _window_duration(transcript, new_lo, new_hi) > max_duration:
            # Try the other direction before giving up on expansion entirely.
            if len(candidates) == 2:
                direction = "right" if direction == "left" else "left"
                new_lo, new_hi = (lo - 1, hi) if direction == "left" else (lo, hi + 1)
                if _window_duration(transcript, new_lo, new_hi) > max_duration:
                    break
            else:
                break
        lo, hi = new_lo, new_hi
    return lo, hi


def _trim_window_to_duration(
    transcript: list[TranscriptSegment],
    scores: list[SegmentScore],
    lo: int,
    hi: int,
    max_duration: float,
) -> tuple[int, int]:
    """Shrink a window under a duration cap by shedding the weaker edge first."""
    while lo < hi and _window_duration(transcript, lo, hi) > max_duration:
        if scores[lo].retention <= scores[hi].retention:
            lo += 1
        else:
            hi -= 1
    return lo, hi


def _grow_window_to_min(
    transcript: list[TranscriptSegment],
    lo: int,
    hi: int,
    min_duration: float,
    max_duration: float,
    claimed: set[int],
) -> tuple[int, int]:
    """Expand a too-short window using unclaimed neighbors until it can carry
    a story beat, without breaching the platform cap or sibling footage."""
    while _window_duration(transcript, lo, hi) < min_duration:
        left_ok = lo - 1 >= 0 and (lo - 1) not in claimed
        right_ok = hi + 1 < len(transcript) and (hi + 1) not in claimed
        if right_ok and _window_duration(transcript, lo, hi + 1) <= max_duration:
            hi += 1
        elif left_ok and _window_duration(transcript, lo - 1, hi) <= max_duration:
            lo -= 1
        else:
            break
    return lo, hi


def _anchor_on_punchy_start(
    transcript: list[TranscriptSegment],
    lo: int,
    hi: int,
    span: int = ANCHOR_SEARCH_SPAN,
) -> tuple[int, int]:
    """Advance the window start to the punchiest early segment.

    Vertical clips live or die in the first second, so the clip must open on
    the strongest available line within the front of the window. ``span``
    controls how deep into the window the search reaches; the Showrunner
    escalates it when the default search fails to clear the rubric.
    """
    search_hi = min(hi, lo + span)
    best = lo
    best_punch = opening_punchiness(transcript[lo].text)
    for position in range(lo + 1, search_hi + 1):
        punch = opening_punchiness(transcript[position].text)
        if punch > best_punch + 1e-9:
            best, best_punch = position, punch
    return best, hi


def _reseed_peak(
    transcript: list[TranscriptSegment],
    scores: list[SegmentScore],
    blocked: set[int],
    max_duration: float,
) -> tuple[int, int] | None:
    """Abandon the current window: rebuild around the best unblocked peak.

    ``blocked`` covers both sibling footage and the failing window itself, so
    the reseeded clip is guaranteed to be materially different footage.
    """
    for peak in sorted(scores, key=lambda score: -score.retention):
        position = _segment_index(transcript)[peak.segment_id]
        if position in blocked:
            continue
        lo, hi = _expand_window(transcript, scores, position, max_duration, blocked)
        lo, hi = _anchor_on_punchy_start(transcript, lo, hi)
        lo, hi = _trim_window_to_duration(transcript, scores, lo, hi, max_duration)
        return lo, hi
    return None


def _shift_after_collision(
    transcript: list[TranscriptSegment],
    lo: int,
    hi: int,
    collision_end: float,
    max_duration: float,
) -> tuple[int, int]:
    """Slide the whole window to start after the colliding sibling's footage."""
    length = hi - lo
    new_lo = lo
    while new_lo < len(transcript) and transcript[new_lo].start < collision_end:
        new_lo += 1
    if new_lo >= len(transcript):
        return lo, hi  # nowhere to shift; caller falls through to later rungs
    new_hi = min(new_lo + length, len(transcript) - 1)
    while new_lo < new_hi and _window_duration(transcript, new_lo, new_hi) > max_duration:
        new_hi -= 1
    return new_lo, new_hi


# ---------------------------------------------------------------------------
# Hook assembly
# ---------------------------------------------------------------------------


def _make_hook(
    transcript: list[TranscriptSegment],
    scores: list[SegmentScore],
    hook_id: str,
    rank: int,
    lo: int,
    hi: int,
    revision: int,
    config: RunnableConfig | None = None,
) -> HookCandidate:
    """Assemble a fully-scored HookCandidate from a segment window."""
    window_segments = transcript[lo : hi + 1]
    window_scores = scores[lo : hi + 1]
    count = len(window_scores)

    avg_density = sum(score.semantic_density for score in window_scores) / count
    avg_emotion = sum(score.emotional_intensity for score in window_scores) / count
    avg_novelty = sum(score.topic_novelty for score in window_scores) / count
    punch = opening_punchiness(window_segments[0].text)

    # The clip inherits its type from the retention peak that earned it a slot:
    # the classification is z-scored against the whole episode in analysis.py.
    peak_position = max(range(count), key=lambda position: window_scores[position].retention)
    peak_type = window_scores[peak_position].peak_type
    peak_start = window_segments[peak_position].start

    raw = 0.30 * avg_density + 0.34 * avg_emotion + 0.16 * avg_novelty + 0.20 * punch
    virality = round(min(100.0, raw * 145.0), 1)

    window_text = " ".join(segment.text for segment in window_segments)
    keywords = top_keywords(window_text, limit=3)
    keyword = keywords[0] if keywords else "this"
    engine = engine_from_config(config)
    title = engine.craft_title(window_text, keyword, peak_type, rank)[:80]

    duration = _window_duration(transcript, lo, hi)
    justification = (
        f"Retention composite {virality:.1f}/100 across {count} segments "
        f"({duration:.1f}s): semantic density {avg_density:.2f}, emotional "
        f"intensity {avg_emotion:.2f}, topic novelty {avg_novelty:.2f}, and an "
        f"opening-punch score of {punch:.2f} on the first line "
        f"('{window_segments[0].text[:60].strip()}…'). The retention peak at "
        f"{peak_start:.0f}s registers as {peak_type.replace('_', ' ')} against "
        f"the episode baseline."
    )

    return HookCandidate(
        hook_id=hook_id,
        rank=rank,
        hook_title=title,
        virality_score=virality,
        virality_justification=justification,
        peak_type=peak_type,
        start_seconds=window_segments[0].start,
        end_seconds=window_segments[-1].end,
        segment_ids=[segment.segment_id for segment in window_segments],
        opening_line=window_segments[0].text,
        score_breakdown=ScoreBreakdown(
            semantic_density=round(min(1.0, avg_density), 4),
            emotional_intensity=round(min(1.0, avg_emotion), 4),
            topic_novelty=round(min(1.0, avg_novelty), 4),
            opening_punch=round(min(1.0, punch), 4),
        ),
        revision=revision,
    )


def _extract_fresh(
    transcript: list[TranscriptSegment],
    scores: list[SegmentScore],
    config: RunnableConfig | None,
    max_window_seconds: float = NARRATIVE_ARC_CAP_SECONDS,
) -> list[HookCandidate]:
    """First-pass extraction: top peaks expanded to their narrative arcs.

    Peaks are selected and expanded in one greedy pass over the retention
    ranking: a candidate peak that already fell inside a claimed window is
    skipped in favor of the next-best unclaimed peak, which structurally
    guarantees the three windows never overlap.
    """
    ranked_peaks = sorted(scores, key=lambda score: -score.retention)
    positions = _segment_index(transcript)

    claimed: set[int] = set()
    chosen_peaks: list[int] = []
    windows: list[tuple[int, int, int]] = []  # (peak_pos, lo, hi)
    for peak in ranked_peaks:
        position = positions[peak.segment_id]
        if position in claimed:
            continue
        if any(abs(position - other) < MIN_PEAK_GAP_SEGMENTS for other in chosen_peaks):
            continue
        lo, hi = _expand_window(transcript, scores, position, max_window_seconds, claimed)
        lo, _ = _anchor_on_punchy_start(transcript, lo, hi)
        claimed.update(range(lo, hi + 1))
        chosen_peaks.append(position)
        windows.append((position, lo, hi))
        if len(windows) == TOP_HOOK_COUNT:
            break

    # Rank hooks by their peak segment's retention (strongest peak = rank 1).
    windows.sort(key=lambda window: -scores[window[0]].retention)
    return [
        _make_hook(transcript, scores, f"hook-{rank}", rank, lo, hi, revision=0, config=config)
        for rank, (_, lo, hi) in enumerate(windows, start=1)
    ]


# ---------------------------------------------------------------------------
# Strategy execution (the Showrunner plans; the extractor executes)
# ---------------------------------------------------------------------------

# Rules are repaired in dependency order: fix footage collisions before
# duration, duration before the opening anchor (anchoring can't lengthen a
# clip, and trimming can invalidate a previously-chosen anchor).
_RULE_ORDER = (
    "valid_timestamps",
    "non_overlapping_times",
    "duration_under_60s",
    "punchy_opening_line",
    "metadata_completeness",
    "justified_virality",
)


def _repair_hook(
    hook: HookCandidate,
    plan: dict[str, str],
    transcript: list[TranscriptSegment],
    scores: list[SegmentScore],
    sibling_windows: dict[str, tuple[float, float]],
    config: RunnableConfig | None,
) -> HookCandidate:
    """Execute the Showrunner's per-rule strategy plan for one failing hook."""
    positions = _segment_index(transcript)
    known_ids = [sid for sid in hook.segment_ids if sid in positions]
    if known_ids:
        lo = positions[known_ids[0]]
        hi = positions[known_ids[-1]]
    else:
        lo, hi = 0, min(len(transcript) - 1, MIN_PEAK_GAP_SEGMENTS)
    if lo > hi:
        lo, hi = hi, lo
    sibling_claimed = _claimed_positions(transcript, sibling_windows)

    reseed_requested = "reseed_peak" in plan.values()
    if reseed_requested:
        # Radical strategy: block both sibling footage AND the failing window,
        # then rebuild somewhere genuinely new. Solves every rule at once.
        blocked = sibling_claimed | set(range(lo, hi + 1))
        reseeded = _reseed_peak(transcript, scores, blocked, PLATFORM_CAP_SECONDS)
        if reseeded is not None:
            lo, hi = reseeded
    else:
        for rule in _RULE_ORDER:
            strategy = plan.get(rule)
            if strategy is None:
                continue

            if rule == "valid_timestamps":
                # Re-snapping happens structurally: _make_hook rebuilds start/end
                # from real segment boundaries, so nothing to do geometrically.
                continue

            if rule == "non_overlapping_times":
                for other_start, other_end in sibling_windows.values():
                    overlaps = (
                        transcript[lo].start < other_end
                        and transcript[hi].end > other_start
                    )
                    if not overlaps:
                        continue
                    if strategy == "shift_after_collision":
                        lo, hi = _shift_after_collision(
                            transcript, lo, hi, other_end, PLATFORM_CAP_SECONDS
                        )
                    else:  # shrink_from_collision
                        while (
                            lo < hi
                            and transcript[lo].start < other_end
                            and transcript[hi].end > other_start
                        ):
                            if transcript[lo].start >= other_start:
                                lo += 1
                            else:
                                hi -= 1

            elif rule == "duration_under_60s":
                duration = _window_duration(transcript, lo, hi)
                if duration < MIN_CLIP_SECONDS:
                    lo, hi = _grow_window_to_min(
                        transcript, lo, hi, MIN_CLIP_SECONDS,
                        PLATFORM_CAP_SECONDS, sibling_claimed,
                    )
                elif strategy == "tight_rewindow":
                    window_scores = scores[lo : hi + 1]
                    peak_pos = lo + max(
                        range(len(window_scores)),
                        key=lambda position: window_scores[position].retention,
                    )
                    lo, hi = _expand_window(
                        transcript, scores, peak_pos,
                        PLATFORM_CAP_SECONDS, sibling_claimed,
                    )
                else:  # trim_weak_edges
                    lo, hi = _trim_window_to_duration(
                        transcript, scores, lo, hi, PLATFORM_CAP_SECONDS
                    )

            elif rule == "punchy_opening_line":
                span = (
                    WIDE_ANCHOR_SEARCH_SPAN
                    if strategy == "widen_anchor_search"
                    else ANCHOR_SEARCH_SPAN
                )
                lo, _ = _anchor_on_punchy_start(transcript, lo, hi, span=span)

            # metadata_completeness / justified_virality need no geometry work:
            # the revision bump below forces the Scriptwriter to regenerate
            # artifacts and _make_hook recomputes the justification.

    # Final safety pass: whatever the strategies did, the emitted window must
    # respect the platform cap and never invert.
    if lo > hi:
        lo, hi = hi, lo
    lo, hi = _trim_window_to_duration(transcript, scores, lo, hi, PLATFORM_CAP_SECONDS)

    return _make_hook(
        transcript, scores, hook.hook_id, hook.rank, lo, hi,
        revision=hook.revision + 1, config=config,
    )


def hook_extractor_node(state: HookGraphState, config: RunnableConfig) -> dict:
    """LangGraph node handler: fresh extraction or directive-driven repair."""
    transcript = state["transcript"]
    scores = score_segments(transcript)
    attempt = state["extraction_attempts"] + 1
    directives = state["repair_directives"]

    structural = "package:exactly_three_hooks" in directives
    if not directives or not state["hooks"] or structural:
        # A structural failure means the wide narrative-arc windows could not
        # coexist; retry the whole extraction with tight platform-sized windows.
        cap = PLATFORM_CAP_SECONDS if structural else NARRATIVE_ARC_CAP_SECONDS
        hooks = _extract_fresh(transcript, scores, config, max_window_seconds=cap)
        events = [
            f"[HookExtractor] Attempt {attempt}: scored {len(transcript)} segments and "
            f"extracted {len(hooks)} hooks (window cap {cap:.0f}s): "
            + "; ".join(
                f"{hook.hook_id} '{hook.hook_title}' "
                f"({hook.start_seconds:.0f}s-{hook.end_seconds:.0f}s, "
                f"score {hook.virality_score})"
                for hook in hooks
            )
        ]
        return {
            "hooks": hooks,
            "extraction_attempts": attempt,
            "active_violations": [],
            "repair_directives": {},
            "pipeline_events": events,
        }

    # Repair pass: execute the Showrunner's plan, touching only flagged hooks.
    plans: dict[str, dict[str, str]] = {}
    for key, strategy in directives.items():
        hook_id, _, rule = key.partition(":")
        plans.setdefault(hook_id, {})[rule] = strategy

    current = {hook.hook_id: hook for hook in state["hooks"]}
    sibling_windows = {
        hook.hook_id: (hook.start_seconds, hook.end_seconds)
        for hook in state["hooks"]
        if hook.hook_id not in plans  # only healthy siblings constrain repairs
    }

    repaired: list[HookCandidate] = []
    events: list[str] = []
    for hook_id, plan in sorted(plans.items()):
        if hook_id not in current:
            continue
        fixed = _repair_hook(
            current[hook_id], plan, transcript, scores, sibling_windows, config
        )
        sibling_windows[fixed.hook_id] = (fixed.start_seconds, fixed.end_seconds)
        repaired.append(fixed)
        applied = ", ".join(f"{rule}={strategy}" for rule, strategy in sorted(plan.items()))
        events.append(
            f"[HookExtractor] Attempt {attempt}: repaired {hook_id} (rev "
            f"{fixed.revision}) via [{applied}] -> now "
            f"{fixed.start_seconds:.0f}s-{fixed.end_seconds:.0f}s "
            f"({fixed.duration_seconds:.1f}s), opening line "
            f"'{fixed.opening_line[:50].strip()}…'"
        )

    return {
        "hooks": repaired,  # upsert reducer merges these over the failing ones
        "extraction_attempts": attempt,
        "active_violations": [],
        "repair_directives": {},
        "pipeline_events": events,
    }
