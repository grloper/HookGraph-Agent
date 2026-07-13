"""Unit tests for the deterministic linguistic scoring engines."""

from hookgraph.analysis import (
    is_punchy_opening,
    opening_punchiness,
    score_segments,
    semantic_density,
    top_keywords,
    topic_novelty,
)
from hookgraph.state import TranscriptSegment


def seg(segment_id: int, text: str, start: float = 0.0, dur: float = 10.0) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=segment_id, start=start, end=start + dur, speaker="A", text=text
    )


class TestOpeningPunchiness:
    def test_filler_openers_fail_the_gate(self):
        assert not is_punchy_opening("Um, so, I guess we could maybe talk about that thing.")
        assert not is_punchy_opening("Well, you know, it depends on a lot of different factors here honestly.")

    def test_power_openers_pass_the_gate(self):
        assert is_punchy_opening("Stop doing this one thing every morning!")
        assert is_punchy_opening("Why do 90 percent of habits collapse?")

    def test_scores_stay_in_unit_interval(self):
        for text in ["", "hi", "STOP!", "so " * 40, "Nobody warns you about the cliff."]:
            assert 0.0 <= opening_punchiness(text) <= 1.0


class TestSemanticDensity:
    def test_filler_scores_below_dense_content(self):
        filler = seg(0, "you know it was kind of like really just sort of a thing")
        dense = seg(1, "Dopamine prediction error drives reinforcement learning circuits")
        assert semantic_density(filler) < semantic_density(dense)

    def test_bounds(self):
        assert 0.0 <= semantic_density(seg(0, "a the of and")) <= 1.0


class TestTopicNovelty:
    def test_first_segment_is_maximal_pivot(self):
        assert topic_novelty(None, seg(0, "anything at all")) == 1.0

    def test_repeated_vocabulary_scores_low(self):
        a = seg(0, "habits anchors rituals identity receipts compression")
        b = seg(1, "habits anchors rituals identity receipts compression")
        assert topic_novelty(a, b) == 0.0


class TestScoreSegments:
    def test_every_segment_scored_with_valid_peak_type(self):
        transcript = [
            seg(0, "Welcome back to the show everyone.", 0.0),
            seg(1, "This catastrophic collapse shocked everyone!", 10.0),
            seg(2, "Three ingredients: anchor, ritual, receipt.", 20.0),
        ]
        scores = score_segments(transcript)
        assert len(scores) == 3
        for score in scores:
            assert score.peak_type in {"semantic_density", "emotional_spike", "topic_transition"}
            assert 0.0 <= score.retention <= 1.0


def test_top_keywords_deterministic_and_stopword_free():
    text = "the dopamine loop and the dopamine spike and the anchor"
    assert top_keywords(text, limit=2) == ["dopamine", "anchor"]
