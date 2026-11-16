"""Tests for SRT/VTT/JSON subtitle generation and word-timing utilities."""

from __future__ import annotations

import json
import os

from tts_app.models import WordTiming
from tts_app.subtitles import (
    estimate_word_timings,
    format_srt_time,
    format_vtt_time,
    group_words_into_cues,
    to_srt,
    to_vtt,
    write_subtitle_artifacts,
)


def test_time_formatting():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(3_661_250) == "01:01:01,250"
    assert format_vtt_time(3_661_250) == "01:01:01.250"


def test_estimate_word_timings_spans_full_window():
    words = estimate_word_timings("Hello, world! Ready?", 1000, 3000, scene_id="intro")
    assert [w.text for w in words] == ["Hello,", "world!", "Ready?"]
    assert words[0].start_ms == 1000
    assert words[-1].end_ms == 3000
    assert all(w.scene_id == "intro" for w in words)
    assert all(words[i].end_ms <= words[i + 1].start_ms + 1 for i in range(len(words) - 1))


def test_group_words_respects_sentence_pause_and_scene_boundaries():
    words = [
        WordTiming("First", 0, 200, "s1"),
        WordTiming("sentence.", 210, 450, "s1"),
        WordTiming("Second", 500, 700, "s1"),
        WordTiming("scene", 710, 900, "s2"),
        WordTiming("far", 2000, 2200, "s2"),
    ]
    cues = group_words_into_cues(words, max_words=6, max_chars=42, max_pause_ms=400)
    assert [c.text for c in cues] == ["First sentence.", "Second", "scene", "far"]


def test_write_subtitle_artifacts_outputs_valid_files(tmp_path):
    words = [
        WordTiming("Build", 0, 300, "s1"),
        WordTiming("videos", 300, 700, "s1"),
        WordTiming("faster.", 700, 1100, "s1"),
    ]
    stem = os.path.join(str(tmp_path), "narration")
    srt_p, vtt_p, json_p, cues = write_subtitle_artifacts(
        stem_path=stem,
        words=words,
        duration_ms=1200,
        fps=30,
        timing_mode="word_boundary",
        max_words_per_cue=3,
    )
    assert len(cues) == 1
    assert "00:00:00,000 --> 00:00:01,100" in to_srt(cues)
    assert "WEBVTT" in to_vtt(cues)
    assert os.path.isfile(srt_p)
    assert os.path.isfile(vtt_p)
    data = json.loads(open(json_p, encoding="utf-8").read())
    assert data["fps"] == 30
    assert data["total_frames"] == 36
    assert data["cues"][0]["start_frame"] == 0
    assert data["cues"][0]["end_frame"] == 33
