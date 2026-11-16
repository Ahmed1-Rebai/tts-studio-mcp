"""Tests for the video voiceover narration pipeline and FFmpeg muxing."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess

import pytest

from tts_app.audio_utils import has_ffmpeg, probe_audio
from tts_app.video_pipeline import generate_video_voiceover, split_script_to_scenes


def test_split_script_to_scenes_paragraphs_and_tags():
    tagged = "[scene: hook]\nWelcome!\n\n[scene: demo]\nLook at this workflow."
    scenes = split_script_to_scenes(tagged)
    assert [s.id for s in scenes] == ["hook", "demo"]
    assert scenes[0].text == "Welcome!"

    paras = split_script_to_scenes("Paragraph one.\n\nParagraph two.")
    assert [s.id for s in paras] == ["scene_01", "scene_02"]
    assert paras[1].pause_after_ms == 0


def test_generate_video_voiceover_and_optional_mux(tmp_path):
    if importlib.util.find_spec("pyttsx3") is None:
        pytest.skip("pyttsx3 not installed")

    out_dir = str(tmp_path / "vo_out")
    scenes = [
        {"id": "intro", "text": "Welcome to the video generator.", "pause_after_ms": 250},
        {"id": "outro", "text": "Subscribe for more updates.", "pause_after_ms": 0},
    ]

    video_path = None
    if has_ffmpeg():
        video_path = str(tmp_path / "blank.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x120:d=3",
                "-c:v",
                "libx264",
                video_path,
            ],
            check=True,
        )

    res = generate_video_voiceover(
        scenes=scenes,
        output_dir=out_dir,
        output_name="promo",
        audio_format="mp3" if has_ffmpeg() else "wav",
        fps=30,
        default_engine="pyttsx3",
        default_language="en",
        video_path=video_path,
    )

    assert os.path.isfile(res["master_audio_path"])
    assert os.path.isfile(res["srt_path"])
    assert os.path.isfile(res["vtt_path"])
    assert os.path.isfile(res["timeline_json_path"])
    assert res["scene_count"] == 2
    assert res["total_frames"] > 0

    timeline = json.loads(open(res["timeline_json_path"], encoding="utf-8").read())
    assert len(timeline["scenes"]) == 2
    assert timeline["scenes"][1]["start_ms"] >= timeline["scenes"][0]["end_ms"] + 240
    assert probe_audio(res["master_audio_path"]).duration_ms > 500

    if video_path:
        assert res["muxed_video_path"] and os.path.isfile(res["muxed_video_path"])
