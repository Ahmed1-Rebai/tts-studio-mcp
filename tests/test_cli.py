"""Tests for the tts-cli command-line interface."""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

from tts_app.cli import main


def test_cli_status_and_profiles_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("TTS_APP_DATA_DIR", str(tmp_path))
    rc = main(["status", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "engines" in payload
    assert "hardware" in payload

    rc_prof = main(["profiles", "list"])
    assert rc_prof == 0
    assert json.loads(capsys.readouterr().out) == []

    rc_missing = main(["profiles", "inspect", "does_not_exist"])
    assert rc_missing == 1


def test_cli_speak_and_video_voiceover(monkeypatch, tmp_path, capsys):
    if importlib.util.find_spec("pyttsx3") is None:
        pytest.skip("pyttsx3 not installed")
    monkeypatch.setenv("TTS_APP_DATA_DIR", str(tmp_path))

    out_wav = str(tmp_path / "cli_out.wav")
    rc = main(
        [
            "speak",
            "Testing CLI speech output.",
            "-o",
            out_wav,
            "--engine",
            "pyttsx3",
            "--subtitles",
            "--json",
        ]
    )
    assert rc == 0
    speak_data = json.loads(capsys.readouterr().out)
    assert os.path.isfile(speak_data["audio_path"])
    assert os.path.isfile(speak_data["srt_path"])

    vo_dir = str(tmp_path / "cli_vo")
    rc_vo = main(
        [
            "video-voiceover",
            "--script",
            "First line.\n\nSecond line.",
            "--output-dir",
            vo_dir,
            "--format",
            "wav",
            "--engine",
            "pyttsx3",
            "--json",
        ]
    )
    assert rc_vo == 0
    vo_data = json.loads(capsys.readouterr().out)
    assert vo_data["scene_count"] == 2
    assert os.path.isfile(vo_data["master_audio_path"])
    assert os.path.isfile(vo_data["timeline_json_path"])
