"""Tests for the TTS engine layer, smart routing, fallback, and async safety."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os

import pytest

from tts_app.audio_utils import has_ffmpeg, probe_audio
from tts_app.models import TtsError
from tts_app.tts_engine import (
    EdgeTtsEngine,
    Pyttsx3Engine,
    available_engines,
    get_engine,
    language_display_name,
    language_subtag,
    normalize_xtts_language,
    pick_default_voice,
    smart_synthesize,
    smart_synthesize_async,
    speed_to_edge_rate,
    split_text_smart,
)


def test_language_and_text_helpers():
    assert language_subtag("en-US") == "en"
    assert language_subtag("zh_CN") == "zh"
    assert language_display_name("ar") == "Arabic"
    assert language_display_name("ko") == "Korean"
    assert normalize_xtts_language("zh") == "zh-cn"
    assert normalize_xtts_language("fr-FR") == "fr"
    assert speed_to_edge_rate(1.25) == "+25%"
    assert speed_to_edge_rate(0.8) == "-20%"

    chunks = split_text_smart(
        "First sentence is here. Second sentence is slightly longer, with a comma, to test splitting! Third?",
        max_chars=45,
    )
    assert len(chunks) >= 2
    assert all(len(c) <= 45 for c in chunks)


def test_engine_registry_aliases():
    engines = available_engines()
    assert len(engines) >= 1
    for eng in engines:
        assert get_engine(eng.id).id == eng.id
        assert get_engine(eng.label).id == eng.id

    with pytest.raises(TtsError):
        get_engine("nonexistent-engine")


def test_pyttsx3_wav_mp3_and_subtitles(tmp_path):
    if importlib.util.find_spec("pyttsx3") is None:
        pytest.skip("pyttsx3 not installed")
    eng = Pyttsx3Engine()
    voices = eng.list_voices("en")
    if not voices:
        pytest.skip("No pyttsx3 voices on this system")

    wav_out = str(tmp_path / "offline.wav")
    res_wav = smart_synthesize(
        "Testing offline voice synthesis with subtitles.",
        wav_out,
        engine="pyttsx3",
        language="en",
        generate_subtitles=True,
    )
    assert os.path.isfile(res_wav.audio_path)
    assert res_wav.duration_ms > 200
    assert res_wav.srt_path and os.path.isfile(res_wav.srt_path)
    assert res_wav.timings_path and os.path.isfile(res_wav.timings_path)

    if has_ffmpeg():
        mp3_out = str(tmp_path / "offline.mp3")
        res_mp3 = smart_synthesize(
            "Testing MP3 conversion from pyttsx3.",
            mp3_out,
            engine="pyttsx3",
            language="en",
        )
        assert os.path.isfile(res_mp3.audio_path)
        assert probe_audio(res_mp3.audio_path).duration_ms > 200


def test_smart_auto_fallback_when_primary_fails(monkeypatch, tmp_path):
    if importlib.util.find_spec("pyttsx3") is None:
        pytest.skip("pyttsx3 not installed")

    async def broken_edge(*_args, **_kwargs):
        raise TtsError("Simulated network outage")

    monkeypatch.setattr(EdgeTtsEngine, "synthesize_rich_async", broken_edge)

    out = str(tmp_path / "fallback.wav")
    res = smart_synthesize(
        "Fallback should work cleanly.",
        out,
        engine="auto",
        language="en",
        allow_fallback=True,
    )
    assert res.fallback_used is True
    assert res.engine in ("xtts_v2", "pyttsx3")
    assert os.path.isfile(out)
    assert res.warnings


def test_sync_wrapper_safe_inside_running_event_loop(tmp_path):
    if importlib.util.find_spec("pyttsx3") is None:
        pytest.skip("pyttsx3 not installed")

    async def inner():
        out = str(tmp_path / "inside_loop.wav")
        res = smart_synthesize("Inside running loop.", out, engine="pyttsx3")
        assert os.path.isfile(res.audio_path)
        res2 = await smart_synthesize_async(
            "Async synthesis works too.",
            str(tmp_path / "async_ok.wav"),
            engine="pyttsx3",
        )
        assert os.path.isfile(res2.audio_path)

    asyncio.run(inner())


def test_edge_voice_cache_and_fast_path(monkeypatch, tmp_path):
    monkeypatch.setenv("TTS_APP_DATA_DIR", str(tmp_path))
    eng = EdgeTtsEngine()

    # Fast-path explicit voice ID without hitting disk or network
    v = pick_default_voice(eng, language="en", voice_id="en-US-AndrewNeural")
    assert v.id == "en-US-AndrewNeural"
    assert v.locale == "en-US"

    # Disk cache prevents network fetch when fresh
    cache_path = EdgeTtsEngine._cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "ShortName": "en-US-CachedNeural",
                    "FriendlyName": "Microsoft Cached Online",
                    "Locale": "en-US",
                    "Gender": "Female",
                }
            ],
            fh,
        )
    voices = eng._discover()
    assert len(voices) == 1
    assert voices[0].id == "en-US-CachedNeural"
