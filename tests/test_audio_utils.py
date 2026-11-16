"""Tests for audio probing, conversion, trimming, normalization, and concatenation."""

from __future__ import annotations

import array
import math
import os
import wave

import pytest

from tts_app.audio_utils import (
    concat_audio_segments,
    convert_audio,
    has_ffmpeg,
    normalize_loudness,
    preprocess_voice_sample,
    probe_audio,
    trim_silence,
)


def _write_tone_wav(
    path: str,
    *,
    tone_s: float = 0.6,
    lead_silence_s: float = 0.0,
    tail_silence_s: float = 0.0,
    sample_rate: int = 24000,
    amplitude: float = 0.4,
) -> str:
    samples = array.array("h")
    samples.extend([0] * int(sample_rate * lead_silence_s))
    n_tone = int(sample_rate * tone_s)
    for i in range(n_tone):
        val = int(32767 * amplitude * math.sin(2 * math.pi * 440.0 * i / sample_rate))
        samples.append(val)
    samples.extend([0] * int(sample_rate * tail_silence_s))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return path


def test_probe_trim_and_concat_wav(tmp_path):
    a = _write_tone_wav(
        str(tmp_path / "a.wav"), tone_s=0.5, lead_silence_s=0.35, tail_silence_s=0.35
    )
    b = _write_tone_wav(str(tmp_path / "b.wav"), tone_s=0.4)

    info_before = probe_audio(a)
    assert 1150 <= info_before.duration_ms <= 1250

    trimmed = str(tmp_path / "a_trimmed.wav")
    trim_silence(a, trimmed)
    info_after = probe_audio(trimmed)
    assert info_after.duration_ms < info_before.duration_ms - 300

    out = str(tmp_path / "joined.wav")
    _, timings = concat_audio_segments([(trimmed, 250), (b, 0)], out)
    assert len(timings) == 2
    assert timings[0]["start_ms"] == 0
    assert timings[1]["start_ms"] == timings[0]["end_ms"] + 250
    joined_info = probe_audio(out)
    assert joined_info.duration_ms >= timings[1]["end_ms"] - 30


@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg required for MP3/loudnorm tests")
def test_ffmpeg_convert_normalize_and_preprocess(tmp_path):
    wav = _write_tone_wav(str(tmp_path / "tone.wav"), tone_s=2.5, amplitude=0.1)
    mp3 = str(tmp_path / "tone.mp3")
    convert_audio(wav, mp3)
    assert os.path.isfile(mp3)
    assert probe_audio(mp3).duration_ms >= 2300

    norm_mp3 = str(tmp_path / "norm.mp3")
    normalize_loudness(mp3, norm_mp3)
    assert os.path.isfile(norm_mp3)

    pre_wav = str(tmp_path / "clean_ref.wav")
    preprocess_voice_sample(mp3, pre_wav)
    info = probe_audio(pre_wav)
    assert info.sample_rate == 22050
    assert info.channels == 1
