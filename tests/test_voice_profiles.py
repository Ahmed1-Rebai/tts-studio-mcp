"""Tests for voice profile enrollment, dataset preparation, and hardware checks."""

from __future__ import annotations

import array
import csv
import math
import os
import wave

import pytest

from tts_app.models import TtsError
from tts_app.voice_profiles import (
    check_hardware_requirements,
    delete_profile,
    enroll_voice_profile,
    get_profile,
    list_profiles,
    prepare_training_dataset,
    slugify_profile,
    train_custom_voice,
)


def _make_wav(path: str, seconds: float = 3.0, sr: int = 22050) -> str:
    samples = array.array("h")
    for i in range(int(seconds * sr)):
        samples.append(int(12000 * math.sin(2 * math.pi * 220.0 * i / sr)))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


def test_slugify_and_hardware_check(monkeypatch, tmp_path):
    monkeypatch.setenv("TTS_APP_DATA_DIR", str(tmp_path))
    assert slugify_profile("Ahmed's Voice 01!") == "ahmed_s_voice_01"
    with pytest.raises(TtsError):
        slugify_profile("!!!")

    hw = check_hardware_requirements()
    assert hw["cpu_cores"] >= 1
    assert "tiers" in hw
    assert "xtts_finetuning" in hw["tiers"]


def test_enroll_list_get_and_delete_profile(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("TTS_APP_DATA_DIR", str(tmp_path))

    class FakeXttsModel:
        class config:
            gpt_cond_len = 6
            max_ref_len = 30
            sound_norm_refs = False

        def get_conditioning_latents(self, audio_path, **_kwargs):
            assert len(audio_path) == 2
            return torch.zeros((1, 32, 1024)), torch.ones((1, 512, 1))

    s1 = _make_wav(str(tmp_path / "clip1.wav"), seconds=3.0)
    s2 = _make_wav(str(tmp_path / "clip2.wav"), seconds=3.5)

    prof = enroll_voice_profile(
        name="My Studio Voice",
        audio_paths=[s1, s2],
        language="en",
        preprocess=True,
        xtts_model=FakeXttsModel(),
    )
    assert prof.id == "my_studio_voice"
    assert prof.sample_count == 2
    assert prof.total_source_duration_s >= 4.0
    assert prof.latent_path and os.path.isfile(prof.latent_path)

    loaded = list_profiles()
    assert len(loaded) == 1
    assert get_profile("My Studio Voice").id == prof.id
    assert delete_profile("my_studio_voice") is True
    assert list_profiles() == []


def test_prepare_training_dataset_and_dry_run(monkeypatch, tmp_path):
    monkeypatch.setenv("TTS_APP_DATA_DIR", str(tmp_path / "appdata"))
    c1 = _make_wav(str(tmp_path / "c1.wav"), seconds=2.0)
    c2 = _make_wav(str(tmp_path / "c2.wav"), seconds=2.0)

    ds_dir = str(tmp_path / "ds")
    with pytest.raises(TtsError):
        prepare_training_dataset(
            output_dir=ds_dir,
            audio_paths=[c1, c2],
            transcripts=["Clip one.", "Clip two."],
            allow_short=False,
        )

    info = prepare_training_dataset(
        output_dir=ds_dir,
        audio_paths=[c1, c2],
        transcripts={"c1.wav": "First training clip.", "c2.wav": "Second training clip."},
        allow_short=True,
    )
    assert info["total_clips"] == 2
    assert os.path.isfile(info["train_csv"])
    assert os.path.isfile(info["eval_csv"])

    with open(info["train_csv"], encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="|"))
    assert rows[0] == ["audio_file", "text", "speaker_name"]
    assert len(rows) >= 2

    dry = train_custom_voice(
        profile_name="Trained Voice",
        dataset_dir=ds_dir,
        allow_cpu=True,
        allow_short=True,
        dry_run=True,
    )
    assert dry["status"] == "dry_run_ok"
    assert dry["profile_id"] == "trained_voice"
