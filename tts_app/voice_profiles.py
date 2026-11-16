"""Voice profile enrollment, dataset preparation, hardware checks, and XTTS training.

Supports two ways for a user to personalize speech with their own voice:

1. **Voice Profile Enrollment** (``enroll_voice_profile``):
   Preprocesses one or more voice recordings (WAV/MP3/FLAC/OGG/M4A) and caches
   XTTS speaker conditioning latents (``gpt_cond_latent`` and ``speaker_embedding``)
   as a reusable profile. Works on both CPU and GPU in seconds.

2. **Full Custom Voice Fine-Tuning** (``train_custom_voice``):
   Prepares a segmented dataset (``metadata_train.csv`` / ``metadata_eval.csv``)
   and fine-tunes the XTTS v2 GPT encoder using ``coqui-tts-trainer``. Intended
   for NVIDIA CUDA GPUs with >= 8 GB VRAM and >= 2 minutes of clean speech.
"""

from __future__ import annotations

import csv
import ctypes.util
import datetime as _dt
import gc
import importlib.util
import json
import os
import random
import re
import shutil
import typing as _t

from tts_app.audio_utils import (
    convert_audio,
    has_ffmpeg,
    has_ffprobe,
    preprocess_voice_sample,
    probe_audio,
)
from tts_app.models import TtsError, VoiceProfile

_SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_DEFAULT_XTTS_MODEL_SUBDIR = "tts_models--multilingual--multi-dataset--xtts_v2"


def get_data_dir() -> str:
    """Return the root directory for local profiles, training runs, and outputs."""
    override = os.environ.get("TTS_APP_DATA_DIR")
    if override:
        return os.path.abspath(override)
    return os.path.join(os.path.expanduser("~"), ".local", "share", "tts_app")


def get_profiles_dir() -> str:
    return os.path.join(get_data_dir(), "profiles")


def get_outputs_dir() -> str:
    return os.path.join(get_data_dir(), "outputs")


def slugify_profile(name: str) -> str:
    """Normalize a user-supplied voice profile name into a safe filesystem id."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_-")
    if not cleaned:
        raise TtsError("Profile name must contain at least one letter or digit.")
    return cleaned


def _system_ram_gb() -> tuple[float, float]:
    """Return ``(total_gb, available_gb)`` using psutil or ``/proc/meminfo``."""
    if importlib.util.find_spec("psutil") is not None:
        import psutil

        vm = psutil.virtual_memory()
        return round(vm.total / (1024**3), 2), round(vm.available / (1024**3), 2)

    meminfo_path = "/proc/meminfo"
    if os.path.isfile(meminfo_path):
        vals: dict[str, int] = {}
        with open(meminfo_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    vals[parts[0].rstrip(":")] = int(parts[1]) * 1024
        total = vals.get("MemTotal", 0)
        avail = vals.get("MemAvailable", vals.get("MemFree", 0))
        if total > 0:
            return round(total / (1024**3), 2), round(avail / (1024**3), 2)
    return 0.0, 0.0


def _espeak_available() -> bool:
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return True
    return bool(ctypes.util.find_library("espeak-ng") or ctypes.util.find_library("espeak"))


def check_hardware_requirements() -> dict[str, _t.Any]:
    """Inspect local CPU, RAM, disk, GPU/VRAM, and optional TTS libraries."""
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    disk = shutil.disk_usage(data_dir)
    disk_free_gb = round(disk.free / (1024**3), 2)
    ram_total_gb, ram_available_gb = _system_ram_gb()
    cpu_cores = os.cpu_count() or 1

    edge_installed = importlib.util.find_spec("edge_tts") is not None
    pyttsx3_installed = importlib.util.find_spec("pyttsx3") is not None
    torch_installed = importlib.util.find_spec("torch") is not None
    coqui_installed = importlib.util.find_spec("TTS") is not None
    trainer_installed = importlib.util.find_spec("trainer") is not None
    whisper_installed = importlib.util.find_spec("faster_whisper") is not None

    torch_version = None
    cuda_available = False
    mps_available = False
    gpu_name = None
    vram_gb = 0.0

    if torch_installed:
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                props = torch.cuda.get_device_properties(0)
                gpu_name = props.name
                vram_gb = round(props.total_memory / (1024**3), 2)
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                mps_available = True
        except Exception:
            pass

    xtts_cache_dir = os.path.join(
        os.path.expanduser("~"), ".local", "share", "tts", _DEFAULT_XTTS_MODEL_SUBDIR
    )
    xtts_cached = os.path.isfile(os.path.join(xtts_cache_dir, "model.pth"))

    # Tier readiness evaluation
    standard_ready = edge_installed or pyttsx3_installed

    xtts_inf_reasons: list[str] = []
    if not (torch_installed and coqui_installed):
        xtts_inf_reasons.append("Install optional deps from requirements-xtts.txt.")
    if ram_total_gb and ram_total_gb < 4.0:
        xtts_inf_reasons.append(f"Need >= 4 GB RAM (detected {ram_total_gb} GB).")
    if not xtts_cached and disk_free_gb < 2.5:
        xtts_inf_reasons.append(f"Need >= 2.5 GB free disk for XTTS model download ({disk_free_gb} GB free).")

    xtts_train_reasons: list[str] = []
    if not (torch_installed and coqui_installed and trainer_installed):
        xtts_train_reasons.append("Install optional deps from requirements-xtts.txt.")
    if not cuda_available:
        xtts_train_reasons.append(
            "No CUDA GPU detected in PyTorch (CPU training is too slow; install CUDA PyTorch on an >= 8 GB VRAM NVIDIA GPU)."
        )
    elif vram_gb < 7.5:
        xtts_train_reasons.append(f"At least 8 GB VRAM recommended for XTTS fine-tuning (detected {vram_gb} GB).")
    if ram_total_gb and ram_total_gb < 12.0:
        xtts_train_reasons.append(f"16 GB RAM recommended for fine-tuning (detected {ram_total_gb} GB).")
    if disk_free_gb < 8.0:
        xtts_train_reasons.append(f"At least 8-10 GB free disk recommended ({disk_free_gb} GB free).")

    return {
        "cpu_cores": cpu_cores,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "disk_free_gb": disk_free_gb,
        "ffmpeg_installed": has_ffmpeg(),
        "ffprobe_installed": has_ffprobe(),
        "espeak_installed": _espeak_available(),
        "edge_tts_installed": edge_installed,
        "pyttsx3_installed": pyttsx3_installed,
        "torch_installed": torch_installed,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "coqui_tts_installed": coqui_installed,
        "trainer_installed": trainer_installed,
        "faster_whisper_installed": whisper_installed,
        "xtts_model_cached": xtts_cached,
        "xtts_cache_dir": xtts_cache_dir,
        "tiers": {
            "standard_tts": {
                "ready": standard_ready,
                "minimum": "Dual-core CPU, 512 MB RAM, 200 MB disk, internet (for edge-tts) or OS voices (pyttsx3)",
                "notes": [] if standard_ready else ["Install requirements.txt"],
            },
            "xtts_inference": {
                "ready": len(xtts_inf_reasons) == 0,
                "minimum": "Quad-core CPU, 8 GB RAM, 5 GB disk (GPU w/ 4 GB+ VRAM recommended)",
                "notes": xtts_inf_reasons,
            },
            "xtts_finetuning": {
                "ready": len(xtts_train_reasons) == 0,
                "minimum": "NVIDIA CUDA GPU >= 8 GB VRAM, 16 GB RAM, 10+ GB free disk, >= 2 min clean speech",
                "notes": xtts_train_reasons,
            },
        },
    }


def _profile_dir(profile_id: str) -> str:
    return os.path.join(get_profiles_dir(), slugify_profile(profile_id))


def save_profile(profile: VoiceProfile) -> VoiceProfile:
    """Persist ``profile.json`` inside the profile directory."""
    pdir = _profile_dir(profile.id)
    os.makedirs(pdir, exist_ok=True)
    meta_path = os.path.join(pdir, "profile.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(profile.to_dict(), fh, indent=2, ensure_ascii=False)
    return profile


def list_profiles() -> list[VoiceProfile]:
    """Return all saved enrolled and fine-tuned voice profiles sorted by id."""
    root = get_profiles_dir()
    if not os.path.isdir(root):
        return []
    profiles: list[VoiceProfile] = []
    for entry in sorted(os.listdir(root)):
        meta_path = os.path.join(root, entry, "profile.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    profiles.append(VoiceProfile.from_dict(json.load(fh)))
            except (OSError, ValueError, KeyError):
                continue
    return profiles


def get_profile(name_or_id: str) -> VoiceProfile:
    """Look up a saved voice profile by id or display name."""
    target = slugify_profile(name_or_id)
    for prof in list_profiles():
        if prof.id == target or prof.name.strip().lower() == name_or_id.strip().lower():
            return prof
    raise TtsError(f"Voice profile not found: {name_or_id}")


def delete_profile(name_or_id: str) -> bool:
    """Delete a saved voice profile and its cached files."""
    prof = get_profile(name_or_id)
    pdir = _profile_dir(prof.id)
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
        return True
    return False


def _collect_audio_inputs(paths: _t.Sequence[str] | str) -> list[str]:
    if isinstance(paths, str):
        raw_list = [paths]
    else:
        raw_list = list(paths)

    collected: list[str] = []
    for item in raw_list:
        if os.path.isdir(item):
            for root, _dirs, files in os.walk(item):
                for fn in sorted(files):
                    if os.path.splitext(fn)[1].lower() in _SUPPORTED_AUDIO_EXTS:
                        collected.append(os.path.join(root, fn))
        elif os.path.isfile(item):
            collected.append(item)
        else:
            raise TtsError(f"Audio path does not exist: {item}")

    if not collected:
        raise TtsError("No supported audio files (.wav, .mp3, .flac, .ogg, .m4a) found.")
    return collected


def _load_base_xtts_model():
    """Load the base XTTS v2 model onto CUDA if available, otherwise CPU."""
    if importlib.util.find_spec("TTS") is None:
        raise TtsError(
            "Coqui XTTS is not installed. Install requirements-xtts.txt to enroll or train voices."
        )
    import torch
    from TTS.api import TTS as TtsApi

    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    device = os.environ.get("TTS_APP_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    api = TtsApi("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return api.synthesizer.tts_model


def enroll_voice_profile(
    name: str,
    audio_paths: _t.Sequence[str] | str,
    language: str = "en",
    preprocess: bool = True,
    notes: str = "",
    xtts_model: _t.Any = None,
) -> VoiceProfile:
    """Enroll a reusable user voice profile from one or more audio samples.

    Preprocesses the supplied sample(s), extracts XTTS speaker conditioning
    latents (``gpt_cond_latent`` and ``speaker_embedding``), and persists them
    to ``<data_dir>/profiles/<id>/latents.pth`` for fast future synthesis.
    """
    profile_id = slugify_profile(name)
    sources = _collect_audio_inputs(audio_paths)

    pdir = _profile_dir(profile_id)
    refs_dir = os.path.join(pdir, "refs")
    os.makedirs(refs_dir, exist_ok=True)

    prepared_files: list[str] = []
    total_duration_s = 0.0
    for idx, src in enumerate(sources):
        dst = os.path.join(refs_dir, f"sample_{idx:02d}.wav")
        if preprocess:
            preprocess_voice_sample(src, dst, max_duration_s=30.0)
        else:
            convert_audio(src, dst, sample_rate=22050, channels=1)
        info = probe_audio(dst)
        total_duration_s += info.duration_s
        prepared_files.append(dst)

    if total_duration_s < 2.0:
        raise TtsError(
            f"Voice sample is too short ({total_duration_s:.2f}s). Provide at least 2 seconds (6-30s recommended)."
        )

    model = xtts_model if xtts_model is not None else _load_base_xtts_model()
    import torch

    cfg = getattr(model, "config", None)
    gpt_cond_len = getattr(cfg, "gpt_cond_len", 6) if cfg else 6
    max_ref_length = getattr(cfg, "max_ref_len", 30) if cfg else 30
    sound_norm_refs = getattr(cfg, "sound_norm_refs", False) if cfg else False

    with torch.inference_mode():
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=prepared_files,
            max_ref_length=max_ref_length,
            gpt_cond_len=gpt_cond_len,
            sound_norm_refs=sound_norm_refs,
        )

    latent_path = os.path.join(pdir, "latents.pth")
    torch.save(
        {
            "gpt_cond_latent": gpt_cond_latent.detach().cpu(),
            "speaker_embedding": speaker_embedding.detach().cpu(),
        },
        latent_path,
    )

    profile = VoiceProfile(
        id=profile_id,
        name=name.strip(),
        kind="enrolled",
        language=language.lower(),
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        sample_count=len(prepared_files),
        total_source_duration_s=round(total_duration_s, 2),
        sample_files=prepared_files,
        latent_path=latent_path,
        speaker_ref_wav=prepared_files[0],
        notes=notes,
    )
    return save_profile(profile)


def _read_metadata_rows(metadata_csv: str, base_dir: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(metadata_csv, "r", encoding="utf-8") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        delimiter = "|" if "|" in sample else ","
        reader = csv.reader(fh, delimiter=delimiter)
        for line in reader:
            if not line or len(line) < 2:
                continue
            first = line[0].strip()
            second = line[1].strip()
            if first.lower() in ("audio_file", "file", "path") and second.lower() in (
                "text",
                "transcript",
            ):
                continue
            rel = first if os.path.isabs(first) else os.path.join(base_dir, first)
            if not os.path.isfile(rel) and not rel.endswith(".wav"):
                wav_rel = rel + ".wav"
                if os.path.isfile(wav_rel):
                    rel = wav_rel
            if os.path.isfile(rel) and second:
                rows.append((rel, second))
    return rows


def prepare_training_dataset(
    output_dir: str,
    audio_paths: _t.Optional[_t.Sequence[str] | str] = None,
    transcripts: _t.Optional[_t.Union[dict[str, str], _t.Sequence[str]]] = None,
    metadata_csv: _t.Optional[str] = None,
    language: str = "en",
    speaker_name: str = "custom_voice",
    auto_transcribe: bool = False,
    eval_ratio: float = 0.15,
    min_duration_s: float = 120.0,
    allow_short: bool = False,
) -> dict[str, _t.Any]:
    """Build Coqui-formatted ``metadata_train.csv`` and ``metadata_eval.csv``."""
    os.makedirs(output_dir, exist_ok=True)
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    pairs: list[tuple[str, str]] = []
    if metadata_csv:
        if not os.path.isfile(metadata_csv):
            raise TtsError(f"Metadata CSV not found: {metadata_csv}")
        pairs = _read_metadata_rows(metadata_csv, os.path.dirname(os.path.abspath(metadata_csv)))
    elif audio_paths is not None:
        sources = _collect_audio_inputs(audio_paths)
        if isinstance(transcripts, dict):
            for src in sources:
                base = os.path.basename(src)
                stem = os.path.splitext(base)[0]
                text = transcripts.get(src) or transcripts.get(base) or transcripts.get(stem)
                if text and text.strip():
                    pairs.append((src, text.strip()))
        elif isinstance(transcripts, (list, tuple)):
            if len(transcripts) != len(sources):
                raise TtsError(
                    f"Transcripts count ({len(transcripts)}) does not match audio files count ({len(sources)})."
                )
            for src, text in zip(sources, transcripts, strict=True):
                if text and str(text).strip():
                    pairs.append((src, str(text).strip()))
        elif auto_transcribe:
            pairs = _auto_transcribe_clips(sources, language=language)
        else:
            raise TtsError(
                "Provide transcripts, metadata_csv, or set auto_transcribe=True (requires faster-whisper)."
            )
    else:
        raise TtsError("Provide either metadata_csv or audio_paths for training dataset preparation.")

    if not pairs:
        raise TtsError("No valid (audio, transcript) pairs found for training.")

    entries: list[tuple[str, str, str]] = []
    total_duration_s = 0.0
    for idx, (src, text) in enumerate(pairs):
        dst_name = f"clip_{idx:05d}.wav"
        dst_path = os.path.join(wavs_dir, dst_name)
        convert_audio(src, dst_path, sample_rate=22050, channels=1)
        info = probe_audio(dst_path)
        if info.duration_ms < 350:
            continue
        total_duration_s += info.duration_s
        entries.append((f"wavs/{dst_name}", " ".join(text.split()), speaker_name))

    if not entries:
        raise TtsError("All audio clips were shorter than 0.35s or invalid.")
    if not allow_short and total_duration_s < min_duration_s:
        raise TtsError(
            f"Total training audio is {total_duration_s:.1f}s; at least {min_duration_s:.0f}s "
            f"(2+ minutes) of speech is recommended for XTTS fine-tuning."
        )

    shuffled = list(entries)
    random.Random(42).shuffle(shuffled)
    if len(shuffled) == 1:
        train_rows = shuffled
        eval_rows = shuffled
    else:
        eval_count = max(1, round(len(shuffled) * eval_ratio))
        eval_count = min(eval_count, len(shuffled) - 1)
        eval_rows = sorted(shuffled[:eval_count], key=lambda r: r[0])
        train_rows = sorted(shuffled[eval_count:], key=lambda r: r[0])

    train_csv = os.path.join(output_dir, "metadata_train.csv")
    eval_csv = os.path.join(output_dir, "metadata_eval.csv")
    for path, rows in ((train_csv, train_rows), (eval_csv, eval_rows)):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="|")
            writer.writerow(["audio_file", "text", "speaker_name"])
            writer.writerows(rows)

    return {
        "dataset_dir": output_dir,
        "train_csv": train_csv,
        "eval_csv": eval_csv,
        "total_clips": len(entries),
        "train_clips": len(train_rows),
        "eval_clips": len(eval_rows),
        "total_duration_s": round(total_duration_s, 2),
    }


def _auto_transcribe_clips(
    sources: _t.Sequence[str], language: str
) -> list[tuple[str, str]]:
    if importlib.util.find_spec("faster_whisper") is None:
        raise TtsError(
            "auto_transcribe=True requires `faster-whisper` (`pip install faster-whisper`). "
            "Alternatively, pass `transcripts` or `metadata_csv`."
        )
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    asr = WhisperModel("base", device=device, compute_type=compute_type)
    pairs: list[tuple[str, str]] = []
    for src in sources:
        segments, _ = asr.transcribe(src, language=language.split("-")[0])
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        if text:
            pairs.append((src, text))
    del asr
    gc.collect()
    return pairs


def train_custom_voice(
    profile_name: str,
    audio_paths: _t.Optional[_t.Sequence[str] | str] = None,
    dataset_dir: _t.Optional[str] = None,
    transcripts: _t.Optional[_t.Union[dict[str, str], _t.Sequence[str]]] = None,
    metadata_csv: _t.Optional[str] = None,
    language: str = "en",
    epochs: int = 10,
    batch_size: _t.Optional[int] = None,
    grad_accum_steps: _t.Optional[int] = None,
    learning_rate: float = 5e-6,
    max_audio_seconds: float = 11.0,
    auto_transcribe: bool = False,
    allow_cpu: bool = False,
    allow_short: bool = False,
    dry_run: bool = False,
) -> dict[str, _t.Any]:
    """Fine-tune XTTS v2 on a user's voice dataset and register a VoiceProfile."""
    profile_id = slugify_profile(profile_name)
    hw = check_hardware_requirements()
    train_tier = hw["tiers"]["xtts_finetuning"]

    if not (hw["torch_installed"] and hw["coqui_tts_installed"] and hw["trainer_installed"]):
        raise TtsError(
            "XTTS fine-tuning requires `torch`, `coqui-tts`, and `coqui-tts-trainer` (see requirements-xtts.txt)."
        )
    if not hw["cuda_available"] and not allow_cpu:
        raise TtsError(
            "No CUDA GPU detected. Fine-tuning XTTS v2 requires an NVIDIA CUDA GPU (>= 8 GB VRAM recommended). "
            "Use `enroll_voice_profile` for fast CPU/GPU voice cloning, or pass allow_cpu=True to force CPU training."
        )

    pdir = _profile_dir(profile_id)
    work_dataset = dataset_dir or os.path.join(pdir, "dataset")
    if dataset_dir and os.path.isfile(os.path.join(dataset_dir, "metadata_train.csv")):
        train_csv = os.path.join(dataset_dir, "metadata_train.csv")
        eval_csv = os.path.join(dataset_dir, "metadata_eval.csv")
        if not os.path.isfile(eval_csv):
            eval_csv = train_csv
        ds_info = {
            "dataset_dir": dataset_dir,
            "train_csv": train_csv,
            "eval_csv": eval_csv,
            "total_clips": sum(1 for _ in open(train_csv, encoding="utf-8")) - 1,
            "total_duration_s": 0.0,
        }
    else:
        ds_info = prepare_training_dataset(
            output_dir=work_dataset,
            audio_paths=audio_paths,
            transcripts=transcripts,
            metadata_csv=metadata_csv,
            language=language,
            speaker_name=profile_id,
            auto_transcribe=auto_transcribe,
            allow_short=allow_short,
        )
        train_csv = ds_info["train_csv"]
        eval_csv = ds_info["eval_csv"]

    # Choose conservative VRAM settings for 8 GB GPUs if caller didn't override.
    vram_gb = hw.get("vram_gb", 0.0)
    resolved_batch = batch_size if batch_size is not None else (2 if vram_gb <= 9.0 else 4)
    resolved_accum = grad_accum_steps if grad_accum_steps is not None else (8 if resolved_batch <= 2 else 4)

    if dry_run:
        return {
            "status": "dry_run_ok",
            "profile_id": profile_id,
            "hardware": hw,
            "training_ready": train_tier["ready"] or allow_cpu,
            "dataset": ds_info,
            "hyperparameters": {
                "language": language,
                "epochs": epochs,
                "batch_size": resolved_batch,
                "grad_accum_steps": resolved_accum,
                "learning_rate": learning_rate,
                "max_audio_seconds": max_audio_seconds,
            },
        }

    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import (
        GPTArgs,
        GPTTrainer,
        GPTTrainerConfig,
    )
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.utils.manage import ModelManager

    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    run_root = os.path.join(pdir, "training_run")
    base_files_dir = os.path.join(run_root, "base_xtts")
    os.makedirs(base_files_dir, exist_ok=True)

    # Reuse already cached XTTS v2 model files when present.
    cache_dir = hw["xtts_cache_dir"]
    dvae_ckpt = os.path.join(base_files_dir, "dvae.pth")
    mel_norm = os.path.join(base_files_dir, "mel_stats.pth")
    vocab_file = os.path.join(cache_dir, "vocab.json")
    xtts_ckpt = os.path.join(cache_dir, "model.pth")
    xtts_cfg = os.path.join(cache_dir, "config.json")

    if not os.path.isfile(vocab_file) or not os.path.isfile(xtts_ckpt) or not os.path.isfile(xtts_cfg):
        vocab_file = os.path.join(base_files_dir, "vocab.json")
        xtts_ckpt = os.path.join(base_files_dir, "model.pth")
        xtts_cfg = os.path.join(base_files_dir, "config.json")
        ModelManager._download_model_files(
            [
                "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
                "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
                "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json",
            ],
            base_files_dir,
            progress_bar=False,
        )

    if not os.path.isfile(dvae_ckpt) or not os.path.isfile(mel_norm):
        ModelManager._download_model_files(
            [
                "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
                "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
            ],
            base_files_dir,
            progress_bar=False,
        )

    xtts_lang = "zh-cn" if language.lower() in ("zh", "zh-cn") else language.lower()
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=train_csv,
        meta_file_val=eval_csv,
        language=xtts_lang,
    )

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=44100,
        debug_loading_failures=False,
        max_wav_length=int(max_audio_seconds * 22050),
        max_text_length=200,
        mel_norm_file=mel_norm,
        dvae_checkpoint=dvae_ckpt,
        xtts_checkpoint=xtts_ckpt,
        tokenizer_file=vocab_file,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000
    )
    config = GPTTrainerConfig(
        epochs=int(epochs),
        output_path=run_root,
        model_args=model_args,
        run_name=f"xtts_ft_{profile_id}",
        project_name="tts_app_voice_training",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=resolved_batch,
        batch_group_size=24,
        eval_batch_size=resolved_batch,
        num_loader_workers=2,
        eval_split_max_size=128,
        print_step=25,
        plot_step=100,
        log_model_step=500,
        save_step=500,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=float(learning_rate),
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=resolved_accum,
        ),
        config,
        output_path=run_root,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    exp_dir = trainer.output_path
    ckpt_dir = os.path.join(pdir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model = os.path.join(exp_dir, "best_model.pth")
    if not os.path.isfile(best_model):
        best_model = os.path.join(exp_dir, "checkpoint_0.pth")
    shutil.copy2(best_model, os.path.join(ckpt_dir, "model.pth"))
    shutil.copy2(xtts_cfg, os.path.join(ckpt_dir, "config.json"))
    shutil.copy2(vocab_file, os.path.join(ckpt_dir, "vocab.json"))

    speaker_ref = train_samples[0]["audio_file"]
    ref_copy = os.path.join(ckpt_dir, "speaker_ref.wav")
    shutil.copy2(speaker_ref, ref_copy)

    del model, trainer
    gc.collect()

    profile = VoiceProfile(
        id=profile_id,
        name=profile_name.strip(),
        kind="finetuned",
        language=xtts_lang,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        sample_count=int(ds_info.get("total_clips", len(train_samples))),
        total_source_duration_s=float(ds_info.get("total_duration_s", 0.0)),
        sample_files=[ref_copy],
        checkpoint_dir=ckpt_dir,
        speaker_ref_wav=ref_copy,
        notes=f"Fine-tuned XTTS v2 ({epochs} epochs, lr={learning_rate})",
    )
    save_profile(profile)
    return {
        "status": "trained",
        "profile": profile.to_dict(),
        "dataset": ds_info,
        "checkpoint_dir": ckpt_dir,
    }
