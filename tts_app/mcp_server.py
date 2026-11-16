"""Model Context Protocol (MCP) stdio server for AI agents and video pipelines.

Exposes Text-to-Speech synthesis, multi-scene video narration with SRT/VTT/JSON
timings, FFmpeg video muxing, hardware diagnostics, and user voice enrollment /
XTTS fine-tuning over standard newline-delimited JSON-RPC 2.0 on stdin/stdout.

Run with:
    python -m tts_app.mcp_server
    # or via console script:
    tts-mcp
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import typing as _t

from tts_app.audio_utils import mux_audio_to_video
from tts_app.models import TtsError
from tts_app.tts_engine import available_engines, get_engine, smart_synthesize_async
from tts_app.video_pipeline import generate_video_voiceover_async
from tts_app.voice_profiles import (
    check_hardware_requirements,
    delete_profile,
    enroll_voice_profile,
    get_outputs_dir,
    get_profile,
    list_profiles,
    train_custom_voice,
)

SERVER_INFO = {
    "name": "tts-mcp",
    "title": "Smart TTS, Video Voiceover & Voice Training MCP",
    "version": "2.0.0",
}

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

TOOLS: list[dict[str, _t.Any]] = [
    {
        "name": "tts_status",
        "title": "Check TTS engines, hardware, and voice profiles",
        "description": (
            "Inspect available TTS engines (edge-tts, xtts_v2, pyttsx3), GPU/CUDA/RAM/disk "
            "hardware readiness for inference and voice training, ffmpeg availability, and "
            "saved user voice profiles."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "list_voices",
        "title": "List available TTS voices and user voice profiles",
        "description": (
            "Discover voices across engines (edge-tts, xtts_v2, pyttsx3) plus saved "
            "enrolled or fine-tuned user voice profiles. Filter by engine, language, "
            "gender, or search text."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "engine": {
                    "type": "string",
                    "description": "Engine id ('all', 'edge-tts', 'xtts_v2', 'pyttsx3'). Default: 'all'.",
                    "default": "all",
                },
                "language": {
                    "type": "string",
                    "description": "Language subtag filter (e.g. 'en', 'ar', 'fr').",
                    "default": "",
                },
                "gender": {
                    "type": "string",
                    "description": "Optional gender filter ('Male', 'Female', 'Unknown').",
                    "default": "",
                },
                "search": {
                    "type": "string",
                    "description": "Case-insensitive substring match against voice id/name/locale.",
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 40,
                },
            },
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    {
        "name": "synthesize_speech",
        "title": "Synthesize speech audio and optional captions",
        "description": (
            "Convert text to spoken audio (MP3 or WAV) using smart engine selection, "
            "built-in neural voices, or a user's cloned/trained voice profile. "
            "Can normalize loudness (-16 LUFS), trim silence, and output SRT/VTT/JSON word timings."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text"],
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "output_path": {
                    "type": "string",
                    "description": "Destination audio file path (.mp3 or .wav). Auto-generated if omitted.",
                },
                "format": {
                    "type": "string",
                    "enum": ["mp3", "wav"],
                    "default": "mp3",
                },
                "engine": {
                    "type": "string",
                    "description": "'auto', 'edge-tts', 'xtts_v2', or 'pyttsx3'. Default: 'auto'.",
                    "default": "auto",
                },
                "language": {"type": "string", "default": "en"},
                "voice_id": {"type": "string", "default": ""},
                "gender": {"type": "string", "default": ""},
                "voice_profile": {
                    "type": "string",
                    "description": "Saved user voice profile name/id (uses XTTS v2).",
                },
                "reference_audio": {
                    "type": "string",
                    "description": "Path to a 6-30s voice sample WAV/MP3 for zero-shot XTTS cloning.",
                },
                "speed": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 2.0,
                    "default": 1.0,
                },
                "pitch": {"type": "string", "default": "+0Hz"},
                "trim_silence": {"type": "boolean", "default": False},
                "normalize_loudness": {"type": "boolean", "default": False},
                "generate_subtitles": {"type": "boolean", "default": True},
                "fps": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
                "max_words_per_cue": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 6,
                },
                "offline_only": {"type": "boolean", "default": False},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    {
        "name": "generate_video_voiceover",
        "title": "Generate multi-scene narration, subtitles, and frame timeline for video",
        "description": (
            "Turn a list of scenes or a narration script into per-scene clips, a stitched "
            "loudness-normalized master audio track (-16 LUFS), SRT/VTT captions, and a "
            "frame-accurate .timings.json manifest for Remotion, HyperFrames, or FFmpeg."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Full script text (split on blank lines or '[scene: id]' tags). Required if 'scenes' is omitted.",
                },
                "scenes": {
                    "type": "array",
                    "description": "Explicit scene list with optional per-scene voice, language, speed, or pause overrides.",
                    "items": {
                        "type": "object",
                        "required": ["text"],
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string", "minLength": 1},
                            "pause_after_ms": {"type": "integer", "minimum": 0, "default": 350},
                            "engine": {"type": "string"},
                            "language": {"type": "string"},
                            "voice_id": {"type": "string"},
                            "gender": {"type": "string"},
                            "voice_profile": {"type": "string"},
                            "reference_audio": {"type": "string"},
                            "speed": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                            "pitch": {"type": "string"},
                        },
                    },
                },
                "output_dir": {"type": "string"},
                "output_name": {"type": "string", "default": "voiceover"},
                "audio_format": {
                    "type": "string",
                    "enum": ["mp3", "wav"],
                    "default": "mp3",
                },
                "fps": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
                "engine": {"type": "string", "default": "auto"},
                "language": {"type": "string", "default": "en"},
                "voice_id": {"type": "string", "default": ""},
                "gender": {"type": "string", "default": ""},
                "voice_profile": {"type": "string"},
                "reference_audio": {"type": "string"},
                "speed": {"type": "number", "minimum": 0.5, "maximum": 2.0, "default": 1.0},
                "pitch": {"type": "string", "default": "+0Hz"},
                "default_pause_ms": {"type": "integer", "minimum": 0, "default": 350},
                "trim_silence": {"type": "boolean", "default": True},
                "normalize_loudness": {"type": "boolean", "default": True},
                "max_words_per_cue": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 6,
                },
                "keep_scene_files": {"type": "boolean", "default": True},
                "video_path": {
                    "type": "string",
                    "description": "Optional video file to mux the generated voiceover and subtitles onto.",
                },
                "burn_subtitles": {"type": "boolean", "default": False},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    {
        "name": "mux_audio_to_video",
        "title": "Mux narration audio and subtitles onto a video file",
        "description": (
            "Attach a generated narration track (and optional .srt subtitles, either soft-muxed "
            "or burned in) onto an existing MP4/MOV/WEBM video using FFmpeg."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["video_path", "audio_path", "output_path"],
            "properties": {
                "video_path": {"type": "string"},
                "audio_path": {"type": "string"},
                "output_path": {"type": "string"},
                "subtitle_path": {"type": "string"},
                "burn_subtitles": {"type": "boolean", "default": False},
                "replace_audio": {"type": "boolean", "default": True},
                "bg_audio_volume": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.2,
                },
                "fit_mode": {
                    "type": "string",
                    "enum": ["keep_video", "shortest"],
                    "default": "keep_video",
                },
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "enroll_voice_profile",
        "title": "Enroll a reusable voice profile from user voice samples",
        "description": (
            "Preprocess one or more voice recordings (WAV/MP3/FLAC/OGG/M4A) and compute "
            "XTTS v2 speaker conditioning latents saved as a named voice profile. Fast on both "
            "CPU and GPU; no full training required."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["profile_name", "audio_paths"],
            "properties": {
                "profile_name": {"type": "string", "minLength": 1},
                "audio_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "One or more paths to audio files or directories of voice samples.",
                },
                "language": {"type": "string", "default": "en"},
                "preprocess": {"type": "boolean", "default": True},
                "notes": {"type": "string", "default": ""},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "train_custom_voice",
        "title": "Fine-tune XTTS v2 on a user's voice dataset",
        "description": (
            "Prepare a training dataset and fine-tune XTTS v2 GPT weights on the user's voice "
            "(requires NVIDIA CUDA GPU >= 8 GB VRAM unless allow_cpu=True). Use dry_run=True "
            "first to verify hardware and dataset readiness."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["profile_name"],
            "properties": {
                "profile_name": {"type": "string", "minLength": 1},
                "audio_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "dataset_dir": {"type": "string"},
                "metadata_csv": {"type": "string"},
                "transcripts": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Mapping from audio filename/path to its exact spoken transcript.",
                },
                "language": {"type": "string", "default": "en"},
                "epochs": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 64},
                "grad_accum_steps": {"type": "integer", "minimum": 1, "maximum": 64},
                "learning_rate": {"type": "number", "default": 5e-6},
                "auto_transcribe": {"type": "boolean", "default": False},
                "allow_cpu": {"type": "boolean", "default": False},
                "allow_short": {"type": "boolean", "default": False},
                "dry_run": {"type": "boolean", "default": False},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "manage_voice_profiles",
        "title": "List, inspect, or delete saved user voice profiles",
        "description": "Manage enrolled and fine-tuned custom voice profiles.",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "inspect", "delete"],
                },
                "profile_name": {"type": "string"},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
]

RESOURCES = [
    {
        "uri": "tts://status",
        "name": "TTS System & Hardware Status",
        "description": "Live capability report for engines, GPU/CUDA/CPU hardware, and ffmpeg.",
        "mimeType": "application/json",
    },
    {
        "uri": "tts://voices/profiles",
        "name": "Saved User Voice Profiles",
        "description": "List of enrolled and fine-tuned custom voice profiles.",
        "mimeType": "application/json",
    },
    {
        "uri": "tts://guides/hardware-and-training",
        "name": "Hardware Requirements & Voice Training Guide",
        "description": "Minimum hardware specs and instructions for voice enrollment and fine-tuning.",
        "mimeType": "text/markdown",
    },
]

PROMPTS = [
    {
        "name": "video_narration_script",
        "description": "Structure a video script into timed scenes and synthesize voiceover + subtitles.",
        "arguments": [
            {"name": "topic", "description": "Video topic or brief", "required": True},
            {"name": "fps", "description": "Target frame rate (24, 30, 60)", "required": False},
            {"name": "voice_profile", "description": "Optional user voice profile name", "required": False},
        ],
    },
    {
        "name": "voice_training_checklist",
        "description": "Guide the user through recording, enrolling, or fine-tuning their own voice.",
        "arguments": [
            {"name": "mode", "description": "'enroll' (fast) or 'finetune' (full training)", "required": False},
        ],
    },
]

_HARDWARE_GUIDE_MD = """# Hardware Requirements & Voice Training Guide

## Minimum Hardware Tiers

1. **Standard TTS & Video Narration (`edge-tts` / `pyttsx3`)**
   - CPU: Dual-core x86_64/ARM64 | RAM: 512 MB+ | Disk: 200 MB | GPU: None
   - Optional: `ffmpeg` + `ffprobe` for MP3/WAV transcoding, -16 LUFS normalization, and video muxing.

2. **Local Voice Cloning & Enrolled Profiles (`XTTS v2` Inference)**
   - CPU minimum: Quad-core CPU, 8 GB RAM, 5 GB free disk
   - GPU recommended: NVIDIA CUDA GPU with 4 GB+ VRAM (or Apple Silicon MPS)

3. **Full Custom Voice Training (`train_custom_voice`)**
   - GPU required: NVIDIA CUDA GPU with >= 8 GB VRAM (12 GB+ recommended)
   - RAM: 16 GB+ | Disk: 10+ GB free
   - Dataset: >= 2 minutes (10-30 minutes ideal) of clean mono speech with transcripts
"""


def _tool_result(
    summary: str, data: _t.Optional[dict[str, _t.Any]] = None, is_error: bool = False
) -> dict[str, _t.Any]:
    payload: dict[str, _t.Any] = {
        "content": [{"type": "text", "text": summary}],
        "isError": is_error,
    }
    if data is not None:
        payload["structuredContent"] = data
    return payload


def _status_payload() -> dict[str, _t.Any]:
    engines = []
    for eng in available_engines():
        engines.append(
            {
                "id": eng.id,
                "label": eng.label,
                "supports_cloning": eng.supports_cloning,
            }
        )
    profiles = [p.to_dict() for p in list_profiles()]
    return {
        "engines": engines,
        "hardware": check_hardware_requirements(),
        "voice_profiles": profiles,
        "outputs_dir": get_outputs_dir(),
    }


async def _call_tool(name: str, args: _t.Optional[dict[str, _t.Any]]) -> dict[str, _t.Any]:
    args = args or {}

    if name == "tts_status":
        data = _status_payload()
        eng_ids = ", ".join(e["id"] for e in data["engines"])
        hw = data["hardware"]
        gpu_info = f"{hw['gpu_name']} ({hw['vram_gb']} GB)" if hw["cuda_available"] else "CPU-only"
        summary = (
            f"Engines: {eng_ids} | Compute: {gpu_info} | "
            f"FFmpeg: {hw['ffmpeg_installed']} | Saved profiles: {len(data['voice_profiles'])}"
        )
        return _tool_result(summary, data)

    if name == "list_voices":
        engine_filter = str(args.get("engine", "all")).strip().lower()
        language = str(args.get("language", "")).strip().lower()
        gender = str(args.get("gender", "")).strip().lower()
        search = str(args.get("search", "")).strip().lower()
        limit = int(args.get("limit", 40))

        target_engines = (
            available_engines()
            if engine_filter in ("", "all")
            else [get_engine(engine_filter)]
        )
        items: list[dict[str, _t.Any]] = []
        for eng in target_engines:
            for v in eng.list_voices(language):
                if gender and v.gender.lower() != gender:
                    continue
                if search and not (
                    search in v.id.lower()
                    or search in v.name.lower()
                    or search in v.locale.lower()
                ):
                    continue
                items.append(v.to_dict())
                if len(items) >= limit:
                    break
            if len(items) >= limit:
                break

        profiles = [p.to_dict() for p in list_profiles()]
        data = {"voices": items, "count": len(items), "voice_profiles": profiles}
        return _tool_result(
            f"Found {len(items)} voices and {len(profiles)} saved voice profiles.",
            data,
        )

    if name == "synthesize_speech":
        text = str(args.get("text", "")).strip()
        if not text:
            return _tool_result("`text` is required and cannot be empty.", is_error=True)

        fmt = str(args.get("format", "mp3")).lower().lstrip(".")
        if fmt not in ("mp3", "wav"):
            return _tool_result("`format` must be 'mp3' or 'wav'.", is_error=True)

        output_path = args.get("output_path")
        if not output_path:
            out_dir = get_outputs_dir()
            os.makedirs(out_dir, exist_ok=True)
            import time

            output_path = os.path.join(out_dir, f"speech_{int(time.time() * 1000)}.{fmt}")
        else:
            output_path = os.path.abspath(str(output_path))

        res = await smart_synthesize_async(
            text=text,
            output_path=output_path,
            engine=str(args.get("engine", "auto")),
            language=str(args.get("language", "en")),
            voice_id=str(args.get("voice_id", "")),
            gender=str(args.get("gender", "")),
            voice_profile=args.get("voice_profile"),
            reference_audio=args.get("reference_audio"),
            speed=float(args.get("speed", 1.0)),
            pitch=str(args.get("pitch", "+0Hz")),
            trim_silence_flag=bool(args.get("trim_silence", False)),
            normalize_loudness_flag=bool(args.get("normalize_loudness", False)),
            generate_subtitles=bool(args.get("generate_subtitles", True)),
            fps=int(args.get("fps", 30)),
            max_words_per_cue=int(args.get("max_words_per_cue", 6)),
            offline_only=bool(args.get("offline_only", False)),
        )
        data = res.to_dict()
        summary = (
            f"Synthesized {res.duration_s}s {res.format.upper()} via {res.engine} "
            f"({res.voice_id}) -> {res.audio_path}"
        )
        if res.srt_path:
            summary += f"\nSubtitles: {res.srt_path} | Timings: {res.timings_path}"
        return _tool_result(summary, data)

    if name == "generate_video_voiceover":
        data = await generate_video_voiceover_async(
            scenes=args.get("scenes"),
            script=args.get("script"),
            output_dir=args.get("output_dir"),
            output_name=str(args.get("output_name", "voiceover")),
            audio_format=str(args.get("audio_format", "mp3")),
            fps=int(args.get("fps", 30)),
            default_engine=str(args.get("engine", "auto")),
            default_language=str(args.get("language", "en")),
            default_voice_id=str(args.get("voice_id", "")),
            default_gender=str(args.get("gender", "")),
            default_voice_profile=args.get("voice_profile"),
            default_reference_audio=args.get("reference_audio"),
            default_speed=float(args.get("speed", 1.0)),
            default_pitch=str(args.get("pitch", "+0Hz")),
            default_pause_ms=int(args.get("default_pause_ms", 350)),
            trim_silence_flag=bool(args.get("trim_silence", True)),
            normalize_loudness_flag=bool(args.get("normalize_loudness", True)),
            max_words_per_cue=int(args.get("max_words_per_cue", 6)),
            keep_scene_files=bool(args.get("keep_scene_files", True)),
            video_path=args.get("video_path"),
            burn_subtitles=bool(args.get("burn_subtitles", False)),
        )
        summary = (
            f"Generated {data['scene_count']}-scene video voiceover ({data['total_duration_s']}s / "
            f"{data['total_frames']} frames @ {data['fps']} fps): {data['master_audio_path']}\n"
            f"Timeline: {data['timeline_json_path']} | SRT: {data['srt_path']}"
        )
        if data.get("muxed_video_path"):
            summary += f"\nMuxed video: {data['muxed_video_path']}"
        return _tool_result(summary, data)

    if name == "mux_audio_to_video":
        out = mux_audio_to_video(
            video_path=str(args["video_path"]),
            audio_path=str(args["audio_path"]),
            output_path=str(args["output_path"]),
            subtitle_path=args.get("subtitle_path"),
            burn_subtitles=bool(args.get("burn_subtitles", False)),
            replace_audio=bool(args.get("replace_audio", True)),
            bg_audio_volume=float(args.get("bg_audio_volume", 0.2)),
            fit_mode=str(args.get("fit_mode", "keep_video")),
        )
        return _tool_result(f"Muxed video written to {out}", {"output_path": out})

    if name == "enroll_voice_profile":
        prof = await asyncio.to_thread(
            enroll_voice_profile,
            name=str(args["profile_name"]),
            audio_paths=list(args["audio_paths"]),
            language=str(args.get("language", "en")),
            preprocess=bool(args.get("preprocess", True)),
            notes=str(args.get("notes", "")),
        )
        data = prof.to_dict()
        return _tool_result(
            f"Enrolled voice profile '{prof.name}' (id={prof.id}, {prof.sample_count} samples, {prof.total_source_duration_s}s).",
            data,
        )

    if name == "train_custom_voice":
        data = await asyncio.to_thread(
            train_custom_voice,
            profile_name=str(args["profile_name"]),
            audio_paths=args.get("audio_paths"),
            dataset_dir=args.get("dataset_dir"),
            transcripts=args.get("transcripts"),
            metadata_csv=args.get("metadata_csv"),
            language=str(args.get("language", "en")),
            epochs=int(args.get("epochs", 10)),
            batch_size=int(args["batch_size"]) if args.get("batch_size") is not None else None,
            grad_accum_steps=(
                int(args["grad_accum_steps"])
                if args.get("grad_accum_steps") is not None
                else None
            ),
            learning_rate=float(args.get("learning_rate", 5e-6)),
            auto_transcribe=bool(args.get("auto_transcribe", False)),
            allow_cpu=bool(args.get("allow_cpu", False)),
            allow_short=bool(args.get("allow_short", False)),
            dry_run=bool(args.get("dry_run", False)),
        )
        return _tool_result(
            f"Voice training status for '{args['profile_name']}': {data.get('status')}",
            data,
        )

    if name == "manage_voice_profiles":
        action = str(args.get("action", "list"))
        if action == "list":
            profs = [p.to_dict() for p in list_profiles()]
            return _tool_result(
                f"{len(profs)} saved voice profile(s).", {"profiles": profs}
            )
        target = str(args.get("profile_name", "")).strip()
        if not target:
            return _tool_result("`profile_name` is required for inspect/delete.", is_error=True)
        if action == "inspect":
            prof = get_profile(target)
            return _tool_result(
                f"Profile '{prof.name}' ({prof.kind}, {prof.language}, {prof.total_source_duration_s}s)",
                prof.to_dict(),
            )
        if action == "delete":
            deleted = delete_profile(target)
            return _tool_result(
                f"Deleted voice profile '{target}'.", {"deleted": deleted, "profile_name": target}
            )
        return _tool_result(f"Unsupported action: {action}", is_error=True)

    return _tool_result(f"Unknown tool: {name}", is_error=True)


class McpServer:
    """Zero-dependency stdio JSON-RPC 2.0 MCP server."""

    async def handle_message(self, message: dict[str, _t.Any]) -> _t.Optional[dict[str, _t.Any]]:
        msg_id = message.get("id")
        method = message.get("method", "")
        params = message.get("params") or {}

        # Notifications carry no id and expect no reply.
        if msg_id is None:
            return None

        def _ok(result: dict[str, _t.Any]) -> dict[str, _t.Any]:
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}

        def _err(code: int, text: str) -> dict[str, _t.Any]:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}}

        if method == "initialize":
            req_ver = params.get("protocolVersion", SUPPORTED_PROTOCOLS[0])
            proto = req_ver if req_ver in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
            return _ok(
                {
                    "protocolVersion": proto,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": SERVER_INFO,
                    "instructions": (
                        "Use tts_status to inspect engines and hardware, list_voices to discover voices, "
                        "synthesize_speech for single voice clips, generate_video_voiceover for multi-scene "
                        "video narration + SRT/VTT/timings.json, and enroll_voice_profile / train_custom_voice "
                        "to personalize speech with the user's own voice."
                    ),
                }
            )

        if method == "ping":
            return _ok({})

        if method == "tools/list":
            return _ok({"tools": TOOLS})

        if method == "tools/call":
            tool_name = params.get("name", "")
            raw_args = params.get("arguments")
            if raw_args is None:
                tool_args = {}
            elif not isinstance(raw_args, dict):
                return _ok(_tool_result("Tool `arguments` must be a JSON object.", is_error=True))
            else:
                tool_args = raw_args
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = await _call_tool(tool_name, tool_args)
            except (TtsError, ValueError, KeyError) as exc:
                result = _tool_result(str(exc), is_error=True)
            except Exception as exc:
                result = _tool_result(f"Tool '{tool_name}' failed: {exc}", is_error=True)
            return _ok(result)

        if method == "resources/list":
            return _ok({"resources": RESOURCES})

        if method == "resources/read":
            uri = params.get("uri", "")
            if uri == "tts://status":
                return _ok(
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(_status_payload(), indent=2),
                            }
                        ]
                    }
                )
            if uri == "tts://voices/profiles":
                return _ok(
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(
                                    [p.to_dict() for p in list_profiles()], indent=2
                                ),
                            }
                        ]
                    }
                )
            if uri == "tts://guides/hardware-and-training":
                return _ok(
                    {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/markdown",
                                "text": _HARDWARE_GUIDE_MD,
                            }
                        ]
                    }
                )
            return _err(-32002, f"Resource not found: {uri}")

        if method == "prompts/list":
            return _ok({"prompts": PROMPTS})

        if method == "prompts/get":
            pname = params.get("name", "")
            pargs = params.get("arguments") or {}
            if pname == "video_narration_script":
                topic = pargs.get("topic", "product demo")
                fps = pargs.get("fps", 30)
                vp = pargs.get("voice_profile", "")
                vp_note = f" using voice_profile='{vp}'" if vp else ""
                return _ok(
                    {
                        "description": "Generate a scene-by-scene video narration and timeline.",
                        "messages": [
                            {
                                "role": "user",
                                "content": {
                                    "type": "text",
                                    "text": (
                                        f"Write a concise multi-scene narration script about '{topic}', "
                                        f"then call `generate_video_voiceover` at {fps} fps{vp_note} "
                                        "with normalize_loudness=true and trim_silence=true to produce "
                                        "the master audio track, SRT captions, and .timings.json."
                                    ),
                                },
                            }
                        ],
                    }
                )
            if pname == "voice_training_checklist":
                mode = pargs.get("mode", "enroll")
                return _ok(
                    {
                        "description": "Checklist for enrolling or fine-tuning a custom user voice.",
                        "messages": [
                            {
                                "role": "user",
                                "content": {
                                    "type": "text",
                                    "text": (
                                        f"First call `tts_status` to check GPU/VRAM readiness. "
                                        f"For mode='{mode}': use `enroll_voice_profile` with 6-30s of clean "
                                        "speech for fast cloning, or prepare >= 2 minutes of labeled clips "
                                        "and call `train_custom_voice` (dry_run=true first) on an >= 8 GB CUDA GPU."
                                    ),
                                },
                            }
                        ],
                    }
                )
            return _err(-32602, f"Unknown prompt: {pname}")

        return _err(-32601, f"Method not found: {method}")


async def serve_stdio(
    stdin: _t.Optional[_t.TextIO] = None,
    stdout: _t.Optional[_t.TextIO] = None,
) -> int:
    """Run the JSON-RPC 2.0 MCP loop reading from stdin and writing to stdout."""
    in_stream = stdin or sys.stdin
    proto_out = stdout or sys.stdout
    # Prevent any third-party library print() from corrupting JSON-RPC stdout framing.
    if stdout is None:
        sys.stdout = sys.stderr

    server = McpServer()
    while True:
        raw_line = in_stream.readline()
        if not raw_line:
            break
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            err_reply = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            proto_out.write(json.dumps(err_reply) + "\n")
            proto_out.flush()
            continue

        reply = await server.handle_message(msg)
        if reply is not None:
            proto_out.write(json.dumps(reply, ensure_ascii=False) + "\n")
            proto_out.flush()
    return 0


def main() -> int:
    """Console entry point (`tts-mcp`)."""
    return asyncio.run(serve_stdio())


if __name__ == "__main__":
    sys.exit(main())
