"""Video narration & timeline pipeline for AI video generation workflows.

Synthesizes single scripts or multi-scene storyboards into:
  - Per-scene audio clips (trimmed & loudness-matched)
  - A stitched master narration track (MP3 or WAV, default -16 LUFS)
  - Frame-accurate JSON timeline & word timings (for Remotion / HyperFrames)
  - SubRip (.srt) and WebVTT (.vtt) caption files
  - Optional FFmpeg muxing onto an existing video file
"""

from __future__ import annotations

import asyncio
import os
import re
import typing as _t
from dataclasses import dataclass

from tts_app.audio_utils import (
    concat_audio_segments,
    mux_audio_to_video,
    normalize_loudness,
    probe_audio,
    safe_remove,
    trim_silence,
)
from tts_app.models import TtsError, WordTiming
from tts_app.subtitles import (
    estimate_word_timings,
    shift_word_timings,
    write_subtitle_artifacts,
)
from tts_app.tts_engine import _run_coro_sync, smart_synthesize_async
from tts_app.voice_profiles import get_outputs_dir

_SCENE_TAG_RE = re.compile(r"^\s*\[scene\s*:\s*([^\]]+)\]\s*$", re.IGNORECASE)


@dataclass
class SceneInput:
    """One narrated scene inside a video storyboard."""

    id: str
    text: str
    pause_after_ms: int = 350
    engine: str = ""
    language: str = ""
    voice_id: str = ""
    gender: str = ""
    voice_profile: _t.Optional[str] = None
    reference_audio: _t.Optional[_t.Union[str, _t.Sequence[str]]] = None
    speed: _t.Optional[float] = None
    pitch: _t.Optional[str] = None


def split_script_to_scenes(
    script: str, default_pause_ms: int = 400
) -> list[SceneInput]:
    """Parse a raw script or ``[scene: name]``-annotated script into scenes."""
    cleaned = (script or "").strip()
    if not cleaned:
        raise TtsError("Script text cannot be empty.")

    lines = cleaned.splitlines()
    has_tags = any(_SCENE_TAG_RE.match(line) for line in lines)

    scenes: list[SceneInput] = []
    if has_tags:
        current_id = "scene_01"
        buf: list[str] = []
        idx = 1
        for line in lines:
            m = _SCENE_TAG_RE.match(line)
            if m:
                body = "\n".join(buf).strip()
                if body:
                    scenes.append(
                        SceneInput(
                            id=current_id,
                            text=body,
                            pause_after_ms=default_pause_ms,
                        )
                    )
                    idx += 1
                raw_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", m.group(1).strip()).strip("_")
                current_id = raw_id or f"scene_{idx:02d}"
                buf = []
            else:
                buf.append(line)
        tail = "\n".join(buf).strip()
        if tail:
            scenes.append(
                SceneInput(id=current_id, text=tail, pause_after_ms=default_pause_ms)
            )
    else:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]
        for i, para in enumerate(paragraphs, start=1):
            scenes.append(
                SceneInput(
                    id=f"scene_{i:02d}",
                    text=" ".join(para.splitlines()),
                    pause_after_ms=default_pause_ms if i < len(paragraphs) else 0,
                )
            )

    if not scenes:
        raise TtsError("No scenes could be parsed from the provided script.")
    return scenes


def _realign_trimmed_words(
    words: list[WordTiming], raw_duration_ms: int, final_duration_ms: int
) -> list[WordTiming]:
    """Adjust word boundary offsets when leading/trailing silence was trimmed."""
    if not words or final_duration_ms <= 0:
        return words
    delta = max(0, raw_duration_ms - final_duration_ms)
    lead_shift = 0
    if delta > 0 and words[0].start_ms > 20:
        lead_shift = min(words[0].start_ms - 15, delta)

    adjusted: list[WordTiming] = []
    for w in words:
        s = max(0, min(final_duration_ms - 1, w.start_ms - lead_shift))
        e = max(s + 1, min(final_duration_ms, w.end_ms - lead_shift))
        adjusted.append(WordTiming(text=w.text, start_ms=s, end_ms=e, scene_id=w.scene_id))
    return adjusted


async def generate_video_voiceover_async(
    *,
    scenes: _t.Optional[_t.Sequence[_t.Union[SceneInput, dict[str, _t.Any]]]] = None,
    script: _t.Optional[str] = None,
    output_dir: _t.Optional[str] = None,
    output_name: str = "voiceover",
    audio_format: str = "mp3",
    fps: int = 30,
    default_engine: str = "auto",
    default_language: str = "en",
    default_voice_id: str = "",
    default_gender: str = "",
    default_voice_profile: _t.Optional[str] = None,
    default_reference_audio: _t.Optional[_t.Union[str, _t.Sequence[str]]] = None,
    default_speed: float = 1.0,
    default_pitch: str = "+0Hz",
    default_pause_ms: int = 350,
    trim_silence_flag: bool = True,
    normalize_loudness_flag: bool = True,
    max_words_per_cue: int = 6,
    max_chars_per_cue: int = 42,
    keep_scene_files: bool = True,
    video_path: _t.Optional[str] = None,
    burn_subtitles: bool = False,
    replace_video_audio: bool = True,
    bg_audio_volume: float = 0.2,
) -> dict[str, _t.Any]:
    """Synthesize a multi-scene video narration track, subtitles, and timeline."""
    parsed_scenes: list[SceneInput] = []
    if scenes:
        for idx, item in enumerate(scenes, start=1):
            if isinstance(item, SceneInput):
                parsed_scenes.append(item)
            else:
                text = str(item.get("text", "")).strip()
                if not text:
                    raise TtsError(f"Scene #{idx} has empty text.")
                raw_id = str(item.get("id") or f"scene_{idx:02d}")
                safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_id).strip("_") or f"scene_{idx:02d}"
                parsed_scenes.append(
                    SceneInput(
                        id=safe_id,
                        text=text,
                        pause_after_ms=int(
                            item.get(
                                "pause_after_ms",
                                default_pause_ms if idx < len(scenes) else 0,
                            )
                        ),
                        engine=str(item.get("engine", "")),
                        language=str(item.get("language", "")),
                        voice_id=str(item.get("voice_id", "")),
                        gender=str(item.get("gender", "")),
                        voice_profile=item.get("voice_profile"),
                        reference_audio=item.get("reference_audio"),
                        speed=float(item["speed"]) if item.get("speed") is not None else None,
                        pitch=str(item["pitch"]) if item.get("pitch") is not None else None,
                    )
                )
    elif script:
        parsed_scenes = split_script_to_scenes(script, default_pause_ms=default_pause_ms)
    else:
        raise TtsError("Provide either `scenes` or `script`.")

    fmt = audio_format.lower().lstrip(".")
    if fmt not in ("mp3", "wav"):
        raise TtsError(f"Unsupported audio format '{audio_format}'. Use 'mp3' or 'wav'.")

    out_dir = os.path.abspath(output_dir or os.path.join(get_outputs_dir(), output_name))
    scenes_dir = os.path.join(out_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    net_sem = asyncio.Semaphore(4)
    local_lock = asyncio.Lock()

    async def _render_scene(idx: int, sc: SceneInput):
        scene_file = os.path.join(scenes_dir, f"{idx:03d}_{sc.id}.wav")
        sc_engine = (sc.engine or default_engine or "auto").strip().lower()
        sc_profile = sc.voice_profile if sc.voice_profile is not None else default_voice_profile
        sc_ref = sc.reference_audio if sc.reference_audio is not None else default_reference_audio
        is_local = (
            sc_profile is not None
            or sc_ref is not None
            or sc_engine in ("xtts", "xtts_v2", "xtts-v2", "pyttsx3", "offline")
        )

        async def _synth_once():
            return await smart_synthesize_async(
                text=sc.text,
                output_path=scene_file,
                engine=sc_engine,
                language=sc.language or default_language,
                voice_id=sc.voice_id or default_voice_id,
                gender=sc.gender or default_gender,
                voice_profile=sc_profile,
                reference_audio=sc_ref,
                speed=sc.speed if sc.speed is not None else default_speed,
                pitch=sc.pitch if sc.pitch is not None else default_pitch,
                trim_silence_flag=False,
                normalize_loudness_flag=False,
                generate_subtitles=False,
                fps=fps,
            )

        if is_local:
            async with local_lock:
                res = await _synth_once()
        else:
            async with net_sem:
                res = await _synth_once()

        raw_dur = res.duration_ms
        if trim_silence_flag:
            await asyncio.to_thread(trim_silence, scene_file)
        if normalize_loudness_flag:
            await asyncio.to_thread(normalize_loudness, scene_file)

        post_info = probe_audio(scene_file)
        post_dur = post_info.duration_ms or raw_dur
        if trim_silence_flag and res.timing_mode == "word_boundary":
            res.word_timings = _realign_trimmed_words(res.word_timings, raw_dur, post_dur)
        elif not res.word_timings or res.timing_mode != "word_boundary":
            res.word_timings = estimate_word_timings(sc.text, 0, post_dur, scene_id=sc.id)
        return sc, res, scene_file, post_dur

    rendered = await asyncio.gather(
        *(_render_scene(idx, sc) for idx, sc in enumerate(parsed_scenes, start=1))
    )
    scene_results = list(rendered)
    concat_inputs: list[tuple[str, int]] = [
        (scene_file, max(0, sc.pause_after_ms))
        for sc, _res, scene_file, _dur in scene_results
    ]
    warnings: list[str] = [
        w for _sc, res, _sf, _dur in scene_results for w in res.warnings
    ]

    master_stem = os.path.join(out_dir, output_name)
    master_audio_path = f"{master_stem}.{fmt}"
    _, seg_timings = concat_audio_segments(concat_inputs, master_audio_path)
    if normalize_loudness_flag:
        normalize_loudness(master_audio_path)

    master_info = probe_audio(master_audio_path)
    total_duration_ms = master_info.duration_ms or (
        seg_timings[-1]["end_ms"] + seg_timings[-1]["pause_after_ms"] if seg_timings else 0
    )

    all_words: list[WordTiming] = []
    timeline_scenes: list[dict[str, _t.Any]] = []
    timing_modes = set()

    for idx, ((sc, res, scene_file, _post_dur), seg) in enumerate(
        zip(scene_results, seg_timings, strict=True), start=1
    ):
        timing_modes.add(res.timing_mode)
        shifted = shift_word_timings(res.word_timings, seg["start_ms"], scene_id=sc.id)
        all_words.extend(shifted)

        start_frame = round(seg["start_ms"] * fps / 1000.0)
        end_frame = round(seg["end_ms"] * fps / 1000.0)
        timeline_scenes.append(
            {
                "index": idx,
                "id": sc.id,
                "text": sc.text,
                "audio_path": scene_file if keep_scene_files else None,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "duration_ms": seg["duration_ms"],
                "pause_after_ms": seg["pause_after_ms"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": max(1, end_frame - start_frame),
                "engine": res.engine,
                "voice_id": res.voice_id,
                "voice_name": res.voice_name,
                "language": res.language,
            }
        )

    if not keep_scene_files:
        for _, _, sf, _ in scene_results:
            safe_remove(sf)

    overall_timing_mode = (
        "word_boundary"
        if timing_modes == {"word_boundary"}
        else "mixed"
        if "word_boundary" in timing_modes
        else "estimated_from_audio"
    )

    srt_path, vtt_path, timeline_path, cues = write_subtitle_artifacts(
        stem_path=master_stem,
        words=all_words,
        duration_ms=total_duration_ms,
        fps=fps,
        timing_mode=overall_timing_mode,
        max_words_per_cue=max_words_per_cue,
        max_chars_per_cue=max_chars_per_cue,
        scenes=timeline_scenes,
        metadata={
            "audio_path": master_audio_path,
            "format": fmt,
            "loudness_normalized": normalize_loudness_flag,
            "silence_trimmed": trim_silence_flag,
        },
    )

    muxed_video_path = None
    if video_path:
        muxed_video_path = f"{master_stem}_video.mp4"
        mux_audio_to_video(
            video_path=video_path,
            audio_path=master_audio_path,
            output_path=muxed_video_path,
            subtitle_path=srt_path,
            burn_subtitles=burn_subtitles,
            replace_audio=replace_video_audio,
            bg_audio_volume=bg_audio_volume,
        )

    return {
        "master_audio_path": master_audio_path,
        "format": fmt,
        "fps": fps,
        "total_duration_ms": total_duration_ms,
        "total_duration_s": round(total_duration_ms / 1000.0, 3),
        "total_frames": round(total_duration_ms * fps / 1000.0),
        "timing_mode": overall_timing_mode,
        "srt_path": srt_path,
        "vtt_path": vtt_path,
        "timeline_json_path": timeline_path,
        "muxed_video_path": muxed_video_path,
        "scene_count": len(timeline_scenes),
        "cue_count": len(cues),
        "word_count": len(all_words),
        "scenes": timeline_scenes,
        "warnings": list(dict.fromkeys(warnings)),
    }


def generate_video_voiceover(**kwargs: _t.Any) -> dict[str, _t.Any]:
    """Synchronous wrapper around :func:`generate_video_voiceover_async`."""
    return _run_coro_sync(lambda: generate_video_voiceover_async(**kwargs))
