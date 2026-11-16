"""Audio inspection, conversion, normalization, concatenation, and video muxing.

Uses ``ffmpeg`` / ``ffprobe`` when available and provides pure-Python WAV
fallbacks for probing, silence trimming, peak normalization, and WAV
concatenation so core workflows remain resilient even in minimal environments.
"""

from __future__ import annotations

import array
import json
import os
import shutil
import subprocess
import tempfile
import typing as _t
import wave
from dataclasses import asdict, dataclass

from tts_app.models import TtsError


@dataclass
class AudioInfo:
    """Metadata returned by :func:`probe_audio`."""

    path: str
    format: str
    duration_ms: int
    sample_rate: int
    channels: int
    size_bytes: int

    @property
    def duration_s(self) -> float:
        return round(self.duration_ms / 1000.0, 3)

    def to_dict(self) -> dict[str, _t.Any]:
        data = asdict(self)
        data["duration_s"] = self.duration_s
        return data


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def _run_tool(cmd: list[str], error_prefix: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise TtsError(f"{error_prefix}: {detail or f'exit code {proc.returncode}'}")
    return proc


def _probe_wav_stdlib(path: str) -> AudioInfo:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or 1
        channels = wf.getnchannels() or 1
    duration_ms = max(0, round(frames * 1000.0 / rate))
    return AudioInfo(
        path=path,
        format="wav",
        duration_ms=duration_ms,
        sample_rate=rate,
        channels=channels,
        size_bytes=os.path.getsize(path),
    )


def probe_audio(path: str) -> AudioInfo:
    """Return duration, sample rate, channels, and size for an audio file."""
    if not path or not os.path.isfile(path):
        raise TtsError(f"Audio file not found: {path}")
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "unknown"
    size_bytes = os.path.getsize(path)

    if has_ffprobe():
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                payload = json.loads(proc.stdout)
                streams = payload.get("streams", [])
                audio_stream = next(
                    (s for s in streams if s.get("codec_type") == "audio"), {}
                )
                fmt = payload.get("format", {})
                dur_str = audio_stream.get("duration") or fmt.get("duration") or "0"
                duration_ms = max(0, round(float(dur_str) * 1000.0))
                sample_rate = int(audio_stream.get("sample_rate") or 0)
                channels = int(audio_stream.get("channels") or 0)
                if duration_ms > 0 or ext != "wav":
                    return AudioInfo(
                        path=path,
                        format=ext,
                        duration_ms=duration_ms,
                        sample_rate=sample_rate or 24000,
                        channels=channels or 1,
                        size_bytes=size_bytes,
                    )
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

    if ext == "wav":
        return _probe_wav_stdlib(path)

    return AudioInfo(
        path=path,
        format=ext,
        duration_ms=0,
        sample_rate=0,
        channels=0,
        size_bytes=size_bytes,
    )


def convert_audio(
    src: str,
    dst: str,
    sample_rate: _t.Optional[int] = None,
    channels: _t.Optional[int] = None,
) -> str:
    """Convert/transcode ``src`` to ``dst`` matching ``dst``'s extension."""
    if not os.path.isfile(src):
        raise TtsError(f"Source audio file not found: {src}")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    src_ext = os.path.splitext(src)[1].lower()
    dst_ext = os.path.splitext(dst)[1].lower()

    if src_ext == dst_ext and sample_rate is None and channels is None:
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        return dst

    if not has_ffmpeg():
        if src_ext == ".wav" and dst_ext == ".wav":
            shutil.copy2(src, dst)
            return dst
        raise TtsError(
            f"ffmpeg is required to convert {src_ext or 'audio'} to {dst_ext}."
        )

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    if channels is not None:
        cmd += ["-ac", str(channels)]

    if dst_ext == ".mp3":
        cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
    elif dst_ext == ".wav":
        cmd += ["-codec:a", "pcm_s16le"]
    elif dst_ext in (".m4a", ".aac"):
        cmd += ["-codec:a", "aac", "-b:a", "192k"]
    elif dst_ext == ".ogg":
        cmd += ["-codec:a", "libvorbis", "-q:a", "5"]

    target = dst
    same_path = os.path.abspath(src) == os.path.abspath(dst)
    if same_path:
        fd, target = tempfile.mkstemp(suffix=dst_ext, prefix="tts_conv_")
        os.close(fd)

    cmd.append(target)
    try:
        _run_tool(cmd, "Audio conversion failed")
        if same_path:
            shutil.move(target, dst)
    except Exception:
        if same_path and os.path.exists(target):
            os.remove(target)
        raise
    return dst


def _trim_silence_wav_stdlib(
    src: str, dst: str, threshold_db: float = -45.0, pad_ms: int = 30
) -> str:
    with wave.open(src, "rb") as wf:
        params = wf.getparams()
        if params.sampwidth != 2:
            shutil.copy2(src, dst)
            return dst
        raw = wf.readframes(params.nframes)

    samples = array.array("h")
    samples.frombytes(raw)
    channels = max(1, params.nchannels)
    frame_count = len(samples) // channels
    if frame_count == 0:
        shutil.copy2(src, dst)
        return dst

    amp_thresh = int(32767 * (10.0 ** (threshold_db / 20.0)))
    first_frame = 0
    last_frame = frame_count - 1

    for i in range(frame_count):
        base = i * channels
        if any(abs(samples[base + c]) > amp_thresh for c in range(channels)):
            first_frame = i
            break
    else:
        shutil.copy2(src, dst)
        return dst

    for i in range(frame_count - 1, -1, -1):
        base = i * channels
        if any(abs(samples[base + c]) > amp_thresh for c in range(channels)):
            last_frame = i
            break

    pad_frames = int(params.framerate * pad_ms / 1000.0)
    start_f = max(0, first_frame - pad_frames)
    end_f = min(frame_count, last_frame + 1 + pad_frames)
    sliced = samples[start_f * channels : end_f * channels]

    with wave.open(dst, "wb") as out:
        out.setparams(params)
        out.writeframes(sliced.tobytes())
    return dst


def trim_silence(
    src: str,
    dst: _t.Optional[str] = None,
    threshold_db: float = -45.0,
    min_silence_s: float = 0.05,
) -> str:
    """Trim leading and trailing silence from an audio file."""
    target = dst or src
    ext = os.path.splitext(target)[1].lower() or os.path.splitext(src)[1].lower()
    in_place = os.path.abspath(src) == os.path.abspath(target)
    work_out = target
    if in_place:
        fd, work_out = tempfile.mkstemp(suffix=ext, prefix="tts_trim_")
        os.close(fd)

    try:
        if has_ffmpeg():
            thresh = f"{threshold_db}dB"
            dur = f"{min_silence_s:.3f}"
            af = (
                f"silenceremove=start_periods=1:start_duration={dur}:start_threshold={thresh},"
                f"areverse,"
                f"silenceremove=start_periods=1:start_duration={dur}:start_threshold={thresh},"
                f"areverse"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                src,
                "-af",
                af,
                work_out,
            ]
            _run_tool(cmd, "Silence trimming failed")
        elif os.path.splitext(src)[1].lower() == ".wav" and ext == ".wav":
            _trim_silence_wav_stdlib(src, work_out, threshold_db=threshold_db)
        else:
            if not in_place:
                shutil.copy2(src, work_out)
            return target

        # Guard against overly aggressive trimming producing an empty clip.
        if not os.path.exists(work_out) or os.path.getsize(work_out) == 0:
            shutil.copy2(src, work_out)
        else:
            info = probe_audio(work_out)
            if info.duration_ms == 0:
                shutil.copy2(src, work_out)

        if in_place:
            shutil.move(work_out, target)
    except Exception:
        if in_place and os.path.exists(work_out):
            os.remove(work_out)
        raise
    return target


def normalize_loudness(
    src: str,
    dst: _t.Optional[str] = None,
    target_lufs: float = -16.0,
    true_peak_db: float = -1.5,
) -> str:
    """Normalize audio loudness for web/video playback (default -16 LUFS)."""
    target = dst or src
    ext = os.path.splitext(target)[1].lower() or os.path.splitext(src)[1].lower()
    in_place = os.path.abspath(src) == os.path.abspath(target)
    work_out = target
    if in_place:
        fd, work_out = tempfile.mkstemp(suffix=ext, prefix="tts_norm_")
        os.close(fd)

    try:
        if has_ffmpeg():
            af = f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=11"
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                src,
                "-af",
                af,
            ]
            if ext == ".mp3":
                cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
            elif ext == ".wav":
                cmd += ["-codec:a", "pcm_s16le"]
            cmd.append(work_out)
            _run_tool(cmd, "Loudness normalization failed")
        elif os.path.splitext(src)[1].lower() == ".wav" and ext == ".wav":
            _peak_normalize_wav_stdlib(src, work_out, target_db=true_peak_db)
        else:
            if not in_place:
                shutil.copy2(src, work_out)
            return target

        if in_place:
            shutil.move(work_out, target)
    except Exception:
        if in_place and os.path.exists(work_out):
            os.remove(work_out)
        raise
    return target


def _peak_normalize_wav_stdlib(src: str, dst: str, target_db: float = -1.5) -> str:
    with wave.open(src, "rb") as wf:
        params = wf.getparams()
        if params.sampwidth != 2:
            shutil.copy2(src, dst)
            return dst
        raw = wf.readframes(params.nframes)

    samples = array.array("h")
    samples.frombytes(raw)
    peak = max((abs(s) for s in samples), default=0)
    if peak == 0:
        shutil.copy2(src, dst)
        return dst
    desired = 32767.0 * (10.0 ** (target_db / 20.0))
    gain = min(desired / peak, 10.0)
    scaled = array.array(
        "h",
        (max(-32768, min(32767, round(s * gain))) for s in samples),
    )
    with wave.open(dst, "wb") as out:
        out.setparams(params)
        out.writeframes(scaled.tobytes())
    return dst


def preprocess_voice_sample(
    src: str, dst: str, max_duration_s: float = 30.0
) -> str:
    """Clean and standardize a reference voice recording for XTTS cloning/training.

    Converts to 22050 Hz mono PCM16 WAV, applies a 70 Hz high-pass filter to
    remove mic rumble, trims leading/trailing silence, and normalizes loudness.
    """
    if not os.path.isfile(src):
        raise TtsError(f"Voice sample not found: {src}")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if not has_ffmpeg():
        if os.path.splitext(src)[1].lower() == ".wav":
            return _trim_silence_wav_stdlib(src, dst)
        raise TtsError(
            "ffmpeg is required to preprocess non-WAV voice samples."
        )

    af = (
        "highpass=f=70,"
        "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-45dB,"
        "areverse,"
        "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-45dB,"
        "areverse,"
        "loudnorm=I=-18:TP=-1.5:LRA=11"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src,
    ]
    if max_duration_s > 0:
        cmd += ["-t", f"{max_duration_s:.2f}"]
    cmd += [
        "-af",
        af,
        "-ar",
        "22050",
        "-ac",
        "1",
        "-codec:a",
        "pcm_s16le",
        dst,
    ]
    _run_tool(cmd, "Voice sample preprocessing failed")

    # If trimming stripped too much (e.g., quiet sample), retry without silenceremove.
    info = probe_audio(dst)
    if info.duration_ms < 500:
        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            src,
        ]
        if max_duration_s > 0:
            fallback_cmd += ["-t", f"{max_duration_s:.2f}"]
        fallback_cmd += [
            "-af",
            "highpass=f=70,loudnorm=I=-18:TP=-1.5:LRA=11",
            "-ar",
            "22050",
            "-ac",
            "1",
            "-codec:a",
            "pcm_s16le",
            dst,
        ]
        _run_tool(fallback_cmd, "Voice sample preprocessing fallback failed")
    return dst


def _concat_wav_stdlib(
    segments: _t.Sequence[tuple[str, int]], output_path: str
) -> tuple[str, list[dict[str, int]]]:
    if not segments:
        raise TtsError("No audio segments provided to concatenate.")

    timings: list[dict[str, int]] = []
    cursor_ms = 0
    out_params = None
    chunks: list[bytes] = []

    for path, pause_ms in segments:
        with wave.open(path, "rb") as wf:
            params = wf.getparams()
            if out_params is None:
                out_params = params
            elif (
                params.nchannels != out_params.nchannels
                or params.sampwidth != out_params.sampwidth
                or params.framerate != out_params.framerate
            ):
                raise TtsError(
                    "WAV segments have mismatched audio parameters and ffmpeg is unavailable."
                )
            data = wf.readframes(params.nframes)
            dur_ms = max(0, round(params.nframes * 1000.0 / (params.framerate or 1)))
        chunks.append(data)
        timings.append(
            {
                "start_ms": cursor_ms,
                "end_ms": cursor_ms + dur_ms,
                "duration_ms": dur_ms,
                "pause_after_ms": max(0, int(pause_ms)),
            }
        )
        cursor_ms += dur_ms
        if pause_ms > 0 and out_params is not None:
            sil_frames = int(out_params.framerate * pause_ms / 1000.0)
            chunks.append(b"\x00" * (sil_frames * out_params.nchannels * out_params.sampwidth))
            cursor_ms += int(pause_ms)

    assert out_params is not None
    with wave.open(output_path, "wb") as out:
        out.setparams(out_params)
        for chunk in chunks:
            out.writeframes(chunk)
    return output_path, timings


def concat_audio_segments(
    segments: _t.Sequence[tuple[str, int]],
    output_path: str,
    sample_rate: int = 24000,
    channels: int = 1,
) -> tuple[str, list[dict[str, int]]]:
    """Concatenate ``[(audio_path, pause_after_ms), ...]`` into ``output_path``.

    Returns ``(output_path, segment_timings)`` where each timing entry has
    ``start_ms``, ``end_ms``, ``duration_ms``, and ``pause_after_ms``.
    """
    if not segments:
        raise TtsError("Cannot concatenate an empty list of audio segments.")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if not has_ffmpeg():
        return _concat_wav_stdlib(segments, output_path)

    timings: list[dict[str, int]] = []
    cursor_ms = 0
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    cl = "mono" if channels == 1 else "stereo"

    for idx, (path, pause_ms) in enumerate(segments):
        info = probe_audio(path)
        dur_ms = info.duration_ms
        pause = max(0, int(pause_ms))
        timings.append(
            {
                "start_ms": cursor_ms,
                "end_ms": cursor_ms + dur_ms,
                "duration_ms": dur_ms,
                "pause_after_ms": pause,
            }
        )
        cursor_ms += dur_ms + pause

        cmd += ["-i", path]
        filter_parts.append(
            f"[{idx}:a]aformat=sample_rates={sample_rate}:channel_layouts={cl},aresample={sample_rate}[a{idx}]"
        )
        concat_inputs.append(f"[a{idx}]")
        if pause > 0:
            dur_s = pause / 1000.0
            filter_parts.append(
                f"anullsrc=r={sample_rate}:cl={cl}:d={dur_s:.4f}[s{idx}]"
            )
            concat_inputs.append(f"[s{idx}]")

    filter_complex = (
        ";".join(filter_parts)
        + f";{''.join(concat_inputs)}concat=n={len(concat_inputs)}:v=0:a=1[out]"
    )
    cmd += ["-filter_complex", filter_complex, "-map", "[out]"]

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp3":
        cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
    elif ext == ".wav":
        cmd += ["-codec:a", "pcm_s16le"]
    cmd.append(output_path)

    _run_tool(cmd, "Audio segment concatenation failed")
    return output_path, timings


def mux_audio_to_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    subtitle_path: _t.Optional[str] = None,
    burn_subtitles: bool = False,
    replace_audio: bool = True,
    bg_audio_volume: float = 0.2,
    fit_mode: str = "keep_video",
) -> str:
    """Attach narration audio (and optional SRT subtitles) onto a video file.

    ``fit_mode`` controls stream alignment: ``'keep_video'`` preserves the
    video/narration without prematurely truncating either, while ``'shortest'``
    ends at the shorter stream.
    """
    if not has_ffmpeg():
        raise TtsError("ffmpeg is required to mux audio onto a video.")
    if not os.path.isfile(video_path):
        raise TtsError(f"Video file not found: {video_path}")
    if not os.path.isfile(audio_path):
        raise TtsError(f"Audio file not found: {audio_path}")
    if subtitle_path and not os.path.isfile(subtitle_path):
        raise TtsError(f"Subtitle file not found: {subtitle_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        audio_path,
    ]

    # Check if video already contains an audio stream when mixing.
    video_has_audio = False
    if not replace_audio and has_ffprobe():
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        video_has_audio = bool(probe.stdout.strip())

    filter_parts: list[str] = []
    video_map = "0:v:0"
    audio_map = "1:a:0"

    if subtitle_path and burn_subtitles:
        escaped_sub = subtitle_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filter_parts.append(f"[0:v:0]subtitles='{escaped_sub}'[vout]")
        video_map = "[vout]"

    if not replace_audio and video_has_audio:
        vol = max(0.0, min(1.0, float(bg_audio_volume)))
        filter_parts.append(
            f"[0:a:0]volume={vol:.3f}[bg];[bg][1:a:0]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
        )
        audio_map = "[aout]"

    if subtitle_path and not burn_subtitles:
        cmd += ["-i", subtitle_path]

    if filter_parts:
        cmd += ["-filter_complex", ";".join(filter_parts)]

    cmd += ["-map", video_map, "-map", audio_map]
    if subtitle_path and not burn_subtitles:
        cmd += ["-map", "2:s:0", "-c:s", "mov_text"]

    if burn_subtitles:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    else:
        cmd += ["-c:v", "copy"]

    if fit_mode not in ("keep_video", "shortest"):
        raise TtsError(f"Unsupported fit_mode '{fit_mode}'. Use 'keep_video' or 'shortest'.")

    cmd += ["-c:a", "aac", "-b:a", "192k"]
    if fit_mode == "shortest":
        cmd.append("-shortest")
    cmd.append(output_path)
    _run_tool(cmd, "Video muxing failed")
    return output_path


def estimate_duration_ms_fallback(text: str, words_per_minute: float = 150.0) -> int:
    """Rough speech duration estimate when no audio metadata is available."""
    words = max(1, len(text.split()))
    return max(400, round(words / words_per_minute * 60_000))


def safe_remove(path: _t.Optional[str]) -> None:
    """Delete a file if it exists, ignoring OS errors."""
    if path:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
