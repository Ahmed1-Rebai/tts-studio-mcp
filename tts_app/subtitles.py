"""Subtitle (SRT / VTT) and frame-accurate JSON timing generation for video."""

from __future__ import annotations

import json
import os
import re
import typing as _t

from tts_app.models import CaptionCue, WordTiming

_TOKEN_RE = re.compile(r"\S+")


def format_srt_time(ms: int) -> str:
    """Format milliseconds as ``HH:MM:SS,mmm``."""
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_vtt_time(ms: int) -> str:
    """Format milliseconds as ``HH:MM:SS.mmm``."""
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def estimate_word_timings(
    text: str,
    start_ms: int,
    end_ms: int,
    scene_id: _t.Optional[str] = None,
) -> list[WordTiming]:
    """Distribute ``[start_ms, end_ms]`` proportionally across words in ``text``.

    Used for offline engines (XTTS / pyttsx3) that do not emit native word
    boundaries, weighted by token length plus punctuation pause weight.
    """
    tokens = _TOKEN_RE.findall(text.strip())
    if not tokens:
        return []
    span = max(1, int(end_ms) - int(start_ms))
    weights: list[float] = []
    for tok in tokens:
        base = max(1.0, float(len(re.sub(r"[^\w]", "", tok)) or len(tok)))
        if tok.endswith((".", "!", "?", "\u3002", "\uff01", "\uff1f")):
            base += 3.0
        elif tok.endswith((",", ";", ":", "\u060c")):
            base += 1.5
        weights.append(base)
    total_w = sum(weights) or 1.0

    timings: list[WordTiming] = []
    cursor = float(start_ms)
    for idx, (tok, w) in enumerate(zip(tokens, weights, strict=True)):
        dur = span * (w / total_w)
        w_start = round(cursor)
        w_end = int(end_ms) if idx == len(tokens) - 1 else max(w_start + 1, round(cursor + dur))
        timings.append(
            WordTiming(text=tok, start_ms=w_start, end_ms=w_end, scene_id=scene_id)
        )
        cursor += dur
    return timings


def shift_word_timings(
    words: _t.Sequence[WordTiming],
    offset_ms: int,
    scene_id: _t.Optional[str] = None,
) -> list[WordTiming]:
    """Return a shifted copy of ``words`` by ``offset_ms``."""
    return [
        WordTiming(
            text=w.text,
            start_ms=max(0, w.start_ms + offset_ms),
            end_ms=max(0, w.end_ms + offset_ms),
            scene_id=scene_id if scene_id is not None else w.scene_id,
        )
        for w in words
    ]


def group_words_into_cues(
    words: _t.Sequence[WordTiming],
    max_words: int = 6,
    max_chars: int = 42,
    max_pause_ms: int = 500,
) -> list[CaptionCue]:
    """Group word timings into readable video caption cues."""
    if not words:
        return []
    max_words = max(1, int(max_words))
    max_chars = max(8, int(max_chars))

    cues: list[CaptionCue] = []
    buf: list[WordTiming] = []

    def _flush() -> None:
        if not buf:
            return
        text = " ".join(w.text for w in buf)
        cues.append(
            CaptionCue(
                index=len(cues) + 1,
                text=text,
                start_ms=buf[0].start_ms,
                end_ms=max(buf[-1].end_ms, buf[0].start_ms + 1),
                words=list(buf),
            )
        )
        buf.clear()

    for word in words:
        if buf:
            gap = word.start_ms - buf[-1].end_ms
            projected_len = sum(len(w.text) for w in buf) + len(buf) + len(word.text)
            prev_ends_sentence = buf[-1].text.endswith(
                (".", "!", "?", "\u3002", "\uff01", "\uff1f")
            )
            scene_changed = (
                word.scene_id is not None
                and buf[-1].scene_id is not None
                and word.scene_id != buf[-1].scene_id
            )
            if (
                len(buf) >= max_words
                or projected_len > max_chars
                or gap > max_pause_ms
                or prev_ends_sentence
                or scene_changed
            ):
                _flush()
        buf.append(word)
    _flush()
    return cues


def to_srt(cues: _t.Sequence[CaptionCue]) -> str:
    """Render ``cues`` in SubRip (.srt) format."""
    blocks: list[str] = []
    for idx, cue in enumerate(cues, start=1):
        blocks.append(
            f"{idx}\n"
            f"{format_srt_time(cue.start_ms)} --> {format_srt_time(cue.end_ms)}\n"
            f"{cue.text}"
        )
    return ("\n\n".join(blocks) + "\n") if blocks else ""


def to_vtt(cues: _t.Sequence[CaptionCue]) -> str:
    """Render ``cues`` in WebVTT (.vtt) format."""
    blocks = ["WEBVTT"]
    for cue in cues:
        blocks.append(
            f"{format_vtt_time(cue.start_ms)} --> {format_vtt_time(cue.end_ms)}\n"
            f"{cue.text}"
        )
    return "\n\n".join(blocks) + "\n"


def to_timings_manifest(
    duration_ms: int,
    fps: int,
    timing_mode: str,
    cues: _t.Sequence[CaptionCue],
    words: _t.Sequence[WordTiming],
    scenes: _t.Optional[_t.Sequence[dict[str, _t.Any]]] = None,
    metadata: _t.Optional[dict[str, _t.Any]] = None,
) -> dict[str, _t.Any]:
    """Build a JSON manifest tailored for Remotion, HyperFrames, and FFmpeg."""
    fps = max(1, int(fps))
    total_frames = round(duration_ms * fps / 1000.0)
    word_dicts: list[dict[str, _t.Any]] = []
    for w in words:
        item = w.to_dict()
        item["start_frame"] = round(w.start_ms * fps / 1000.0)
        item["end_frame"] = round(w.end_ms * fps / 1000.0)
        word_dicts.append(item)

    payload: dict[str, _t.Any] = {
        "fps": fps,
        "total_duration_ms": duration_ms,
        "total_duration_s": round(duration_ms / 1000.0, 3),
        "total_frames": total_frames,
        "timing_mode": timing_mode,
        "cues": [c.to_dict(fps=fps) for c in cues],
        "words": word_dicts,
    }
    if scenes is not None:
        payload["scenes"] = list(scenes)
    if metadata:
        payload["metadata"] = metadata
    return payload


def write_subtitle_artifacts(
    stem_path: str,
    words: _t.Sequence[WordTiming],
    duration_ms: int,
    fps: int = 30,
    timing_mode: str = "word_boundary",
    max_words_per_cue: int = 6,
    max_chars_per_cue: int = 42,
    scenes: _t.Optional[_t.Sequence[dict[str, _t.Any]]] = None,
    metadata: _t.Optional[dict[str, _t.Any]] = None,
) -> tuple[str, str, str, list[CaptionCue]]:
    """Write ``.srt``, ``.vtt``, and ``.timings.json`` next to ``stem_path``."""
    cues = group_words_into_cues(
        words, max_words=max_words_per_cue, max_chars=max_chars_per_cue
    )
    os.makedirs(os.path.dirname(os.path.abspath(stem_path)), exist_ok=True)
    srt_path = f"{stem_path}.srt"
    vtt_path = f"{stem_path}.vtt"
    timings_path = f"{stem_path}.timings.json"

    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write(to_srt(cues))
    with open(vtt_path, "w", encoding="utf-8") as fh:
        fh.write(to_vtt(cues))
    manifest = to_timings_manifest(
        duration_ms=duration_ms,
        fps=fps,
        timing_mode=timing_mode,
        cues=cues,
        words=words,
        scenes=scenes,
        metadata=metadata,
    )
    with open(timings_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return srt_path, vtt_path, timings_path, cues
