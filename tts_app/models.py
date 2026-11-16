"""Shared data models and exceptions for the TTS application."""

from __future__ import annotations

import typing as _t
from dataclasses import asdict, dataclass, field


class TtsError(Exception):
    """Raised when an engine, audio pipeline, or profile operation fails."""


class Voice:
    """A single selectable voice or voice-cloning target.

    Attributes
    ----------
    id: Unique identifier understood by the TTS backend (or sample/profile key).
    name: Human-friendly display name.
    locale: BCP-47 locale / primary language tag (e.g. ``"en-US"`` or ``"en"``).
    gender: ``"Male"``, ``"Female"``, or ``"Unknown"``.
    age: ``"Adult"``, ``"Child"``, or ``"Unknown"``.
    engine: Backend identifier (e.g. ``"edge-tts"``, ``"xtts_v2"``, ``"pyttsx3"``).
    kind: ``"catalog"``, ``"builtin"``, ``"profile"``, ``"finetuned"``, or ``"sample"``.
    """

    __slots__ = ("age", "engine", "gender", "id", "kind", "locale", "name")

    def __init__(
        self,
        voice_id: str,
        name: str,
        locale: str,
        gender: str = "Female",
        age: str = "Adult",
        engine: str = "",
        kind: str = "catalog",
    ) -> None:
        self.id = voice_id
        self.name = name
        self.locale = locale or ""
        self.gender = gender
        self.age = age
        self.engine = engine
        self.kind = kind

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "locale": self.locale,
            "gender": self.gender,
            "age": self.age,
            "engine": self.engine,
            "kind": self.kind,
        }

    def __repr__(self) -> str:
        return (
            f"Voice<{self.id} | {self.name} | {self.locale} "
            f"| {self.gender} | {self.age}>"
        )


@dataclass
class WordTiming:
    """Timing span for a single spoken word or token."""

    text: str
    start_ms: int
    end_ms: int
    scene_id: _t.Optional[str] = None

    def to_dict(self) -> dict[str, _t.Any]:
        data: dict[str, _t.Any] = {
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }
        if self.scene_id is not None:
            data["scene_id"] = self.scene_id
        return data


@dataclass
class CaptionCue:
    """A grouped subtitle caption block suitable for SRT/VTT/video rendering."""

    index: int
    text: str
    start_ms: int
    end_ms: int
    words: list[WordTiming] = field(default_factory=list)

    def to_dict(self, fps: _t.Optional[int] = None) -> dict[str, _t.Any]:
        data: dict[str, _t.Any] = {
            "index": self.index,
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }
        if fps and fps > 0:
            data["start_frame"] = round(self.start_ms * fps / 1000.0)
            data["end_frame"] = round(self.end_ms * fps / 1000.0)
        if self.words:
            data["words"] = [w.to_dict() for w in self.words]
        return data


@dataclass
class SynthesisResult:
    """Rich result returned by smart TTS synthesis."""

    audio_path: str
    format: str
    duration_ms: int
    sample_rate: int
    channels: int
    engine: str
    voice_id: str
    voice_name: str
    language: str
    timing_mode: str = "none"
    word_timings: list[WordTiming] = field(default_factory=list)
    cues: list[CaptionCue] = field(default_factory=list)
    srt_path: _t.Optional[str] = None
    vtt_path: _t.Optional[str] = None
    timings_path: _t.Optional[str] = None
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return round(self.duration_ms / 1000.0, 3)

    def to_dict(self) -> dict[str, _t.Any]:
        return {
            "audio_path": self.audio_path,
            "format": self.format,
            "duration_ms": self.duration_ms,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "engine": self.engine,
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "language": self.language,
            "timing_mode": self.timing_mode,
            "word_count": len(self.word_timings),
            "cue_count": len(self.cues),
            "srt_path": self.srt_path,
            "vtt_path": self.vtt_path,
            "timings_path": self.timings_path,
            "fallback_used": self.fallback_used,
            "warnings": list(self.warnings),
        }


@dataclass
class VoiceProfile:
    """Metadata describing an enrolled or fine-tuned user voice profile."""

    id: str
    name: str
    kind: str  # "enrolled" | "finetuned"
    language: str
    created_at: str
    sample_count: int
    total_source_duration_s: float
    sample_files: list[str] = field(default_factory=list)
    latent_path: _t.Optional[str] = None
    checkpoint_dir: _t.Optional[str] = None
    speaker_ref_wav: _t.Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict[str, _t.Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, _t.Any]) -> "VoiceProfile":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            kind=str(data.get("kind", "enrolled")),
            language=str(data.get("language", "en")),
            created_at=str(data.get("created_at", "")),
            sample_count=int(data.get("sample_count", 0)),
            total_source_duration_s=float(data.get("total_source_duration_s", 0.0)),
            sample_files=list(data.get("sample_files", [])),
            latent_path=data.get("latent_path"),
            checkpoint_dir=data.get("checkpoint_dir"),
            speaker_ref_wav=data.get("speaker_ref_wav"),
            notes=str(data.get("notes", "")),
        )
