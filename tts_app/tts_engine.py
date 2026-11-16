"""Smart multi-engine Text-to-Speech abstraction layer.

Provides a unified async-first and sync-safe interface across:

    * ``EdgeTtsEngine``  - Online Microsoft Edge neural voices with native
      ``WordBoundary`` timestamps for frame-accurate video captions.
    * ``XttsEngine``     - Local Coqui XTTS v2 supporting zero-shot cloning,
      saved enrolled voice profiles, fine-tuned custom checkpoints, and
      58 built-in offline neural speakers.
    * ``Pyttsx3Engine``  - Lightweight offline fallback using OS voices.

Also provides ``smart_synthesize`` / ``smart_synthesize_async`` for automatic
engine selection, network-failure fallback, loudness normalization, silence
trimming, and SRT/VTT/JSON subtitle export.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
import typing as _t
import wave
from concurrent.futures import ThreadPoolExecutor

from tts_app.audio_utils import (
    concat_audio_segments,
    convert_audio,
    estimate_duration_ms_fallback,
    has_ffmpeg,
    normalize_loudness,
    probe_audio,
    safe_remove,
    trim_silence,
)
from tts_app.models import SynthesisResult, TtsError, Voice, WordTiming
from tts_app.subtitles import (
    estimate_word_timings,
    group_words_into_cues,
    shift_word_timings,
    write_subtitle_artifacts,
)
from tts_app.voice_profiles import get_data_dir, get_profile, list_profiles

# ---------------------------------------------------------------------------
# Locale / text helpers
# ---------------------------------------------------------------------------

#: Human-readable display names for language subtags.
_LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bpy": "Bishnupriya",
    "bs": "Bosnian",
    "ca": "Catalan",
    "chr": "Cherokee",
    "cmn": "Mandarin Chinese",
    "cs": "Czech",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "iu": "Inuktitut",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "or": "Odia",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zu": "Zulu",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f\u061b])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:\u060c])\s+")


def language_display_name(subtag: str) -> str:
    """Full human-readable name for a language subtag ('ar' -> 'Arabic')."""
    if not subtag:
        return ""
    key = subtag.lower()
    return _LANGUAGE_NAMES.get(key, _LANGUAGE_NAMES.get(key.split("-")[0], subtag))


def language_subtag(locale: str) -> str:
    """Return the primary language subtag (lower-cased) for a locale."""
    if not locale:
        return ""
    return locale.replace("_", "-").split("-")[0].lower()


def normalize_xtts_language(language: str) -> str:
    """Map generic language tags to XTTS v2 language codes."""
    raw = (language or "en").strip().lower().replace("_", "-")
    if raw in ("zh", "zh-cn", "zh-hans", "cmn"):
        return "zh-cn"
    base = raw.split("-")[0]
    if base == "zh":
        return "zh-cn"
    return base


def speed_to_edge_rate(speed: float) -> str:
    """Convert a float multiplier (e.g. ``1.2``) into edge-tts rate (``'+20%'``)."""
    clamped = max(0.5, min(2.0, float(speed)))
    pct = round((clamped - 1.0) * 100)
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


def split_text_smart(text: str, max_chars: int = 220) -> list[str]:
    """Split ``text`` into natural speech chunks bounded by ``max_chars``."""
    normalized = " ".join(text.strip().split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = _SENTENCE_SPLIT_RE.split(normalized)
    chunks: list[str] = []
    current = ""

    def _push_piece(piece: str) -> None:
        nonlocal current
        if not piece:
            return
        if len(piece) <= max_chars:
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chars:
                current = f"{current} {piece}"
            else:
                chunks.append(current)
                current = piece
            return

        # Oversized sentence: split on clauses, then words.
        sub_parts = _CLAUSE_SPLIT_RE.split(piece)
        if len(sub_parts) > 1:
            for sub in sub_parts:
                _push_piece(sub)
            return

        words = piece.split(" ")
        for w in words:
            if not w:
                continue
            if len(w) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                for idx in range(0, len(w), max_chars):
                    chunks.append(w[idx : idx + max_chars])
            elif not current:
                current = w
            elif len(current) + 1 + len(w) <= max_chars:
                current = f"{current} {w}"
            else:
                chunks.append(current)
                current = w

    for sent in sentences:
        _push_piece(sent.strip())
    if current:
        chunks.append(current)
    return chunks


def _run_coro_sync(coro_fn: _t.Callable[[], _t.Coroutine[_t.Any, _t.Any, _t.Any]]) -> _t.Any:
    """Execute an async coroutine safely whether or not an event loop is running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_fn())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_fn())).result()


def _finalize_audio_and_subtitles(
    raw_path: str,
    output_path: str,
    text: str,
    engine_id: str,
    voice: Voice,
    language: str,
    timing_mode: str,
    word_timings: list[WordTiming],
    trim: bool = False,
    normalize: bool = False,
    generate_subtitles: bool = False,
    fps: int = 30,
    max_words_per_cue: int = 6,
    fallback_used: bool = False,
    warnings: _t.Optional[list[str]] = None,
) -> SynthesisResult:
    """Convert to requested format, optionally trim/normalize, and write captions."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    work_path = raw_path
    temp_files: list[str] = []

    try:
        if trim:
            trimmed = _mktemp_like(work_path)
            temp_files.append(trimmed)
            work_path = trim_silence(work_path, trimmed)

        if normalize:
            normed = _mktemp_like(work_path)
            temp_files.append(normed)
            work_path = normalize_loudness(work_path, normed)

        if os.path.abspath(work_path) != os.path.abspath(output_path):
            convert_audio(work_path, output_path)
    finally:
        for tmp in temp_files:
            safe_remove(tmp)
        if os.path.abspath(raw_path) != os.path.abspath(output_path):
            safe_remove(raw_path)

    info = probe_audio(output_path)
    duration_ms = info.duration_ms or estimate_duration_ms_fallback(text)
    if not word_timings:
        word_timings = estimate_word_timings(text, 0, duration_ms)
        timing_mode = "estimated_from_audio"

    cues = group_words_into_cues(word_timings, max_words=max_words_per_cue)
    srt_path = vtt_path = timings_path = None
    if generate_subtitles:
        stem, _ = os.path.splitext(output_path)
        srt_path, vtt_path, timings_path, cues = write_subtitle_artifacts(
            stem_path=stem,
            words=word_timings,
            duration_ms=duration_ms,
            fps=fps,
            timing_mode=timing_mode,
            max_words_per_cue=max_words_per_cue,
            metadata={
                "engine": engine_id,
                "voice_id": voice.id,
                "voice_name": voice.name,
                "language": language,
            },
        )

    return SynthesisResult(
        audio_path=output_path,
        format=os.path.splitext(output_path)[1].lstrip(".").lower() or info.format,
        duration_ms=duration_ms,
        sample_rate=info.sample_rate or 24000,
        channels=info.channels or 1,
        engine=engine_id,
        voice_id=voice.id,
        voice_name=voice.name,
        language=language,
        timing_mode=timing_mode,
        word_timings=word_timings,
        cues=cues,
        srt_path=srt_path,
        vtt_path=vtt_path,
        timings_path=timings_path,
        fallback_used=fallback_used,
        warnings=list(warnings or []),
    )


def _mktemp_like(ref_path: str) -> str:
    ext = os.path.splitext(ref_path)[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=ext, prefix="tts_work_")
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Base engine interface
# ---------------------------------------------------------------------------


class BaseTtsEngine:
    """Common interface implemented by every TTS backend."""

    id = "base"
    label = "Base"
    has_voice_catalog = True
    supports_cloning = False

    def __init__(self) -> None:
        self._voices: _t.List[Voice] = []

    def _discover(self) -> _t.List[Voice]:
        raise NotImplementedError

    def voices(self) -> _t.List[Voice]:
        if not self._voices:
            self._voices = self._discover()
        return self._voices

    def list_languages(self) -> _t.List[str]:
        return sorted(
            {language_subtag(v.locale) for v in self.voices() if v.locale}
        )

    def list_voices(self, language: str = "") -> _t.List[Voice]:
        if not language:
            return list(self.voices())
        lang = language_subtag(language)
        return [v for v in self.voices() if language_subtag(v.locale) == lang]

    def genders(self, language: str = "") -> _t.List[str]:
        seen = [v.gender for v in self.list_voices(language)]
        uniq = sorted({g for g in seen if g and g != "Unknown"})
        if any(g == "Unknown" for g in seen):
            uniq.append("Unknown")
        return uniq or ["Unknown"]

    def synthesize_rich(
        self,
        text: str,
        voice: Voice,
        output_path: str,
        **kwargs: _t.Any,
    ) -> SynthesisResult:
        raise NotImplementedError(f"{self.label} does not support synthesis")

    async def synthesize_rich_async(
        self,
        text: str,
        voice: Voice,
        output_path: str,
        **kwargs: _t.Any,
    ) -> SynthesisResult:
        return await asyncio.to_thread(
            self.synthesize_rich, text, voice, output_path, **kwargs
        )


# ---------------------------------------------------------------------------
# edge-tts engine (primary online)
# ---------------------------------------------------------------------------


class EdgeTtsEngine(BaseTtsEngine):
    """Online neural TTS via Microsoft Edge with WordBoundary timing support."""

    id = "edge-tts"
    label = "edge-tts (online)"

    _CACHE_TTL_S = 86_400

    @staticmethod
    def _cache_path() -> str:
        return os.path.join(get_data_dir(), "cache", "edge_voices.json")

    @staticmethod
    def _voices_from_raw(results: list[dict[str, _t.Any]]) -> _t.List[Voice]:
        catalog: _t.List[Voice] = []
        for item in results:
            short = item.get("ShortName", "")
            friendly = item.get("FriendlyName", "")
            gender = item.get("Gender", "Unknown") or "Unknown"
            catalog.append(
                Voice(
                    voice_id=short,
                    name=friendly or short,
                    locale=item.get("Locale", ""),
                    gender=gender,
                    age=_infer_age(friendly),
                    engine=EdgeTtsEngine.id,
                    kind="catalog",
                )
            )
        return catalog

    def _discover(self) -> _t.List[Voice]:
        cache_file = self._cache_path()
        stale_results: _t.Optional[list[dict[str, _t.Any]]] = None
        if os.path.isfile(cache_file):
            try:
                age_s = time.time() - os.path.getmtime(cache_file)
                with open(cache_file, "r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                if isinstance(cached, list) and cached:
                    if age_s <= self._CACHE_TTL_S:
                        return self._voices_from_raw(cached)
                    stale_results = cached
            except (OSError, ValueError):
                pass

        import edge_tts

        async def _fetch() -> list[dict[str, _t.Any]]:
            res = edge_tts.list_voices()
            if asyncio.iscoroutine(res):
                return await res
            return res

        try:
            results = _run_coro_sync(_fetch)
        except Exception:
            if stale_results:
                return self._voices_from_raw(stale_results)
            raise

        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(results, fh, ensure_ascii=False)
        except OSError:
            pass

        return self._voices_from_raw(results)

    async def synthesize_rich_async(
        self,
        text: str,
        voice: Voice,
        output_path: str,
        *,
        speed: float = 1.0,
        pitch: str = "+0Hz",
        volume: str = "+0%",
        trim: bool = False,
        normalize: bool = False,
        generate_subtitles: bool = False,
        fps: int = 30,
        max_words_per_cue: int = 6,
    ) -> SynthesisResult:
        import edge_tts

        if not text or not text.strip():
            raise TtsError("Text cannot be empty.")

        chunks = split_text_smart(text, max_chars=2800)
        rate = speed_to_edge_rate(speed)
        chunk_mp3s: list[tuple[str, int]] = []
        all_words: list[WordTiming] = []
        offset_cursor_ms = 0

        fd_raw, raw_mp3 = tempfile.mkstemp(suffix=".mp3", prefix="tts_edge_")
        os.close(fd_raw)

        try:
            for idx, chunk in enumerate(chunks):
                fd_c, c_path = tempfile.mkstemp(suffix=".mp3", prefix=f"tts_edge_{idx}_")
                os.close(fd_c)
                chunk_words: list[WordTiming] = []
                communicate = edge_tts.Communicate(
                    text=chunk,
                    voice=voice.id,
                    rate=rate,
                    volume=volume,
                    pitch=pitch,
                    boundary="WordBoundary",
                )
                try:
                    with open(c_path, "wb") as audio_fh:
                        async for event in communicate.stream():
                            etype = event.get("type")
                            if etype == "audio":
                                audio_fh.write(event["data"])
                            elif etype in ("WordBoundary", "SentenceBoundary"):
                                start_ms = round(event["offset"] / 10_000)
                                dur_ms = max(1, round(event["duration"] / 10_000))
                                token = str(event.get("text", "")).strip()
                                if token:
                                    chunk_words.append(
                                        WordTiming(
                                            text=token,
                                            start_ms=start_ms,
                                            end_ms=start_ms + dur_ms,
                                        )
                                    )
                except Exception as exc:
                    safe_remove(c_path)
                    raise TtsError(f"edge-tts failed: {exc}") from exc

                chunk_mp3s.append((c_path, 0))
                if len(chunks) > 1:
                    c_info = probe_audio(c_path)
                    c_dur = c_info.duration_ms or estimate_duration_ms_fallback(chunk)
                    all_words.extend(shift_word_timings(chunk_words, offset_cursor_ms))
                    offset_cursor_ms += c_dur
                else:
                    all_words.extend(chunk_words)

            if len(chunk_mp3s) == 1:
                shutil.move(chunk_mp3s[0][0], raw_mp3)
                chunk_mp3s.clear()
            elif has_ffmpeg():
                concat_audio_segments(chunk_mp3s, raw_mp3)
            else:
                with open(raw_mp3, "wb") as out_fh:
                    for c_path, _ in chunk_mp3s:
                        with open(c_path, "rb") as in_fh:
                            out_fh.write(in_fh.read())
        finally:
            for c_path, _ in chunk_mp3s:
                safe_remove(c_path)

        timing_mode = "word_boundary" if all_words else "estimated_from_audio"
        return _finalize_audio_and_subtitles(
            raw_path=raw_mp3,
            output_path=output_path,
            text=text,
            engine_id=self.id,
            voice=voice,
            language=language_subtag(voice.locale) or "en",
            timing_mode=timing_mode,
            word_timings=all_words,
            trim=trim,
            normalize=normalize,
            generate_subtitles=generate_subtitles,
            fps=fps,
            max_words_per_cue=max_words_per_cue,
        )

    def synthesize_rich(
        self,
        text: str,
        voice: Voice,
        output_path: str,
        **kwargs: _t.Any,
    ) -> SynthesisResult:
        return _run_coro_sync(
            lambda: self.synthesize_rich_async(text, voice, output_path, **kwargs)
        )


def _infer_age(friendly: str) -> str:
    if not friendly:
        return "Adult"
    lower = friendly.lower()
    if "child" in lower or "kid" in lower:
        return "Child"
    return "Adult"


# ---------------------------------------------------------------------------
# pyttsx3 engine (lightweight offline fallback)
# ---------------------------------------------------------------------------


class Pyttsx3Engine(BaseTtsEngine):
    """Fully offline backend using the operating system's installed voices."""

    id = "pyttsx3"
    label = "pyttsx3 (offline)"

    def _discover(self) -> _t.List[Voice]:
        import pyttsx3

        engine = pyttsx3.init()
        try:
            raw = engine.getProperty("voices")
            voices: _t.List[Voice] = []
            for v in raw:
                name = (
                    getattr(v, "name", None)
                    or os.path.basename(getattr(v, "id", ""))
                    or str(v.id)
                )
                locale = ""
                langs = getattr(v, "languages", None)
                if langs and len(langs) > 0:
                    locale = _decode_lang(langs[0])
                raw_gender = getattr(v, "gender", None)
                gender = raw_gender if raw_gender in ("Male", "Female") else "Unknown"
                voices.append(
                    Voice(
                        voice_id=v.id,
                        name=name,
                        locale=locale,
                        gender=gender,
                        age="Adult",
                        engine=self.id,
                        kind="catalog",
                    )
                )
            return voices
        finally:
            engine.stop()

    def synthesize_rich(
        self,
        text: str,
        voice: Voice,
        output_path: str,
        *,
        speed: float = 1.0,
        volume: _t.Optional[_t.Union[float, str]] = None,
        trim: bool = False,
        normalize: bool = False,
        generate_subtitles: bool = False,
        fps: int = 30,
        max_words_per_cue: int = 6,
        **_ignored: _t.Any,
    ) -> SynthesisResult:
        import pyttsx3

        if not text or not text.strip():
            raise TtsError("Text cannot be empty.")

        fd, raw_wav = tempfile.mkstemp(suffix=".wav", prefix="tts_pyttsx3_")
        os.close(fd)
        engine = pyttsx3.init()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                engine.setProperty("voice", voice.id)
                if abs(float(speed) - 1.0) > 1e-3:
                    base_rate = engine.getProperty("rate") or 200
                    engine.setProperty("rate", max(80, min(400, int(base_rate * float(speed)))))
                if volume is not None:
                    if isinstance(volume, str) and volume.strip().endswith("%"):
                        pct = float(volume.strip().rstrip("%"))
                        vol_val = 1.0 + (pct / 100.0)
                    else:
                        vol_val = float(volume)
                    engine.setProperty("volume", max(0.0, min(1.0, vol_val)))
                engine.save_to_file(text, raw_wav)
                engine.runAndWait()
        except Exception as exc:
            safe_remove(raw_wav)
            raise TtsError(f"pyttsx3 failed: {exc}") from exc
        finally:
            engine.stop()

        return _finalize_audio_and_subtitles(
            raw_path=raw_wav,
            output_path=output_path,
            text=text,
            engine_id=self.id,
            voice=voice,
            language=language_subtag(voice.locale) or "en",
            timing_mode="estimated_from_audio",
            word_timings=[],
            trim=trim,
            normalize=normalize,
            generate_subtitles=generate_subtitles,
            fps=fps,
            max_words_per_cue=max_words_per_cue,
        )


def _decode_lang(value: _t.Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").lstrip("\x05")
    return str(value).lstrip("\x05")


# ---------------------------------------------------------------------------
# XTTS v2 engine (local neural voices, voice cloning, profiles & fine-tuned)
# ---------------------------------------------------------------------------


class XttsEngine(BaseTtsEngine):
    """Coqui XTTS v2 backend: zero-shot cloning, profiles, fine-tuned, & built-in speakers."""

    id = "xtts_v2"
    label = "XTTS v2 (your voice & local)"
    has_voice_catalog = False
    supports_cloning = True

    _LANGUAGES: _t.ClassVar[tuple[str, ...]] = (
        "ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it",
        "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh",
    )

    def __init__(self) -> None:
        super().__init__()
        self._base_model = None
        self._finetuned_models: dict[str, _t.Any] = {}
        self._builtin_speaker_names: _t.Optional[list[str]] = None

    @staticmethod
    def _device() -> str:
        override = os.environ.get("TTS_APP_DEVICE")
        if override:
            return override
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _get_base_model(self):
        if self._base_model is None:
            from TTS.api import TTS as TtsApi

            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            device = self._device()
            api = TtsApi("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            self._base_model = api.synthesizer.tts_model
        return self._base_model

    def _get_finetuned_model(self, checkpoint_dir: str):
        if checkpoint_dir not in self._finetuned_models:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            cfg_path = os.path.join(checkpoint_dir, "config.json")
            ckpt_path = os.path.join(checkpoint_dir, "model.pth")
            vocab_path = os.path.join(checkpoint_dir, "vocab.json")
            if not (os.path.isfile(cfg_path) and os.path.isfile(ckpt_path) and os.path.isfile(vocab_path)):
                raise TtsError(f"Incomplete fine-tuned checkpoint directory: {checkpoint_dir}")

            cfg = XttsConfig()
            cfg.load_json(cfg_path)
            model = Xtts.init_from_config(cfg)
            model.load_checkpoint(
                cfg,
                checkpoint_path=ckpt_path,
                vocab_path=vocab_path,
                use_deepspeed=False,
            )
            model.to(self._device())
            self._finetuned_models[checkpoint_dir] = model
        return self._finetuned_models[checkpoint_dir]

    def list_languages(self) -> _t.List[str]:
        return list(self._LANGUAGES)

    def list_builtin_speakers(self) -> list[str]:
        """Return built-in XTTS speaker names from cached ``speakers_xtts.pth``."""
        if self._builtin_speaker_names is not None:
            return list(self._builtin_speaker_names)

        if self._base_model is not None and getattr(self._base_model, "speaker_manager", None):
            names = sorted(self._base_model.speaker_manager.speakers.keys())
            self._builtin_speaker_names = names
            return list(names)

        spk_path = os.path.join(
            os.path.expanduser("~"),
            ".local",
            "share",
            "tts",
            "tts_models--multilingual--multi-dataset--xtts_v2",
            "speakers_xtts.pth",
        )
        if os.path.isfile(spk_path):
            try:
                import torch

                data = torch.load(spk_path, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    self._builtin_speaker_names = sorted(str(k) for k in data.keys())
                    return list(self._builtin_speaker_names)
            except Exception:
                pass
        self._builtin_speaker_names = []
        return []

    def _discover(self) -> _t.List[Voice]:
        voices: list[Voice] = []
        for prof in list_profiles():
            voices.append(
                Voice(
                    voice_id=f"profile:{prof.id}",
                    name=f"{prof.name} ({prof.kind})",
                    locale=prof.language or "en",
                    gender="Unknown",
                    age="Adult",
                    engine=self.id,
                    kind=prof.kind,
                )
            )
        for spk in self.list_builtin_speakers():
            voices.append(
                Voice(
                    voice_id=f"builtin:{spk}",
                    name=f"{spk} (XTTS built-in)",
                    locale="en",
                    gender="Unknown",
                    age="Adult",
                    engine=self.id,
                    kind="builtin",
                )
            )
        return voices

    def voices(self) -> _t.List[Voice]:
        # Profiles can change at runtime; always refresh quickly.
        return self._discover()

    def list_voices(self, language: str = "") -> _t.List[Voice]:
        # XTTS voices/profiles are multilingual across all 17 XTTS languages.
        if language and normalize_xtts_language(language) not in {
            normalize_xtts_language(lang_code) for lang_code in self._LANGUAGES
        }:
            return []
        return self.voices()

    def synthesize_rich(
        self,
        text: str,
        voice: _t.Optional[Voice],
        output_path: str,
        *,
        language: str = "en",
        voice_profile: _t.Optional[str] = None,
        reference_audio: _t.Optional[_t.Union[str, _t.Sequence[str]]] = None,
        speed: float = 1.0,
        temperature: float = 0.75,
        trim: bool = False,
        normalize: bool = False,
        generate_subtitles: bool = False,
        fps: int = 30,
        max_words_per_cue: int = 6,
        **_ignored: _t.Any,
    ) -> SynthesisResult:
        if not text or not text.strip():
            raise TtsError("Text cannot be empty.")

        import numpy as np
        import torch

        lang = language or (voice.locale if voice else "en") or "en"
        xtts_lang = normalize_xtts_language(lang)
        if xtts_lang not in {
            normalize_xtts_language(lang_code) for lang_code in self._LANGUAGES
        }:
            raise TtsError(
                f"Language '{lang}' is not supported by XTTS v2. "
                f"Supported: {', '.join(self._LANGUAGES)}"
            )

        # Resolve voice source: voice_profile -> reference_audio -> voice.id
        resolved_voice = voice
        profile_obj = None
        builtin_name = None
        ref_files: list[str] = []

        if voice_profile:
            profile_obj = get_profile(voice_profile)
        elif reference_audio:
            ref_files = (
                [reference_audio] if isinstance(reference_audio, str) else list(reference_audio)
            )
        elif voice is not None:
            vid = voice.id or ""
            if vid.startswith("profile:"):
                profile_obj = get_profile(vid[len("profile:") :])
            elif vid.startswith("builtin:"):
                builtin_name = vid[len("builtin:") :]
            elif os.path.isfile(vid):
                ref_files = [vid]
            elif vid in self.list_builtin_speakers():
                builtin_name = vid
            else:
                # Try matching saved profile by id/name as a convenience
                try:
                    profile_obj = get_profile(vid)
                except TtsError:
                    pass

        if profile_obj is None and builtin_name is None and not ref_files:
            raise TtsError(
                "Pick a voice sample, saved voice profile, or built-in XTTS speaker first."
            )

        if profile_obj is not None and profile_obj.kind == "finetuned" and profile_obj.checkpoint_dir:
            model = self._get_finetuned_model(profile_obj.checkpoint_dir)
            resolved_voice = resolved_voice or Voice(
                voice_id=f"profile:{profile_obj.id}",
                name=profile_obj.name,
                locale=xtts_lang,
                gender="Unknown",
                age="Adult",
                engine=self.id,
                kind="finetuned",
            )
        else:
            model = self._get_base_model()

        device = next(model.parameters()).device

        if profile_obj is not None:
            resolved_voice = resolved_voice or Voice(
                voice_id=f"profile:{profile_obj.id}",
                name=profile_obj.name,
                locale=xtts_lang,
                gender="Unknown",
                age="Adult",
                engine=self.id,
                kind=profile_obj.kind,
            )
            if profile_obj.latent_path and os.path.isfile(profile_obj.latent_path):
                lat = torch.load(
                    profile_obj.latent_path, map_location=device, weights_only=False
                )
                gpt_cond_latent = lat["gpt_cond_latent"].to(device)
                speaker_embedding = lat["speaker_embedding"].to(device)
            elif profile_obj.speaker_ref_wav and os.path.isfile(profile_obj.speaker_ref_wav):
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=[profile_obj.speaker_ref_wav]
                )
            else:
                raise TtsError(
                    f"Voice profile '{profile_obj.name}' has no conditioning latents or reference audio."
                )
        elif builtin_name is not None:
            resolved_voice = resolved_voice or Voice(
                voice_id=f"builtin:{builtin_name}",
                name=builtin_name,
                locale=xtts_lang,
                gender="Unknown",
                age="Adult",
                engine=self.id,
                kind="builtin",
            )
            if (
                not getattr(model, "speaker_manager", None)
                or builtin_name not in model.speaker_manager.speakers
            ):
                raise TtsError(f"Built-in XTTS speaker not found: {builtin_name}")
            gpt_cond_latent, speaker_embedding = model.speaker_manager.speakers[
                builtin_name
            ].values()
            gpt_cond_latent = gpt_cond_latent.to(device)
            speaker_embedding = speaker_embedding.to(device)
        else:
            for rf in ref_files:
                if not os.path.isfile(rf):
                    raise TtsError(f"Reference audio file not found: {rf}")
            resolved_voice = resolved_voice or Voice(
                voice_id=ref_files[0],
                name=os.path.basename(ref_files[0]),
                locale=xtts_lang,
                gender="Unknown",
                age="Adult",
                engine=self.id,
                kind="sample",
            )
            cfg = model.config
            with torch.inference_mode():
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=ref_files,
                    max_ref_length=getattr(cfg, "max_ref_len", 30),
                    gpt_cond_len=getattr(cfg, "gpt_cond_len", 6),
                    sound_norm_refs=getattr(cfg, "sound_norm_refs", False),
                )

        chunks = split_text_smart(text, max_chars=220)
        if not chunks:
            raise TtsError("Text cannot be empty.")

        wav_pieces: list[np.ndarray] = []
        word_timings: list[WordTiming] = []
        sample_rate = 24000
        pause_samples = int(sample_rate * 0.08)
        silence_pad = np.zeros(pause_samples, dtype=np.float32)
        fade_samples = min(120, int(sample_rate * 0.005))  # 5 ms boundary ramp
        cursor_ms = 0

        try:
            for idx, chunk in enumerate(chunks):
                out = model.inference(
                    text=chunk,
                    language=xtts_lang,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=float(temperature),
                    speed=max(0.5, min(2.0, float(speed))),
                    enable_text_splitting=False,
                )
                wav_raw = out["wav"]
                if isinstance(wav_raw, torch.Tensor):
                    wav_np = wav_raw.detach().cpu().numpy().astype(np.float32).squeeze()
                else:
                    wav_np = np.asarray(wav_raw, dtype=np.float32).squeeze()
                if len(wav_np) > fade_samples * 2:
                    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                    wav_np = wav_np.copy()
                    wav_np[:fade_samples] *= ramp
                    wav_np[-fade_samples:] *= ramp[::-1]
                dur_ms = max(1, round(len(wav_np) * 1000.0 / sample_rate))
                word_timings.extend(
                    estimate_word_timings(chunk, cursor_ms, cursor_ms + dur_ms)
                )
                wav_pieces.append(wav_np)
                cursor_ms += dur_ms
                if idx < len(chunks) - 1:
                    wav_pieces.append(silence_pad)
                    cursor_ms += round(pause_samples * 1000.0 / sample_rate)
        except Exception as exc:
            raise TtsError(f"XTTS failed: {exc}") from exc

        full_wav = np.concatenate(wav_pieces) if len(wav_pieces) > 1 else wav_pieces[0]
        pcm16 = (np.clip(full_wav, -1.0, 1.0) * 32767.0).astype(np.int16)

        fd, raw_wav = tempfile.mkstemp(suffix=".wav", prefix="tts_xtts_")
        os.close(fd)
        with wave.open(raw_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

        return _finalize_audio_and_subtitles(
            raw_path=raw_wav,
            output_path=output_path,
            text=text,
            engine_id=self.id,
            voice=resolved_voice,
            language=xtts_lang,
            timing_mode="estimated_sentence_aligned",
            word_timings=word_timings,
            trim=trim,
            normalize=normalize,
            generate_subtitles=generate_subtitles,
            fps=fps,
            max_words_per_cue=max_words_per_cue,
        )


# ---------------------------------------------------------------------------
# Engine registry & smart auto-router
# ---------------------------------------------------------------------------

_ENGINE_CACHE: dict[str, BaseTtsEngine] = {}


def available_engines() -> _t.List[BaseTtsEngine]:
    """Return cached instances of every engine whose dependency is importable."""
    engines: _t.List[BaseTtsEngine] = []
    if importlib.util.find_spec("edge_tts") is not None:
        engines.append(_ENGINE_CACHE.setdefault("edge-tts", EdgeTtsEngine()))
    if importlib.util.find_spec("pyttsx3") is not None:
        try:
            engines.append(_ENGINE_CACHE.setdefault("pyttsx3", Pyttsx3Engine()))
        except Exception:
            pass
    if importlib.util.find_spec("TTS") is not None:
        engines.append(_ENGINE_CACHE.setdefault("xtts_v2", XttsEngine()))
    if not engines:
        raise TtsError("No TTS engine available. Install edge-tts or pyttsx3.")
    return engines


def get_engine(name: str) -> BaseTtsEngine:
    """Resolve an engine by id, label, or common alias."""
    norm = (name or "").strip().lower()
    aliases = {
        "edge": "edge-tts",
        "edge-tts": "edge-tts",
        "edge_tts": "edge-tts",
        "xtts": "xtts_v2",
        "xtts_v2": "xtts_v2",
        "xtts-v2": "xtts_v2",
        "xtts v2 (your voice)": "xtts_v2",
        "pyttsx3": "pyttsx3",
        "offline": "pyttsx3",
    }
    target = aliases.get(norm, norm)
    for engine in available_engines():
        if engine.id.lower() == target or engine.label.lower() == norm:
            return engine
    raise TtsError(f"Unknown or unavailable engine: {name}")


def pick_default_voice(
    engine: BaseTtsEngine,
    language: str = "en",
    gender: str = "",
    voice_id: str = "",
) -> Voice:
    """Pick the best matching voice on ``engine`` for ``language`` / ``gender``."""
    if voice_id:
        # Fast-path explicit edge-tts voice IDs to avoid unnecessary catalog latency.
        if engine.id == "edge-tts" and not engine._voices and not os.path.isfile(
            EdgeTtsEngine._cache_path()
        ) and "-" in voice_id:
            parts = voice_id.split("-")
            locale = "-".join(parts[:2]) if len(parts) >= 2 else language
            return Voice(
                voice_id=voice_id,
                name=voice_id,
                locale=locale,
                gender=gender or "Unknown",
                age="Adult",
                engine=engine.id,
                kind="catalog",
            )
        for v in engine.voices():
            if v.id == voice_id or v.name.lower() == voice_id.lower():
                return v

    voices = engine.list_voices(language) or engine.voices()
    if gender and voices:
        g_norm = gender.strip().lower()
        matching = [v for v in voices if v.gender.lower() == g_norm]
        if matching:
            voices = matching
    if not voices:
        raise TtsError(f"No voices found on {engine.label} for language '{language}'.")

    # Prefer modern natural voices on edge-tts when no explicit voice_id given.
    if engine.id == "edge-tts":
        for v in voices:
            if "MultilingualNeural" in v.id or "AndrewNeural" in v.id or "AriaNeural" in v.id:
                return v
    return voices[0]


async def smart_synthesize_async(
    text: str,
    output_path: str,
    *,
    engine: str = "auto",
    language: str = "en",
    voice_id: str = "",
    gender: str = "",
    voice_profile: _t.Optional[str] = None,
    reference_audio: _t.Optional[_t.Union[str, _t.Sequence[str]]] = None,
    speed: float = 1.0,
    pitch: str = "+0Hz",
    volume: str = "+0%",
    temperature: float = 0.75,
    trim_silence_flag: bool = False,
    normalize_loudness_flag: bool = False,
    generate_subtitles: bool = False,
    fps: int = 30,
    max_words_per_cue: int = 6,
    offline_only: bool = False,
    allow_fallback: bool = True,
) -> SynthesisResult:
    """Smart engine router with automatic offline/error fallback & post-processing."""
    if not text or not text.strip():
        raise TtsError("Text cannot be empty.")

    warnings: list[str] = []
    eng_key = (engine or "auto").strip().lower()

    if voice_profile or reference_audio:
        chosen = get_engine("xtts_v2")
    elif eng_key != "auto":
        chosen = get_engine(eng_key)
    elif offline_only:
        avail_ids = {e.id: e for e in available_engines()}
        chosen = avail_ids.get("xtts_v2") or avail_ids.get("pyttsx3")
        if chosen is None:
            raise TtsError("No offline TTS engine is available.")
    else:
        avail_ids = {e.id: e for e in available_engines()}
        chosen = avail_ids.get("edge-tts") or avail_ids.get("xtts_v2") or avail_ids["pyttsx3"]

    async def _run_on(target_eng: BaseTtsEngine, explicit_voice_id: str) -> SynthesisResult:
        if target_eng.id == "xtts_v2":
            v_obj = None
            if not (voice_profile or reference_audio):
                v_obj = pick_default_voice(
                    target_eng, language=language, gender=gender, voice_id=explicit_voice_id
                )
            return await target_eng.synthesize_rich_async(
                text=text,
                voice=v_obj,
                output_path=output_path,
                language=language,
                voice_profile=voice_profile,
                reference_audio=reference_audio,
                speed=speed,
                temperature=temperature,
                trim=trim_silence_flag,
                normalize=normalize_loudness_flag,
                generate_subtitles=generate_subtitles,
                fps=fps,
                max_words_per_cue=max_words_per_cue,
            )

        v_obj = pick_default_voice(
            target_eng, language=language, gender=gender, voice_id=explicit_voice_id
        )
        extra_kwargs: dict[str, _t.Any] = {}
        if target_eng.id == "edge-tts":
            extra_kwargs["pitch"] = pitch
            extra_kwargs["volume"] = volume
        return await target_eng.synthesize_rich_async(
            text=text,
            voice=v_obj,
            output_path=output_path,
            speed=speed,
            trim=trim_silence_flag,
            normalize=normalize_loudness_flag,
            generate_subtitles=generate_subtitles,
            fps=fps,
            max_words_per_cue=max_words_per_cue,
            **extra_kwargs,
        )

    try:
        return await _run_on(chosen, voice_id)
    except TtsError as primary_err:
        if not allow_fallback or eng_key not in ("auto", "edge-tts", "edge"):
            raise
        if chosen.id != "edge-tts":
            raise

        avail_ids = {e.id: e for e in available_engines()}
        fallback_order = [
            avail_ids[k] for k in ("xtts_v2", "pyttsx3") if k in avail_ids
        ]
        for fb in fallback_order:
            warnings.append(
                f"Primary engine '{chosen.id}' failed ({primary_err}); fell back to '{fb.id}'."
            )
            try:
                res = await _run_on(fb, "")
                res.fallback_used = True
                res.warnings = warnings + res.warnings
                return res
            except TtsError as fb_err:
                warnings.append(f"Fallback '{fb.id}' also failed ({fb_err}).")
        raise primary_err


def smart_synthesize(text: str, output_path: str, **kwargs: _t.Any) -> SynthesisResult:
    """Synchronous wrapper around :func:`smart_synthesize_async`."""
    return _run_coro_sync(lambda: smart_synthesize_async(text, output_path, **kwargs))


# ---------------------------------------------------------------------------
# Console smoke test (python -m tts_app.tts_engine <text>)
# ---------------------------------------------------------------------------


def _main() -> int:  # pragma: no cover - dev helper
    text = " ".join(sys.argv[1:]).strip() or "Hello from the TTS engine smoke test!"
    out = os.path.join(tempfile.gettempdir(), "tts_pretest.mp3")
    res = smart_synthesize(text=text, output_path=out, generate_subtitles=True)
    print(f"engine: {res.engine} | voice: {res.voice_id}")
    print(f"audio: {res.audio_path} ({res.duration_s}s)")
    if res.srt_path:
        print(f"subtitles: {res.srt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
