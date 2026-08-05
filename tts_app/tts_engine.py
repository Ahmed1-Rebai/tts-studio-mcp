"""TTS engine layer.

Abstraction over the text-to-speech backends behind a single, uniform interface
used by the GUI. Provides:

    * ``Voice``        - data object describing one selectable voice.
    * ``TtsError``      - exception surfaced to the user as a friendly message.
    * ``BaseTtsEngine`` - interface every engine implements.
    * ``EdgeTtsEngine`` - primary engine (online, huge multilingual catalog with
      gender metadata per voice).
    * ``Pyttsx3Engine`` - offline fallback using the operating system voices.

Voice and language lists are always discovered dynamically from the underlying
library at runtime, never hard-coded, so new voices show up automatically.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import typing as _t

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class Voice:
    """A single selectable voice.

    Attributes
    ----------
    id: Unique identifier understood by the TTS library.
    name: Human friendly display name.
    locale: BCP-47 locale / primary language tag (e.g. "en-US").
    gender: "Male", "Female" or "Unknown".
    age: "Adult", "Child" or "Unknown". Not every engine exposes age; the
        default is "Adult" when the engine cannot tell.
    """

    __slots__ = ("id", "name", "locale", "gender", "age")

    def __init__(self, voice_id, name, locale, gender="Female", age="Adult"):
        self.id = voice_id
        self.name = name
        self.locale = locale or ""
        self.gender = gender
        self.age = age

    def __repr__(self) -> str:
        return (
            f"Voice<{self.id} | {self.name} | {self.locale} "
            f"| {self.gender} | {self.age}>"
        )


class TtsError(Exception):
    """Raised when an engine cannot complete a request; the GUI shows the
    message inside a friendly error dialog."""


# ---------------------------------------------------------------------------
# Locale / preview helpers
# ---------------------------------------------------------------------------

_PREVIEW_TEXT = {
    "en": "Hello, this is a sample of the voice.",
    "fr": "Bonjour, ceci est un exemple de la voix.",
    "es": "Hola, esta es una muestra de la voz.",
    "de": "Hallo, das ist eine Beispielstimme.",
    "ar": "مرحبا، هذا مثال على الصوت.",
    "it": "Ciao, questo è un esempio di voce.",
    "pt": "Olá, esta é uma amostra da voz.",
    "ru": "Привет, это пример голоса.",
    "ja": "こんにちは、これは音声のサンプルです。",
    "zh": "你好，这是语音示例。",
    "tr": "Merhaba, bu sesin bir örneğidir.",
    "hi": "नमस्ते, यह आवाज़ का एक नमूना है।",
}

_DEFAULT_PREVIEW = "Hello, this is a sample of the voice."

#: Human-readable display names for the language subtags. Covers the languages
#: offered by edge-tts; unknown tags fall back to their subtag code.
_LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
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
    "zh": "Chinese",
    "zh-CN": "Chinese (Simplified)",
    "yue": "Cantonese",
}


def language_display_name(subtag: str) -> str:
    """Full human-readable name for a language subtag ('ar' -> 'Arabic').

    Falls back to the subtag itself when the tag is not in the catalogue.
    """
    if not subtag:
        return ""
    key = subtag.lower()
    return _LANGUAGE_NAMES.get(key, _LANGUAGE_NAMES.get(key.split("-")[0], subtag))


def language_subtag(locale: str) -> str:
    """Return the primary language subtag (lower-cased) for a locale.

    'en-US' -> 'en', 'fr-FR' -> 'fr', 'zh-CN' -> 'zh'
    """
    if not locale:
        return ""
    return locale.replace("_", "-").split("-")[0].lower()


def preview_text_for(language: str) -> str:
    """A short sample sentence used to audition/preview a voice."""
    if not language:
        return _DEFAULT_PREVIEW
    key = language.lower()
    if key not in _PREVIEW_TEXT:
        base = key.split("-")[0]
        key = base if base in _PREVIEW_TEXT else "en"
    return _PREVIEW_TEXT.get(key, _DEFAULT_PREVIEW)


# ---------------------------------------------------------------------------
# Base engine interface
# ---------------------------------------------------------------------------


class BaseTtsEngine:
    """Common interface implemented by every TTS backend.

    The shared logic caches the voice catalogue as ``Voice`` objects. Lists
    (languages, voices, genders) are all derived from that catalogue.
    """

    #: Short label shown in the engine selector dropdown.
    label = "Base"

    def __init__(self) -> None:
        #: list of :class:`Voice`; cached after the first call.
        self._voices: _t.List[Voice] = []

    # -- catalog discovery -----------------------------------------------------
    def _discover(self) -> _t.List[Voice]:
        """Query the library and return the full catalogue of voices.

        Subclasses must implement this. It is called once and cached.
        """
        raise NotImplementedError

    def voices(self) -> _t.List[Voice]:
        """Return the full, cached voice catalogue."""
        if not self._voices:
            self._voices = self._discover()
        return self._voices

    def list_languages(self) -> _t.List[str]:
        """De-duplicated, sorted list of every available language subtag."""
        return sorted(
            {language_subtag(v.locale) for v in self.voices() if v.locale}
        )

    def list_voices(self, language: str = "") -> _t.List[Voice]:
        """All voices for a language subtag, or every voice when ``''``.

        Uses the *primary* language subtag so 'en' matches 'en-US'.
        """
        if not language:
            return list(self.voices())
        lang = language.lower()
        return [v for v in self.voices() if language_subtag(v.locale) == lang]

    def genders(self, language: str = "") -> _t.List[str]:
        """Distinct gender labels available for a language."""
        seen = [v.gender for v in self.list_voices(language)]
        uniq = sorted({g for g in seen if g and g != "Unknown"})
        if any(g == "Unknown" for g in seen):
            uniq.append("Unknown")
        return uniq or ["Unknown"]

    # -- synthesis -------------------------------------------------------------
    def synthesize_to_file(self, text, voice, output_path) -> str:
        """Synthesise ``text`` with ``voice`` (a Voice object) to ``output_path``.

        Return ``output_path`` on success. Implemented by subclasses.
        """
        raise NotImplementedError(
            f"{self.label} does not support this operation"
        )


# ---------------------------------------------------------------------------
# edge-tts engine (primary)
# ---------------------------------------------------------------------------


class EdgeTtsEngine(BaseTtsEngine):
    """edge-tts backend.

    Online, free, and includes hundreds of natural voices with gender metadata.
    Each voice name is composed as ``<locale>-<Name><Gender>Neural`` (e.g.
    'en-US-ChristopherNeural' = male adult). Age is derived only when the
    friendly name hints at it, otherwise it stays 'Adult'.
    """

    label = "edge-tts (online)"

    def _discover(self) -> _t.List[Voice]:
        import edge_tts

        results = edge_tts.list_voices()
        # list_voices can return a coroutine in some versions -- flatten it.
        if asyncio.iscoroutine(results):
            loop = asyncio.new_event_loop()
            try:
                results = loop.run_until_complete(results)
            finally:
                loop.close()

        catalog: _t.List[Voice] = []
        for item in results:
            short = item.get("ShortName", "")
            friendly = item.get("FriendlyName", "")
            gender = item.get("Gender", "Unknown") or "Unknown"
            age = _infer_age(friendly, gender)
            catalog.append(
                Voice(
                    voice_id=short,
                    name=friendly or short,
                    locale=item.get("Locale", ""),
                    gender=gender,
                    age=age,
                )
            )
        return catalog

    def synthesize_to_file(self, text, voice, output_path) -> str:
        import edge_tts

        communicate = edge_tts.Communicate(text=text, voice=voice.id)
        try:
            asyncio.run(communicate.save(output_path))
        except Exception as exc:  # network / server failures
            raise TtsError(f"edge-tts failed: {exc}") from exc
        return output_path


def _infer_age(friendly: str, gender: str) -> str:
    """Disambiguate_child-friendly voice names => age tag.

    edge-tts names adult voices like 'Aria', and some friendly names mention
    'Child'. Fall back to 'Adult' (nearly everything is adult).
    """
    if not friendly:
        return "Adult"
    lower = friendly.lower()
    if "child" in lower or "kid" in lower:
        return "Child"
    return "Adult"


# ---------------------------------------------------------------------------
# pyttsx3 engine (offline fallback)
# ---------------------------------------------------------------------------


class Pyttsx3Engine(BaseTtsEngine):
    """Fully-offline backend using the operating system's installed voices.

    Voice gender metadata is not exposed by pyttsx3 (the OS offers no reliable
    gender field on most platforms), so the UI will only show a limited set.
    """

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
                # pyttsx3 voices expose languages in some platforms.
                langs = getattr(v, "languages", None)
                if langs and len(langs) > 0:
                    locale = _decode_lang(langs[0])
                voices.append(
                    Voice(
                        voice_id=v.id,
                        name=name,
                        locale=locale,
                        gender="Unknown",
                        age="Adult",
                    )
                )
            return voices
        finally:
            engine.stop()

    def synthesize_to_file(self, text, voice, output_path) -> str:
        import pyttsx3

        engine = pyttsx3.init()
        try:
            engine.setProperty("voice", voice.id)
            output_path = _wave_path(output_path)
            engine.save_to_file(text, output_path)
            engine.runAndWait()
        except Exception as exc:
            raise TtsError(f"pyttsx3 failed: {exc}") from exc
        finally:
            engine.stop()
        return output_path


def _decode_lang(value: _t.Any) -> str:
    """pyttsx3 may give bytes or str; normalise to a str locale."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _wave_path(path: str) -> str:
    """pyttsx3 only writes .wav/.mp3 — returns a .wav path."""
    root, _ext = os.path.splitext(path)
    return root + ".wav"


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


def available_engines() -> _t.List[BaseTtsEngine]:
    """Instantiate every engine whose dependency is importable.

    A broken optional engine is silently skipped so the app still launches.
    """
    engines: _t.List[BaseTtsEngine] = []
    try:
        import edge_tts  # noqa: F401  (availability probe)

        engines.append(EdgeTtsEngine())
    except Exception:
        pass
    try:
        import pyttsx3  # noqa: F401  (availability probe)

        engines.append(Pyttsx3Engine())
    except Exception:
        pass
    if not engines:
        raise TtsError("No TTS engine available. Install edge-tts or pyttsx3.")
    return engines


def get_engine(name: str) -> BaseTtsEngine:
    """Return a specific engine instance by its ``label``."""
    for engine in available_engines():
        if engine.label == name:
            return engine
    raise TtsError(f"Unknown engine: {name}")


# ---------------------------------------------------------------------------
# Console smoke test (python -m tts_app.tts_engine <text>)
# ---------------------------------------------------------------------------


def _main() -> int:  # pragma: no cover - dev helper
    text = " ".join(sys.argv[1:]) or "Hello, world!"
    engine = EdgeTtsEngine()
    language = engine.list_languages()[:1]
    label = language[0] if language else "en"
    voices = engine.list_voices(label)
    if not voices:
        print("No voices found for that language.")
        return 1
    print(f"engine applied:\n{engine.label}\nlanguages: {len(engine.list_languages())}")
    print(f"voices[{label}]={len(voices)} sample:")
    for v in voices[:8]:
        print(f"  - {v.gender:6} {v.id}")
    out = os.path.join(tempfile.gettempdir(), "tts_pretest.mp3")
    print(f"\nsynthesising '{text}' with {voices[0].id} -> {out}")
    print(engine.synthesize_to_file(text=" ".join(sys.argv[1:]), voice=voices[0], output_path=out))
    return 0


if __name__ == "__main__":
    sys.exit(_main())