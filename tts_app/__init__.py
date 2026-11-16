"""Smart Text-to-Speech desktop app, video narration pipeline, and MCP server."""

__version__ = "2.0.0"

from tts_app.models import (  # noqa: F401
    CaptionCue,
    SynthesisResult,
    TtsError,
    Voice,
    VoiceProfile,
    WordTiming,
)
