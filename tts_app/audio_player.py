"""Audio playback layer.

Wraps QtMultimedia's ``QMediaPlayer`` (Qt5 API) behind a small, signal-based
interface so the GUI can Play / Pause / Stop and drive a progress slider
without knowing the raw Qt multimedia details.
"""

from __future__ import annotations

from PyQt5.QtCore import QObject, QUrl, pyqtSignal
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer

# Player states relayed to the UI.
STOPPED = 0
PLAYING = 1
PAUSED = 2


class AudioPlayer(QObject):
    """Thin wrapper around QMediaPlayer for single-file audio playback."""

    #: (ms) current playback position within the media.
    positionChanged = pyqtSignal(int)
    #: (ms) total duration of the current media (0 while unknown).
    durationChanged = pyqtSignal(int)
    #: One of STOPPED / PLAYING / PAUSED.
    stateChanged = pyqtSignal(int)
    #: Emitted when playback reaches the end of the media.
    finished = pyqtSignal()
    #: Emitted with a human-readable message when playback fails.
    errorOccurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._player = QMediaPlayer()

        self._player.stateChanged.connect(self._emit_state)
        self._player.positionChanged.connect(self._relay_position)
        self._player.durationChanged.connect(self._relay_duration)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.error.connect(self._on_error)

    # -- public API ---------------------------------------------------------
    def load(self, file_path: str) -> None:
        """Load an audio file from disk, ready to play."""
        self._player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

    def play(self) -> None:
        """Start (or resume after pause) playback."""
        self._player.play()

    def pause(self) -> None:
        """Pause playback, keeping the current position."""
        self._player.pause()

    def stop(self) -> None:
        """Stop playback and reset to the start of the media."""
        self._player.stop()

    def set_position(self, ms: int) -> None:
        """Seek to ``ms`` milliseconds (used when scrubbing the progress bar)."""
        self._player.setPosition(ms)

    # -- internal signal handlers -------------------------------------------
    def _relay_position(self, ms: int) -> None:
        self.positionChanged.emit(ms)

    def _relay_duration(self, ms: int) -> None:
        self.durationChanged.emit(ms)

    def _emit_state(self, state) -> None:
        if state == QMediaPlayer.PlayingState:
            self.stateChanged.emit(PLAYING)
        elif state == QMediaPlayer.PausedState:
            self.stateChanged.emit(PAUSED)
        else:
            self.stateChanged.emit(STOPPED)

    def _on_media_status(self, status) -> None:
        if status == QMediaPlayer.EndOfMedia:
            self.finished.emit()

    def _on_error(self, *_) -> None:
        self.errorOccurred.emit(
            self._player.errorString() or "Unknown playback error."
        )