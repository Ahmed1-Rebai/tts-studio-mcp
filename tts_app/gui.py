"""Main window for the Text-to-Speech desktop application.

The window guides the user through:

    1. Picking an engine and a language (dropdowns populated dynamically).
    2. Filtering the generated voice list by gender and age.
    3. Entering text — either pasted/typed into the text box or loaded from a
       .txt file.
    4. Play / Pause / Stop with a progress slider, plus Export to MP3/WAV.

Synthesis can take a second or two (network), so it runs on a background
``QThread`` worker and the UI is only updated from its finished signals.
"""

from __future__ import annotations

import os
import tempfile
import typing as _t

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from tts_app.audio_player import AudioPlayer, PAUSED, PLAYING
from tts_app.tts_engine import (
    BaseTtsEngine,
    TtsError,
    available_engines,
    language_display_name,
)

_ANY = "Any"


class _SynthesisWorker(QThread):
    """Run a single synthesis request on a background thread."""

    succeeded = pyqtSignal(str)  # path of the produced audio file
    failed = pyqtSignal(str)     # user-facing error message

    def __init__(self, engine, text, voice, output_path) -> None:
        super().__init__()
        self._engine = engine
        self._text = text
        self._voice = voice
        self._output = output_path

    def run(self) -> None:
        try:
            result = self._engine.synthesize_to_file(
                self._text, self._voice, self._output
            )
        except TtsError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # unexpected engine bugs
            self.failed.emit(f"Synthesis failed: {exc}")
        else:
            self.succeeded.emit(result)


class MainWindow(QMainWindow):
    """The application's main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Text-to-Speech")
        self.resize(780, 760)

        self._engines = available_engines()
        self._player = AudioPlayer()
        self._worker = None

        #: key (text, voice_id) that produced the current playable audio.
        self._loaded_key = None
        self._playable_path = None
        self._scrubbing = False

        self._build_ui()
        self._wire_signals()
        self._seed_engine()

    # ----------------------------------------------------------------- #
    # UI construction
    # ----------------------------------------------------------------- #
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # -- Text source ---------------------------------------------------
        source_box = QGroupBox("Text source")
        source = QVBoxLayout(source_box)

        file_row = QHBoxLayout()
        self._btn_load = QPushButton("Load .txt file\u2026")
        self._file_label = QLabel("No file selected")
        self._file_label.setStyleSheet("color: #777;")
        file_row.addWidget(self._btn_load)
        file_row.addWidget(self._file_label, 1)
        source.addLayout(file_row)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setPlaceholderText(
            "Paste or type the text you want to hear\u2026"
        )
        self._text_edit.setFont(QFont("Sans Serif", 11))
        self._text_edit.setMinimumHeight(170)
        source.addWidget(self._text_edit)
        layout.addWidget(source_box)

        # -- Voice selection -------------------------------------------------
        voice_box = QGroupBox("Voice")
        form = QFormLayout(voice_box)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._engine_combo = QComboBox()
        for engine in self._engines:
            self._engine_combo.addItem(engine.label)
        form.addRow("Engine:", self._engine_combo)

        self._language_combo = QComboBox()
        form.addRow("Language:", self._language_combo)

        self._voice_combo = QComboBox()
        form.addRow("Voice:", self._voice_combo)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Gender:"))
        self._gender_combo = QComboBox()
        filters.addWidget(self._gender_combo)
        filters.addSpacing(16)
        filters.addWidget(QLabel("Age:"))
        self._age_combo = QComboBox()
        filters.addWidget(self._age_combo)
        filters.addStretch(1)
        form.addRow("Filter:", filters)
        layout.addWidget(voice_box)

        # -- Playback ----------------------------------------------------------
        play_box = QGroupBox("Playback")
        play = QVBoxLayout(play_box)

        self._progress = QSlider(Qt.Horizontal)
        self._progress.setEnabled(False)
        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setAlignment(Qt.AlignCenter)
        play.addWidget(self._progress)
        play.addWidget(self._time_label)

        buttons = QHBoxLayout()
        self._btn_play = QPushButton("Play")
        self._btn_pause = QPushButton("Pause")
        self._btn_stop = QPushButton("Stop")
        self._btn_export = QPushButton("Export\u2026")
        for button in (self._btn_play, self._btn_pause,
                       self._btn_stop, self._btn_export):
            buttons.addWidget(button)
        buttons.addStretch(1)
        play.addLayout(buttons)
        layout.addWidget(play_box)

        self._status = QLabel("Ready.")
        layout.addWidget(self._status)

    def _wire_signals(self) -> None:
        self._btn_load.clicked.connect(self.on_load_file)
        self._btn_play.clicked.connect(self.on_play)
        self._btn_pause.clicked.connect(self._player.pause)
        self._btn_stop.clicked.connect(self.on_stop)
        self._btn_export.clicked.connect(self.on_export)

        self._engine_combo.currentIndexChanged.connect(self._seed_engine)
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)
        self._gender_combo.currentIndexChanged.connect(self._update_voices)
        self._age_combo.currentIndexChanged.connect(self._update_voices)

        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.stateChanged.connect(self._on_player_state)
        self._player.finished.connect(self._on_finished)
        self._player.errorOccurred.connect(self._on_player_error)

        self._progress.sliderPressed.connect(self._start_scrub)
        self._progress.sliderReleased.connect(self._stop_scrub)

    # ----------------------------------------------------------------- #
    # Dynamic dropdown population
    # ----------------------------------------------------------------- #
    def current_engine(self) -> BaseTtsEngine:
        return self._engines[self._engine_combo.currentIndex()]

    def _seed_engine(self) -> None:
        """Populate the language dropdown for the newly chosen engine.

        Items display the full language name (e.g. 'Arabic'); the ISO subtag is
        kept as userData so the engine can match voices.
        """
        engine = self.current_engine()
        languages = engine.list_languages()
        self._block_set_items_voices(self._language_combo, languages)
        if languages:
            self._populate_for_language(languages[0])

    def _selected_language(self) -> str:
        """The current language subtag (from the combo's userData)."""
        data = self._language_combo.currentData()
        if data:
            return data
        return self._language_combo.currentText()

    def _on_language_changed(self, *_args) -> None:
        self._populate_for_language(self._selected_language())

    def _populate_for_language(self, subtag: str) -> None:
        engine = self.current_engine()
        voices = engine.list_voices(subtag)
        genders = engine.genders(subtag)
        ages = sorted({v.age for v in voices})

        self._block_set_items(self._gender_combo, [_ANY] + genders)
        self._block_set_items(self._age_combo, [_ANY] + ages)
        self._update_voices()

    def _update_voices(self) -> None:
        """Recompute the voice list from language + gender + age filters."""
        engine = self.current_engine()
        language = self._selected_language()
        gender = self._gender_combo.currentText()
        age = self._age_combo.currentText()

        voices = engine.list_voices(language)
        if gender != _ANY:
            voices = [v for v in voices if v.gender == gender]
        if age != _ANY:
            voices = [v for v in voices if v.age == age]
        voices.sort(key=lambda v: (v.gender != "Female", v.name))

        self._voice_combo.blockSignals(True)
        self._voice_combo.clear()
        for voice in voices:
            self._voice_combo.addItem(voice.name, userData=voice)
        self._voice_combo.setEnabled(len(voices) > 0)
        self._voice_combo.blockSignals(False)

    @staticmethod
    def _block_set_items(combo: QComboBox, items) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.blockSignals(False)

    @staticmethod
    def _block_set_items_voices(combo: QComboBox, subtags) -> None:
        """Populate a combo with full language names, subtag as userData."""
        combo.blockSignals(True)
        combo.clear()
        for subtag in subtags:
            combo.addItem(language_display_name(subtag), userData=subtag)
        combo.blockSignals(False)

    def selected_voice(self):
        return self._voice_combo.currentData() if self._voice_combo.count() else None

    # ----------------------------------------------------------------- #
    # User actions
    # ----------------------------------------------------------------- #
    def on_load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a text file", "",
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError as exc:
            self.error(f"Could not read the file:\n{exc}")
            return
        self._text_edit.setPlainText(content)
        self._file_label.setText(path)
        self._status.setText(f"Loaded {os.path.basename(path)}")

    def _validation_error(self) -> _t.Optional[str]:
        """Return a user-facing problem message, or None when input is valid."""
        if not self._text_edit.toPlainText().strip():
            return "The text is empty. Add some text or load a .txt file."
        if self.selected_voice() is None:
            return "No voice is available for this selection."
        return None

    def on_play(self) -> None:
        problem = self._validation_error()
        if problem:
            self.error(problem)
            return

        text = self._text_edit.toPlainText()
        voice = self.selected_voice()
        key = (text, voice.id)

        # Reuse the already-generated audio when nothing changed.
        if key == self._loaded_key and self._playable_path \
                and os.path.exists(self._playable_path):
            self._start_playback(self._playable_path)
            return

        output = _new_tmp_path(".mp3")
        self._status.setText("Synthesising\u2026")
        self._set_busy(True)
        self._start_worker(self.current_engine(), text, voice, output,
                           self._on_synth_done, self._on_synth_failed)

    def on_stop(self) -> None:
        self._player.stop()
        self._status.setText("Ready.")

    def on_export(self) -> None:
        problem = self._validation_error()
        if problem:
            self.error(problem)
            return

        path, selected = QFileDialog.getSaveFileName(
            self, "Export audio", "speech.mp3",
            "MP3 audio (*.mp3);;WAV audio (*.wav)",
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".mp3" if "MP3" in selected else ".wav"

        self._status.setText("Exporting\u2026")
        self._set_busy(True)
        self._start_worker(self.current_engine(), self._text_edit.toPlainText(),
                           self.selected_voice(), path,
                           self._on_export_done, self._on_export_failed)

    # ----------------------------------------------------------------- #
    # Background worker plumbing
    # ----------------------------------------------------------------- #
    def _start_worker(self, engine, text, voice, output,
                      on_done, on_failed) -> None:
        worker = _SynthesisWorker(engine, text, voice, output)
        worker.succeeded.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()
        self._worker = worker

    def _on_synth_done(self, path: str) -> None:
        self._set_busy(False)
        self._loaded_key = (self._text_edit.toPlainText(),
                            self.selected_voice().id)
        self._playable_path = path
        self._start_playback(path)

    def _on_synth_failed(self, message: str) -> None:
        self._set_busy(False)
        self.error(message)

    def _on_export_done(self, path: str) -> None:
        self._set_busy(False)
        self._status.setText(f"Exported to {os.path.basename(path)}")
        QMessageBox.information(self, "Export complete", f"Saved to:\n{path}")

    def _on_export_failed(self, message: str) -> None:
        self._set_busy(False)
        self.error(message)

    def _start_playback(self, path: str) -> None:
        self._player.load(path)
        self._player.play()
        self._status.setText("Playing\u2026")

    # ----------------------------------------------------------------- #
    # Player signals
    # ----------------------------------------------------------------- #
    def _on_position(self, ms: int) -> None:
        if not self._scrubbing:
            self._progress.setValue(ms)
        self._time_label.setText(
            f"{_fmt(ms)} / {_fmt(self._progress.maximum())}"
        )

    def _on_duration(self, ms: int) -> None:
        self._progress.setMaximum(ms if ms > 0 else 1)
        self._progress.setEnabled(ms > 0)
        self._time_label.setText(f"00:00 / {_fmt(ms)}")

    def _on_player_state(self, state: int) -> None:
        if state == PLAYING:
            self._status.setText("Playing\u2026")
        elif state == PAUSED:
            self._status.setText("Paused")
        else:
            self._status.setText("Ready.")

    def _on_finished(self) -> None:
        self._progress.setValue(0)
        self._status.setText("Finished.")

    def _on_player_error(self, message: str) -> None:
        self.error(message)

    # ----------------------------------------------------------------- #
    # Progress slider scrubbing
    # ----------------------------------------------------------------- #
    def _start_scrub(self) -> None:
        self._scrubbing = True

    def _stop_scrub(self) -> None:
        self._scrubbing = False
        self._player.set_position(self._progress.value())

    # ----------------------------------------------------------------- #
    # Misc helpers
    # ----------------------------------------------------------------- #
    def _set_busy(self, busy: bool) -> None:
        self._btn_play.setEnabled(not busy)
        self._btn_export.setEnabled(not busy)

    def error(self, message: str) -> None:
        self._status.setText("Error")
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(500)
        self._player.stop()
        super().closeEvent(event)


def _fmt(ms: int) -> str:
    """Format milliseconds as 'mm:ss'."""
    ms = max(0, int(ms))
    seconds, _ = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _new_tmp_path(extension: str) -> str:
    """Create a fresh temporary filename for generated audio."""
    fd, path = tempfile.mkstemp(suffix=extension, prefix="tts_")
    os.close(fd)
    return path
