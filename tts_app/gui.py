"""Main window for the Text-to-Speech desktop application."""

from __future__ import annotations

import os
import sys
import tempfile
import typing as _t


def _fix_frozen_gstreamer() -> None:
    """Help a PyInstaller-frozen app find system GStreamer plugins on Linux."""
    if os.environ.get("GST_PLUGIN_SYSTEM_PATH"):
        return
    candidates = [
        "/usr/lib/x86_64-linux-gnu/gstreamer-1.0",
        "/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
        "/usr/lib/arm-linux-gnueabihf/gstreamer-1.0",
        "/usr/lib64/gstreamer-1.0",
        "/usr/lib/gstreamer-1.0",
    ]
    for path in candidates:
        if os.path.isdir(path):
            os.environ["GST_PLUGIN_SYSTEM_PATH"] = path
            return


_fix_frozen_gstreamer()

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import (
    QAudioEncoderSettings,
    QAudioRecorder,
    QMediaRecorder,
    QMultimedia,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from tts_app.audio_player import PAUSED, PLAYING, AudioPlayer
from tts_app.audio_utils import safe_remove
from tts_app.models import SynthesisResult, TtsError, Voice
from tts_app.tts_engine import (
    BaseTtsEngine,
    available_engines,
    language_display_name,
    language_subtag,
)
from tts_app.voice_profiles import (
    check_hardware_requirements,
    delete_profile,
    enroll_voice_profile,
    list_profiles,
)

_ANY = "Any"
_SAMPLE_VOICE_OPTION = "__custom_sample__"


class _SynthesisWorker(QThread):
    """Run a single synthesis request on a background thread."""

    succeeded = pyqtSignal(object)  # SynthesisResult
    failed = pyqtSignal(str)

    def __init__(
        self,
        engine: BaseTtsEngine,
        text: str,
        voice: _t.Optional[Voice],
        output_path: str,
        *,
        language: str,
        speed: float,
        generate_subtitles: bool,
        normalize_loudness: bool,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._text = text
        self._voice = voice
        self._output = output_path
        self._language = language
        self._speed = speed
        self._generate_subtitles = generate_subtitles
        self._normalize_loudness = normalize_loudness

    def run(self) -> None:
        try:
            kwargs: dict[str, _t.Any] = {
                "speed": self._speed,
                "generate_subtitles": self._generate_subtitles,
                "normalize": self._normalize_loudness,
            }
            if self._engine.id == "xtts_v2":
                kwargs["language"] = self._language
            result = self._engine.synthesize_rich(
                self._text, self._voice, self._output, **kwargs
            )
        except TtsError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Synthesis failed: {exc}")
        else:
            self.succeeded.emit(result)


class _EnrollWorker(QThread):
    """Enroll a reusable XTTS voice profile on a background thread."""

    succeeded = pyqtSignal(object)  # VoiceProfile
    failed = pyqtSignal(str)

    def __init__(self, name: str, sample_path: str, language: str) -> None:
        super().__init__()
        self._name = name
        self._sample_path = sample_path
        self._language = language

    def run(self) -> None:
        try:
            prof = enroll_voice_profile(
                name=self._name,
                audio_paths=[self._sample_path],
                language=self._language,
                preprocess=True,
            )
        except TtsError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Voice enrollment failed: {exc}")
        else:
            self.succeeded.emit(prof)


class MainWindow(QMainWindow):
    """The application's main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Text-to-Speech")
        self.resize(820, 820)

        self._engines = available_engines()
        self._player = AudioPlayer()
        self._worker: _t.Optional[QThread] = None

        #: key (engine_id, text, voice_id, language, speed) for cached playback.
        self._loaded_key: _t.Optional[tuple[str, str, str, str, float]] = None
        self._playable_path: _t.Optional[str] = None
        self._temp_files: list[str] = []
        self._scrubbing = False

        #: path of the user's reference audio for voice-cloning engines.
        self._sample_path: _t.Optional[str] = None
        self._recorder: _t.Optional[QAudioRecorder] = None
        self._record_path: _t.Optional[str] = None

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
        self._text_edit.setMinimumHeight(160)
        source.addWidget(self._text_edit)
        layout.addWidget(source_box)

        # -- Voice selection -------------------------------------------------
        voice_box = QGroupBox("Voice & Prosody")
        form = QFormLayout(voice_box)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._engine_combo = QComboBox()
        for engine in self._engines:
            self._engine_combo.addItem(engine.label)
        form.addRow("Engine:", self._engine_combo)

        self._language_combo = QComboBox()
        form.addRow("Language:", self._language_combo)

        # -- Voice catalog vs. XTTS cloning/profiles -----------------------
        self._voice_stack = QStackedWidget()

        catalog_page = QWidget()
        cv = QVBoxLayout(catalog_page)
        cv.setContentsMargins(0, 0, 0, 0)
        self._voice_combo = QComboBox()
        cv.addWidget(self._voice_combo)
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Gender:"))
        self._gender_combo = QComboBox()
        filters.addWidget(self._gender_combo)
        filters.addSpacing(16)
        filters.addWidget(QLabel("Age:"))
        self._age_combo = QComboBox()
        filters.addWidget(self._age_combo)
        filters.addStretch(1)
        cv.addLayout(filters)
        self._voice_stack.addWidget(catalog_page)

        xtts_page = QWidget()
        sv = QVBoxLayout(xtts_page)
        sv.setContentsMargins(0, 0, 0, 0)

        self._xtts_target_combo = QComboBox()
        sv.addWidget(self._xtts_target_combo)

        sample = QHBoxLayout()
        self._btn_sample = QPushButton("Choose voice sample\u2026")
        self._btn_record = QPushButton("Record")
        self._btn_save_profile = QPushButton("Save as Profile\u2026")
        self._btn_delete_profile = QPushButton("Delete Profile")
        self._btn_hw_info = QPushButton("Training & Hardware Info")
        self._sample_label = QLabel("No sample selected")
        self._sample_label.setStyleSheet("color: #777;")
        sample.addWidget(self._btn_sample)
        sample.addWidget(self._btn_record)
        sample.addWidget(self._btn_save_profile)
        sample.addWidget(self._btn_delete_profile)
        sample.addWidget(self._btn_hw_info)
        sample.addWidget(self._sample_label, 1)
        sv.addLayout(sample)

        self._sample_hint = QLabel(
            "Choose a saved voice profile, a built-in XTTS speaker, or record/pick "
            "6-30s of clear speech. Click 'Save as Profile' to reuse your voice fast."
        )
        self._sample_hint.setWordWrap(True)
        self._sample_hint.setStyleSheet("color: #777;")
        sv.addWidget(self._sample_hint)
        self._voice_stack.addWidget(xtts_page)

        form.addRow("Voice:", self._voice_stack)

        # -- Speed control -------------------------------------------------
        speed_row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Horizontal)
        self._speed_slider.setMinimum(75)
        self._speed_slider.setMaximum(150)
        self._speed_slider.setValue(100)
        self._speed_slider.setSingleStep(5)
        self._speed_label = QLabel("1.00x")
        self._speed_label.setMinimumWidth(45)
        speed_row.addWidget(self._speed_slider, 1)
        speed_row.addWidget(self._speed_label)
        form.addRow("Speed:", speed_row)

        layout.addWidget(voice_box)

        # -- Playback & Export ---------------------------------------------
        play_box = QGroupBox("Playback & Video Export")
        play = QVBoxLayout(play_box)

        self._progress = QSlider(Qt.Horizontal)
        self._progress.setEnabled(False)
        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setAlignment(Qt.AlignCenter)
        play.addWidget(self._progress)
        play.addWidget(self._time_label)

        opts = QHBoxLayout()
        self._chk_subtitles = QCheckBox("Also export video subtitles (.srt, .vtt, .timings.json)")
        self._chk_normalize = QCheckBox("Normalize loudness (-16 LUFS)")
        opts.addWidget(self._chk_subtitles)
        opts.addWidget(self._chk_normalize)
        opts.addStretch(1)
        play.addLayout(opts)

        buttons = QHBoxLayout()
        self._btn_play = QPushButton("Play")
        self._btn_pause = QPushButton("Pause")
        self._btn_stop = QPushButton("Stop")
        self._btn_export = QPushButton("Export\u2026")
        for button in (
            self._btn_play,
            self._btn_pause,
            self._btn_stop,
            self._btn_export,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        play.addLayout(buttons)
        layout.addWidget(play_box)

        self._status = QLabel("Ready.")
        layout.addWidget(self._status)

    def _wire_signals(self) -> None:
        self._btn_load.clicked.connect(self.on_load_file)
        self._btn_sample.clicked.connect(self.on_choose_sample)
        self._btn_record.clicked.connect(self.on_record)
        self._btn_save_profile.clicked.connect(self.on_save_profile)
        self._btn_delete_profile.clicked.connect(self.on_delete_profile)
        self._btn_hw_info.clicked.connect(self.on_show_training_info)
        self._btn_play.clicked.connect(self.on_play)
        self._btn_pause.clicked.connect(self._player.pause)
        self._btn_stop.clicked.connect(self.on_stop)
        self._btn_export.clicked.connect(self.on_export)

        self._engine_combo.currentIndexChanged.connect(self._seed_engine)
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)
        self._gender_combo.currentIndexChanged.connect(self._update_voices)
        self._age_combo.currentIndexChanged.connect(self._update_voices)
        self._xtts_target_combo.currentIndexChanged.connect(self._on_xtts_target_changed)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

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

    def _current_speed(self) -> float:
        return round(self._speed_slider.value() / 100.0, 2)

    def _on_speed_changed(self, value: int) -> None:
        self._speed_label.setText(f"{value / 100.0:.2f}x")

    def _seed_engine(self) -> None:
        engine = self.current_engine()
        languages = engine.list_languages()
        self._block_set_items_languages(self._language_combo, languages)
        if languages:
            default = languages.index("en") if "en" in languages else 0
            self._language_combo.setCurrentIndex(default)
            self._populate_for_language(languages[default])
        self._update_voice_ui()

    def _update_voice_ui(self) -> None:
        engine = self.current_engine()
        is_catalog = engine.has_voice_catalog
        self._voice_stack.setCurrentIndex(0 if is_catalog else 1)
        if not is_catalog:
            self._refresh_xtts_targets()

    def _refresh_xtts_targets(self, select_id: _t.Optional[str] = None) -> None:
        engine = self.current_engine()
        self._xtts_target_combo.blockSignals(True)
        self._xtts_target_combo.clear()
        self._xtts_target_combo.addItem(
            "Custom Sample / Microphone Recording", userData=_SAMPLE_VOICE_OPTION
        )
        for prof in list_profiles():
            self._xtts_target_combo.addItem(
                f"Profile: {prof.name} ({prof.kind}, {prof.total_source_duration_s}s)",
                userData=Voice(
                    voice_id=f"profile:{prof.id}",
                    name=prof.name,
                    locale=prof.language or "en",
                    gender="Unknown",
                    age="Adult",
                    engine=engine.id,
                    kind=prof.kind,
                ),
            )
        if hasattr(engine, "list_builtin_speakers"):
            for spk in engine.list_builtin_speakers():
                self._xtts_target_combo.addItem(
                    f"Built-in: {spk}",
                    userData=Voice(
                        voice_id=f"builtin:{spk}",
                        name=spk,
                        locale="en",
                        gender="Unknown",
                        age="Adult",
                        engine=engine.id,
                        kind="builtin",
                    ),
                )
        if select_id:
            for idx in range(self._xtts_target_combo.count()):
                data = self._xtts_target_combo.itemData(idx)
                if isinstance(data, Voice) and data.id == select_id:
                    self._xtts_target_combo.setCurrentIndex(idx)
                    break
        self._xtts_target_combo.blockSignals(False)
        self._on_xtts_target_changed()

    def _on_xtts_target_changed(self) -> None:
        data = self._xtts_target_combo.currentData()
        using_sample = data == _SAMPLE_VOICE_OPTION
        is_profile = isinstance(data, Voice) and data.id.startswith("profile:")
        self._btn_sample.setEnabled(using_sample)
        self._btn_record.setEnabled(using_sample)
        self._btn_save_profile.setEnabled(using_sample and bool(self._sample_path))
        self._btn_delete_profile.setEnabled(is_profile)

    def _selected_language(self) -> str:
        data = self._language_combo.currentData()
        if data:
            return data
        return self._language_combo.currentText()

    def _on_language_changed(self, *_args) -> None:
        self._populate_for_language(self._selected_language())

    def _populate_for_language(self, subtag: str) -> None:
        engine = self.current_engine()
        if not engine.has_voice_catalog:
            return
        voices = engine.list_voices(subtag)
        genders = engine.genders(subtag)
        ages = sorted({v.age for v in voices})

        self._block_set_items(self._gender_combo, [_ANY, *genders])
        self._block_set_items(self._age_combo, [_ANY, *ages])
        self._update_voices()

    def _update_voices(self) -> None:
        engine = self.current_engine()
        if not engine.has_voice_catalog:
            return
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
    def _block_set_items_languages(combo: QComboBox, subtags) -> None:
        combo.blockSignals(True)
        combo.clear()
        for subtag in subtags:
            combo.addItem(language_display_name(subtag), userData=subtag)
        combo.blockSignals(False)

    def selected_voice(self) -> _t.Optional[Voice]:
        engine = self.current_engine()
        if not engine.has_voice_catalog:
            target = self._xtts_target_combo.currentData()
            if isinstance(target, Voice):
                return Voice(
                    voice_id=target.id,
                    name=target.name,
                    locale=language_subtag(self._selected_language()),
                    gender=target.gender,
                    age=target.age,
                    engine=engine.id,
                    kind=target.kind,
                )
            if not self._sample_path:
                return None
            return Voice(
                voice_id=self._sample_path,
                name=os.path.basename(self._sample_path),
                locale=language_subtag(self._selected_language()),
                gender="Unknown",
                age="Adult",
                engine=engine.id,
                kind="sample",
            )
        return self._voice_combo.currentData() if self._voice_combo.count() else None

    # ----------------------------------------------------------------- #
    # User actions
    # ----------------------------------------------------------------- #
    def on_load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a text file", "", "Text files (*.txt);;All files (*)"
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

    def on_choose_sample(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a voice sample",
            "",
            "Audio files (*.wav *.mp3 *.flac *.ogg *.m4a);;All files (*)",
        )
        if not path:
            return
        self._sample_path = path
        self._sample_label.setText(os.path.basename(path))
        self._sample_label.setStyleSheet("color: #000;")
        self._btn_save_profile.setEnabled(True)
        self._status.setText(f"Voice sample: {os.path.basename(path)}")

    def on_record(self) -> None:
        rec = self._recorder
        if rec is not None and rec.state() == QMediaRecorder.RecordingState:
            rec.stop()
            return

        path = self._new_tmp_path(".wav")
        rec = QAudioRecorder(self)
        settings = QAudioEncoderSettings()
        settings.setCodec("audio/x-raw")
        settings.setSampleRate(44100)
        settings.setChannelCount(1)
        settings.setQuality(QMultimedia.HighQuality)
        rec.setEncodingSettings(settings, container="audio/x-wav")
        rec.setOutputLocation(QUrl.fromLocalFile(path))
        rec.stateChanged.connect(self._on_record_state)
        rec.error.connect(self._on_record_error)
        self._recorder = rec
        self._record_path = path
        rec.record()
        self._btn_record.setText("Stop recording")
        self._status.setText("Recording\u2026 click Stop when done")

    def _on_record_state(self, state: int) -> None:
        if state != QMediaRecorder.StoppedState:
            return
        self._btn_record.setText("Record")
        path = self._record_path
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            self._sample_path = path
            self._sample_label.setText(f"Recorded: {os.path.basename(path)}")
            self._sample_label.setStyleSheet("color: #000;")
            self._btn_save_profile.setEnabled(True)
            self._status.setText("Voice sample recorded.")
        self._record_path = None

    def _on_record_error(self, *_args) -> None:
        self._btn_record.setText("Record")
        self.error("Could not record audio. Check your microphone and try again.")

    def on_save_profile(self) -> None:
        if not self._sample_path or not os.path.isfile(self._sample_path):
            self.error("Record or choose a voice sample first.")
            return
        name, ok = QInputDialog.getText(
            self, "Save Voice Profile", "Profile name (e.g. 'My Voice'):"
        )
        if not ok or not name.strip():
            return

        self._status.setText("Enrolling voice profile (extracting speaker latents)\u2026")
        self._set_busy(True)
        worker = _EnrollWorker(
            name.strip(), self._sample_path, language_subtag(self._selected_language())
        )
        worker.succeeded.connect(self._on_enroll_done)
        worker.failed.connect(self._on_synth_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._worker = worker

    def _on_enroll_done(self, profile) -> None:
        self._set_busy(False)
        self._refresh_xtts_targets(select_id=f"profile:{profile.id}")
        self._status.setText(f"Saved voice profile: {profile.name}")
        QMessageBox.information(
            self,
            "Voice Profile Saved",
            f"Enrolled '{profile.name}' ({profile.total_source_duration_s}s audio).\n"
            "You can now select it anytime without re-uploading your sample.",
        )

    def on_delete_profile(self) -> None:
        data = self._xtts_target_combo.currentData()
        if not isinstance(data, Voice) or not data.id.startswith("profile:"):
            return
        pid = data.id[len("profile:") :]
        reply = QMessageBox.question(
            self,
            "Delete Voice Profile",
            f"Delete saved voice profile '{data.name}'?",
        )
        if reply != QMessageBox.Yes:
            return
        try:
            delete_profile(pid)
        except TtsError as exc:
            self.error(str(exc))
            return
        self._refresh_xtts_targets()
        self._status.setText(f"Deleted profile: {data.name}")

    def on_show_training_info(self) -> None:
        hw = check_hardware_requirements()
        t_inf = hw["tiers"]["xtts_inference"]
        t_ft = hw["tiers"]["xtts_finetuning"]
        gpu_line = (
            f"{hw['gpu_name']} ({hw['vram_gb']} GB VRAM)"
            if hw["cuda_available"]
            else "None detected by PyTorch (CPU mode)"
        )
        msg = (
            f"Detected Hardware:\n"
            f"  CPU Cores: {hw['cpu_cores']} | RAM: {hw['ram_total_gb']} GB | Disk free: {hw['disk_free_gb']} GB\n"
            f"  CUDA GPU: {gpu_line}\n"
            f"  FFmpeg: {'Yes' if hw['ffmpeg_installed'] else 'No'}\n\n"
            f"1. Voice Profile Enrollment (Fast Cloning):\n"
            f"   Ready: {'Yes' if t_inf['ready'] else 'No'}\n"
            f"   Min HW: {t_inf['minimum']}\n"
            f"   Use 'Save as Profile' in the GUI or `tts-cli enroll-voice`.\n\n"
            f"2. Full Custom Voice Fine-Tuning (XTTS GPT Training):\n"
            f"   Ready: {'Yes' if t_ft['ready'] else 'No'}\n"
            f"   Min HW: {t_ft['minimum']}\n"
        )
        if t_ft["notes"]:
            msg += "   Notes: " + "; ".join(t_ft["notes"]) + "\n"
        msg += "\nRun `tts-cli train-voice` or call the MCP `train_custom_voice` tool to fine-tune."
        QMessageBox.information(self, "Voice Training & Hardware", msg)

    def _validation_error(self) -> _t.Optional[str]:
        if not self._text_edit.toPlainText().strip():
            return "The text is empty. Add some text or load a .txt file."
        if self.selected_voice() is None:
            if not self.current_engine().has_voice_catalog:
                return "Pick a voice sample, saved voice profile, or built-in XTTS speaker."
            return "No voice is available for this selection."
        return None

    def _current_cache_key(self, voice: Voice) -> tuple[str, str, str, str, float]:
        return (
            self.current_engine().id,
            self._text_edit.toPlainText(),
            voice.id,
            self._selected_language(),
            self._current_speed(),
        )

    def on_play(self) -> None:
        problem = self._validation_error()
        if problem:
            self.error(problem)
            return

        text = self._text_edit.toPlainText()
        voice = self.selected_voice()
        assert voice is not None
        key = self._current_cache_key(voice)

        if (
            key == self._loaded_key
            and self._playable_path
            and os.path.exists(self._playable_path)
        ):
            self._start_playback(self._playable_path)
            return

        engine = self.current_engine()
        ext = ".mp3" if engine.id == "edge-tts" else ".wav"
        output = self._new_tmp_path(ext)
        if engine.has_voice_catalog:
            self._status.setText("Synthesising\u2026")
        else:
            self._status.setText(
                "Synthesising locally with XTTS v2\u2026 (first run loads ~1.8 GB model)"
            )
        self._set_busy(True)
        self._start_worker(
            engine,
            text,
            voice,
            output,
            on_done=lambda res: self._on_synth_done(key, res),
            on_failed=self._on_synth_failed,
            generate_subtitles=False,
        )

    def on_stop(self) -> None:
        self._player.stop()
        self._status.setText("Ready.")

    def on_export(self) -> None:
        problem = self._validation_error()
        if problem:
            self.error(problem)
            return

        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export audio",
            "speech.mp3",
            "MP3 audio (*.mp3);;WAV audio (*.wav)",
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".mp3" if "MP3" in selected else ".wav"

        voice = self.selected_voice()
        assert voice is not None
        self._status.setText("Exporting\u2026")
        self._set_busy(True)
        self._start_worker(
            self.current_engine(),
            self._text_edit.toPlainText(),
            voice,
            path,
            on_done=self._on_export_done,
            on_failed=self._on_export_failed,
            generate_subtitles=self._chk_subtitles.isChecked(),
        )

    # ----------------------------------------------------------------- #
    # Background worker plumbing
    # ----------------------------------------------------------------- #
    def _start_worker(
        self,
        engine: BaseTtsEngine,
        text: str,
        voice: Voice,
        output: str,
        *,
        on_done,
        on_failed,
        generate_subtitles: bool,
    ) -> None:
        worker = _SynthesisWorker(
            engine,
            text,
            voice,
            output,
            language=self._selected_language(),
            speed=self._current_speed(),
            generate_subtitles=generate_subtitles,
            normalize_loudness=self._chk_normalize.isChecked(),
        )
        worker.succeeded.connect(on_done)
        worker.failed.connect(on_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._worker = worker

    def _on_synth_done(self, key, result: SynthesisResult) -> None:
        self._set_busy(False)
        self._loaded_key = key
        self._playable_path = result.audio_path
        self._start_playback(result.audio_path)

    def _on_synth_failed(self, message: str) -> None:
        self._set_busy(False)
        self.error(message)

    def _on_export_done(self, result: SynthesisResult) -> None:
        self._set_busy(False)
        self._status.setText(f"Exported to {os.path.basename(result.audio_path)}")
        msg = f"Saved audio ({result.duration_s}s):\n{result.audio_path}"
        if result.srt_path:
            msg += f"\n\nSubtitles:\n{result.srt_path}\n{result.vtt_path}\n{result.timings_path}"
        QMessageBox.information(self, "Export complete", msg)

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
        self._btn_sample.setEnabled(not busy)
        self._btn_record.setEnabled(not busy)
        if not busy:
            self._on_xtts_target_changed()
        else:
            self._btn_save_profile.setEnabled(False)

    def _new_tmp_path(self, extension: str) -> str:
        fd, path = tempfile.mkstemp(suffix=extension, prefix="tts_")
        os.close(fd)
        self._temp_files.append(path)
        return path

    def error(self, message: str) -> None:
        self._status.setText("Error")
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        self._player.stop()
        for path in self._temp_files:
            safe_remove(path)
        self._temp_files.clear()
        super().closeEvent(event)


def _fmt(ms: int) -> str:
    ms = max(0, int(ms))
    seconds, _ = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def main() -> int:
    """Create the QApplication, show the window, and run the event loop."""
    app = QApplication(sys.argv)
    app.setApplicationName("Text-to-Speech")
    try:
        window = MainWindow()
    except TtsError as exc:
        QMessageBox.critical(None, "Startup error", str(exc))
        return 1
    window.show()
    return app.exec_()
