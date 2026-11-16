"""Modern PyQt5 studio GUI for multi-engine TTS, video voiceovers, and voice profiles."""

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
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tts_app.audio_player import PAUSED, PLAYING, AudioPlayer
from tts_app.audio_utils import safe_remove
from tts_app.models import SynthesisResult, TtsError, Voice
from tts_app.theme import build_stylesheet
from tts_app.tts_engine import (
    BaseTtsEngine,
    available_engines,
    language_display_name,
    language_subtag,
)
from tts_app.video_pipeline import generate_video_voiceover
from tts_app.voice_profiles import (
    check_hardware_requirements,
    delete_profile,
    enroll_voice_profile,
    get_outputs_dir,
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
        trim_silence: bool,
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
        self._trim_silence = trim_silence

    def run(self) -> None:
        try:
            kwargs: dict[str, _t.Any] = {
                "speed": self._speed,
                "generate_subtitles": self._generate_subtitles,
                "normalize": self._normalize_loudness,
                "trim": self._trim_silence,
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

    def __init__(
        self, name: str, sample_paths: list[str], language: str
    ) -> None:
        super().__init__()
        self._name = name
        self._sample_paths = sample_paths
        self._language = language

    def run(self) -> None:
        try:
            prof = enroll_voice_profile(
                name=self._name,
                audio_paths=self._sample_paths,
                language=self._language,
                preprocess=True,
            )
        except TtsError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Voice enrollment failed: {exc}")
        else:
            self.succeeded.emit(prof)


class _VideoVoiceoverWorker(QThread):
    """Run multi-scene video narration rendering on a background thread."""

    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, kwargs: dict[str, _t.Any]) -> None:
        super().__init__()
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            res = generate_video_voiceover(**self._kwargs)
        except TtsError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Video voiceover failed: {exc}")
        else:
            self.succeeded.emit(res)


def _make_card() -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("Card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)
    return card, layout


class MainWindow(QMainWindow):
    """Modern desktop studio window for Text-to-Speech and Video Narration."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Voice Studio — TTS, Video Narration & Cloning")
        self.resize(1040, 760)
        self.setMinimumSize(860, 640)

        self._dark_mode = True
        self._engines = available_engines()
        self._player = AudioPlayer()
        self._worker: _t.Optional[QThread] = None

        #: key (engine_id, text, voice_id, language, speed) for cached playback.
        self._loaded_key: _t.Optional[tuple[str, str, str, str, float]] = None
        self._playable_path: _t.Optional[str] = None
        self._video_master_path: _t.Optional[str] = None
        self._temp_files: list[str] = []
        self._scrubbing = False

        #: path of the user's reference audio for voice-cloning engines.
        self._sample_path: _t.Optional[str] = None
        self._recorder: _t.Optional[QAudioRecorder] = None
        self._record_path: _t.Optional[str] = None

        self._hw_snapshot = check_hardware_requirements()

        self._build_ui()
        self._wire_signals()
        self._apply_theme()
        self._seed_engine()
        self._refresh_profiles_tab()

    # ----------------------------------------------------------------- #
    # UI construction
    # ----------------------------------------------------------------- #
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("AppRoot")
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(root)

        content_wrap = QWidget()
        content = QVBoxLayout(content_wrap)
        content.setContentsMargins(18, 14, 18, 12)
        content.setSpacing(12)

        # -- Header Bar ----------------------------------------------------
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("Voice Studio")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Smart Multi-Engine TTS · Video Narration & Subtitles · Voice Cloning"
        )
        subtitle.setObjectName("SubtitleLabel")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, 1)

        gpu_text = (
            f"GPU: {self._hw_snapshot['gpu_name']}"
            if self._hw_snapshot["cuda_available"]
            else f"CPU ({self._hw_snapshot['cpu_cores']} cores)"
        )
        gpu_pill = QLabel(gpu_text)
        gpu_pill.setObjectName("StatusBadge")
        ffmpeg_pill = QLabel(
            "FFmpeg Ready" if self._hw_snapshot["ffmpeg_installed"] else "FFmpeg Missing"
        )
        ffmpeg_pill.setObjectName("StatusBadge")
        self._btn_hw_info = QPushButton("Hardware / Training")
        self._btn_theme = QPushButton("Light Mode")

        header.addWidget(gpu_pill)
        header.addWidget(ffmpeg_pill)
        header.addWidget(self._btn_hw_info)
        header.addWidget(self._btn_theme)
        content.addLayout(header)

        # -- Tab Workspace -------------------------------------------------
        self._tabs = QTabWidget()
        tab_font = self._tabs.tabBar().font()
        tab_font.setWeight(QFont.DemiBold)
        self._tabs.tabBar().setFont(tab_font)
        self._tabs.tabBar().setExpanding(False)
        self._tabs.tabBar().setUsesScrollButtons(False)
        self._tabs.addTab(self._build_studio_tab(), "Speech Studio")
        self._tabs.addTab(self._build_video_tab(), "Video Voiceover")
        self._tabs.addTab(self._build_profiles_tab(), "Voice Profiles")
        content.addWidget(self._tabs, 1)

        main_layout.addWidget(content_wrap, 1)

        # -- Docked Transport Bar ------------------------------------------
        transport = QFrame()
        transport.setObjectName("TransportBar")
        t_outer = QVBoxLayout(transport)
        t_outer.setContentsMargins(18, 8, 18, 12)
        t_outer.setSpacing(8)

        self._busy_bar = QProgressBar()
        self._busy_bar.setRange(0, 1)
        self._busy_bar.setValue(0)
        self._busy_bar.setTextVisible(False)
        self._busy_bar.setFixedHeight(4)
        t_outer.addWidget(self._busy_bar)

        seek_row = QHBoxLayout()
        self._progress = QSlider(Qt.Horizontal)
        self._progress.setEnabled(False)
        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setObjectName("MutedLabel")
        self._time_label.setMinimumWidth(95)
        self._time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        seek_row.addWidget(self._progress, 1)
        seek_row.addWidget(self._time_label)
        t_outer.addLayout(seek_row)

        btn_row = QHBoxLayout()
        self._btn_play = QPushButton("Play")
        self._btn_play.setObjectName("PrimaryButton")
        self._btn_pause = QPushButton("Pause")
        self._btn_stop = QPushButton("Stop")
        self._btn_export = QPushButton("Export Audio\u2026")
        for b in (self._btn_play, self._btn_pause, self._btn_stop, self._btn_export):
            btn_row.addWidget(b)
        btn_row.addSpacing(12)
        self._status = QLabel("Ready.")
        self._status.setObjectName("MutedLabel")
        btn_row.addWidget(self._status, 1)
        t_outer.addLayout(btn_row)

        main_layout.addWidget(transport)

    def _build_studio_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)

        # Left Card: Script editor
        script_card, script_l = _make_card()
        script_head = QHBoxLayout()
        lbl_script = QLabel("Script / Text")
        lbl_script.setObjectName("SectionTitle")
        self._word_stats = QLabel("0 words · ~0s")
        self._word_stats.setObjectName("MutedLabel")
        script_head.addWidget(lbl_script)
        script_head.addStretch(1)
        script_head.addWidget(self._word_stats)
        script_l.addLayout(script_head)

        tool_row = QHBoxLayout()
        self._btn_load = QPushButton("Load .txt\u2026")
        self._btn_insert_scene = QPushButton("+ Scene Tag")
        self._btn_clear_text = QPushButton("Clear")
        self._file_label = QLabel("No file loaded")
        self._file_label.setObjectName("MutedLabel")
        tool_row.addWidget(self._btn_load)
        tool_row.addWidget(self._btn_insert_scene)
        tool_row.addWidget(self._btn_clear_text)
        tool_row.addWidget(self._file_label, 1)
        script_l.addLayout(tool_row)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setPlaceholderText(
            "Type or paste narration text here\u2026\n\n"
            "Tip: Use [scene: intro] markers in the Video Voiceover tab to build timed scenes."
        )
        self._text_edit.setFont(QFont("Sans Serif", 11))
        script_l.addWidget(self._text_edit, 1)

        # Right Card: Voice & Prosody settings
        ctrl_card, ctrl_l = _make_card()
        lbl_ctrl = QLabel("Voice & Synthesis Controls")
        lbl_ctrl.setObjectName("SectionTitle")
        ctrl_l.addWidget(lbl_ctrl)

        form = QFormLayout()
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._engine_combo = QComboBox()
        self._engine_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for engine in self._engines:
            self._engine_combo.addItem(engine.label)
        form.addRow("Engine:", self._engine_combo)

        self._language_combo = QComboBox()
        self._language_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addRow("Language:", self._language_combo)
        ctrl_l.addLayout(form)

        # Voice stack (Catalog vs XTTS cloning/profiles)
        self._voice_stack = QStackedWidget()

        catalog_page = QWidget()
        cv = QVBoxLayout(catalog_page)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(8)

        self._voice_search = QLineEdit()
        self._voice_search.setPlaceholderText("Search voices by name or locale (e.g. Andrew, UK)\u2026")
        self._voice_search.setClearButtonEnabled(True)
        cv.addWidget(self._voice_search)

        self._voice_combo = QComboBox()
        cv.addWidget(self._voice_combo)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Gender:"))
        self._gender_combo = QComboBox()
        filters.addWidget(self._gender_combo, 1)
        filters.addSpacing(8)
        filters.addWidget(QLabel("Age:"))
        self._age_combo = QComboBox()
        filters.addWidget(self._age_combo, 1)
        cv.addLayout(filters)
        self._voice_stack.addWidget(catalog_page)

        xtts_page = QWidget()
        sv = QVBoxLayout(xtts_page)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(8)

        self._xtts_target_combo = QComboBox()
        sv.addWidget(self._xtts_target_combo)

        sample_btns = QHBoxLayout()
        self._btn_sample = QPushButton("Choose Sample\u2026")
        self._btn_record = QPushButton("Record Mic")
        sample_btns.addWidget(self._btn_sample, 1)
        sample_btns.addWidget(self._btn_record, 1)
        sv.addLayout(sample_btns)

        prof_btns = QHBoxLayout()
        self._btn_save_profile = QPushButton("Save as Profile\u2026")
        self._btn_delete_profile = QPushButton("Delete Profile")
        self._btn_delete_profile.setObjectName("DangerButton")
        prof_btns.addWidget(self._btn_save_profile, 1)
        prof_btns.addWidget(self._btn_delete_profile, 1)
        sv.addLayout(prof_btns)

        self._sample_label = QLabel("No custom sample selected")
        self._sample_label.setObjectName("MutedLabel")
        self._sample_label.setWordWrap(True)
        sv.addWidget(self._sample_label)

        self._sample_hint = QLabel(
            "Select a saved voice profile, a built-in XTTS speaker, or 6-30s of clean speech."
        )
        self._sample_hint.setObjectName("MutedLabel")
        self._sample_hint.setWordWrap(True)
        sv.addWidget(self._sample_hint)
        self._voice_stack.addWidget(xtts_page)

        ctrl_l.addWidget(self._voice_stack)

        # Speed slider
        speed_row = QHBoxLayout()
        speed_lbl = QLabel("Speed:")
        self._speed_slider = QSlider(Qt.Horizontal)
        self._speed_slider.setMinimum(75)
        self._speed_slider.setMaximum(150)
        self._speed_slider.setValue(100)
        self._speed_slider.setSingleStep(5)
        self._speed_label = QLabel("1.00x")
        self._speed_label.setMinimumWidth(45)
        self._btn_reset_speed = QPushButton("Reset")
        speed_row.addWidget(speed_lbl)
        speed_row.addWidget(self._speed_slider, 1)
        speed_row.addWidget(self._speed_label)
        speed_row.addWidget(self._btn_reset_speed)
        ctrl_l.addLayout(speed_row)

        # Post-processing & video options
        self._chk_normalize = QCheckBox("Normalize loudness (-16 LUFS)")
        self._chk_trim = QCheckBox("Trim leading and trailing silence")
        self._chk_subtitles = QCheckBox("Export captions (.srt, .vtt, .timings.json)")
        ctrl_l.addWidget(self._chk_normalize)
        ctrl_l.addWidget(self._chk_trim)
        ctrl_l.addWidget(self._chk_subtitles)
        ctrl_l.addStretch(1)

        splitter.addWidget(script_card)
        splitter.addWidget(ctrl_card)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)
        return page

    def _build_video_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(12)

        # Left: Multi-scene script editor
        left_card, left_l = _make_card()
        head = QHBoxLayout()
        lbl = QLabel("Multi-Scene Storyboard Script")
        lbl.setObjectName("SectionTitle")
        self._btn_copy_from_studio = QPushButton("Copy from Speech Studio")
        head.addWidget(lbl)
        head.addStretch(1)
        head.addWidget(self._btn_copy_from_studio)
        left_l.addLayout(head)

        self._video_script_edit = QPlainTextEdit()
        self._video_script_edit.setPlaceholderText(
            "[scene: hook]\nStop editing video captions by hand.\n\n"
            "[scene: demo]\nGenerate synced voiceover, SRT, and frame timings in one click."
        )
        self._video_script_edit.setFont(QFont("Sans Serif", 11))
        left_l.addWidget(self._video_script_edit, 1)

        # Right: Render settings & outputs
        right_card, right_l = _make_card()
        rlbl = QLabel("Render Settings & Output")
        rlbl.setObjectName("SectionTitle")
        right_l.addWidget(rlbl)

        form = QFormLayout()
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._video_name_edit = QLineEdit("voiceover")
        form.addRow("Output Name:", self._video_name_edit)

        out_row = QHBoxLayout()
        self._video_dir_edit = QLineEdit(os.path.join(get_outputs_dir(), "studio_video"))
        self._btn_video_dir = QPushButton("Browse\u2026")
        out_row.addWidget(self._video_dir_edit, 1)
        out_row.addWidget(self._btn_video_dir)
        form.addRow("Output Folder:", out_row)

        self._video_fps_combo = QComboBox()
        self._video_fps_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for fps_val in (24, 25, 30, 60):
            self._video_fps_combo.addItem(f"{fps_val} fps", userData=fps_val)
        self._video_fps_combo.setCurrentIndex(2)  # 30 fps
        form.addRow("Frame Rate:", self._video_fps_combo)

        self._video_pause_spin = QSpinBox()
        self._video_pause_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._video_pause_spin.setRange(0, 5000)
        self._video_pause_spin.setSingleStep(50)
        self._video_pause_spin.setValue(350)
        self._video_pause_spin.setSuffix(" ms")
        form.addRow("Scene Pause:", self._video_pause_spin)

        self._video_fmt_combo = QComboBox()
        self._video_fmt_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._video_fmt_combo.addItems(["mp3", "wav"])
        form.addRow("Format:", self._video_fmt_combo)

        mp4_row = QHBoxLayout()
        self._video_mp4_edit = QLineEdit()
        self._video_mp4_edit.setPlaceholderText("Optional MP4/MOV to mux onto\u2026")
        self._btn_pick_mp4 = QPushButton("Pick Video\u2026")
        mp4_row.addWidget(self._video_mp4_edit, 1)
        mp4_row.addWidget(self._btn_pick_mp4)
        form.addRow("Target Video:", mp4_row)
        right_l.addLayout(form)

        self._chk_burn_subs = QCheckBox("Burn subtitles onto video (re-encodes video)")
        right_l.addWidget(self._chk_burn_subs)

        action_row = QHBoxLayout()
        self._btn_render_video = QPushButton("Render Voiceover + Timeline")
        self._btn_render_video.setObjectName("PrimaryButton")
        self._btn_play_video_master = QPushButton("Play Master")
        self._btn_play_video_master.setEnabled(False)
        action_row.addWidget(self._btn_render_video, 1)
        action_row.addWidget(self._btn_play_video_master)
        right_l.addLayout(action_row)

        self._video_result_view = QPlainTextEdit()
        self._video_result_view.setReadOnly(True)
        self._video_result_view.setPlaceholderText(
            "Render summary, generated .srt/.vtt/.timings.json paths, and scene frame offsets appear here."
        )
        right_l.addWidget(self._video_result_view, 1)

        layout.addWidget(left_card, 3)
        layout.addWidget(right_card, 2)
        return page

    def _build_profiles_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(12)

        # Left: Saved Voice Profiles
        left_card, left_l = _make_card()
        lbl = QLabel("Saved Custom Voice Profiles")
        lbl.setObjectName("SectionTitle")
        left_l.addWidget(lbl)

        self._profiles_list = QListWidget()
        left_l.addWidget(self._profiles_list, 1)

        btn_row = QHBoxLayout()
        self._btn_use_profile = QPushButton("Use in Studio")
        self._btn_use_profile.setObjectName("PrimaryButton")
        self._btn_enroll_files = QPushButton("Enroll Audio Clips\u2026")
        self._btn_refresh_profiles = QPushButton("Refresh")
        self._btn_remove_profile_tab = QPushButton("Delete")
        self._btn_remove_profile_tab.setObjectName("DangerButton")
        for b in (
            self._btn_use_profile,
            self._btn_enroll_files,
            self._btn_refresh_profiles,
            self._btn_remove_profile_tab,
        ):
            btn_row.addWidget(b)
        left_l.addLayout(btn_row)

        # Right: Hardware Tiers & Training Guide
        right_card, right_l = _make_card()
        hlbl = QLabel("Hardware Readiness & Custom Voice Training")
        hlbl.setObjectName("SectionTitle")
        right_l.addWidget(hlbl)

        self._hw_summary_view = QPlainTextEdit()
        self._hw_summary_view.setReadOnly(True)
        right_l.addWidget(self._hw_summary_view, 1)
        self._populate_hardware_summary()

        layout.addWidget(left_card, 3)
        layout.addWidget(right_card, 2)
        return page

    def _populate_hardware_summary(self) -> None:
        hw = self._hw_snapshot
        gpu_desc = (
            f"{hw['gpu_name']} ({hw['vram_gb']} GB VRAM)"
            if hw["cuda_available"]
            else "None detected by PyTorch (CPU mode)"
        )
        lines = [
            f"CPU Cores: {hw['cpu_cores']}  |  RAM: {hw['ram_total_gb']} GB  |  Free Disk: {hw['disk_free_gb']} GB",
            f"CUDA GPU: {gpu_desc}",
            f"FFmpeg: {'Installed' if hw['ffmpeg_installed'] else 'Missing'}  |  eSpeak: {'Installed' if hw['espeak_installed'] else 'Missing'}",
            "",
        ]
        for tier_key, title in (
            ("standard_tts", "1. Standard TTS & Video Narration"),
            ("xtts_inference", "2. Local XTTS Cloning & Enrolled Profiles"),
            ("xtts_finetuning", "3. Full Custom Voice Training (Fine-Tuning)"),
        ):
            tier = hw["tiers"][tier_key]
            badge = "READY" if tier["ready"] else "NOT READY"
            lines.append(f"[{badge}] {title}")
            lines.append(f"  Min HW: {tier['minimum']}")
            for note in tier["notes"]:
                lines.append(f"  Note: {note}")
            lines.append("")
        lines.append(
            "Enroll fast reusable profiles via 'Enroll Audio Clips...' or fine-tune XTTS v2 "
            "using `tts-cli train-voice` / MCP tool `train_custom_voice`."
        )
        self._hw_summary_view.setPlainText("\n".join(lines))

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(dark=self._dark_mode))
        self._btn_theme.setText("Light Mode" if self._dark_mode else "Dark Mode")

    def on_toggle_theme(self) -> None:
        self._dark_mode = not self._dark_mode
        self._apply_theme()

    def _wire_signals(self) -> None:
        # Header & Studio text
        self._btn_theme.clicked.connect(self.on_toggle_theme)
        self._btn_hw_info.clicked.connect(self.on_show_training_info)
        self._btn_load.clicked.connect(self.on_load_file)
        self._btn_insert_scene.clicked.connect(self._on_insert_scene_tag)
        self._btn_clear_text.clicked.connect(self._text_edit.clear)
        self._text_edit.textChanged.connect(self._update_word_stats)

        # Voice controls
        self._btn_sample.clicked.connect(self.on_choose_sample)
        self._btn_record.clicked.connect(self.on_record)
        self._btn_save_profile.clicked.connect(self.on_save_profile)
        self._btn_delete_profile.clicked.connect(self.on_delete_profile)
        self._btn_reset_speed.clicked.connect(lambda: self._speed_slider.setValue(100))

        self._engine_combo.currentIndexChanged.connect(self._seed_engine)
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)
        self._gender_combo.currentIndexChanged.connect(self._update_voices)
        self._age_combo.currentIndexChanged.connect(self._update_voices)
        self._voice_search.textChanged.connect(self._update_voices)
        self._xtts_target_combo.currentIndexChanged.connect(self._on_xtts_target_changed)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

        # Video tab
        self._btn_copy_from_studio.clicked.connect(self._on_copy_to_video_tab)
        self._btn_video_dir.clicked.connect(self._on_pick_video_output_dir)
        self._btn_pick_mp4.clicked.connect(self._on_pick_target_mp4)
        self._btn_render_video.clicked.connect(self.on_render_video_voiceover)
        self._btn_play_video_master.clicked.connect(self._on_play_video_master)

        # Profiles tab
        self._btn_use_profile.clicked.connect(self._on_use_selected_profile)
        self._btn_enroll_files.clicked.connect(self._on_enroll_multiple_files)
        self._btn_refresh_profiles.clicked.connect(self._refresh_profiles_tab)
        self._btn_remove_profile_tab.clicked.connect(self._on_delete_selected_profile_tab)

        # Transport & player
        self._btn_play.clicked.connect(self.on_play)
        self._btn_pause.clicked.connect(self._player.pause)
        self._btn_stop.clicked.connect(self.on_stop)
        self._btn_export.clicked.connect(self.on_export)

        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.stateChanged.connect(self._on_player_state)
        self._player.finished.connect(self._on_finished)
        self._player.errorOccurred.connect(self._on_player_error)

        self._progress.sliderPressed.connect(self._start_scrub)
        self._progress.sliderReleased.connect(self._stop_scrub)

    # ----------------------------------------------------------------- #
    # Dynamic dropdown population & script helpers
    # ----------------------------------------------------------------- #
    def _update_word_stats(self) -> None:
        text = self._text_edit.toPlainText().strip()
        words = len(text.split()) if text else 0
        est_s = round(words / 2.5 / max(0.5, self._current_speed())) if words else 0
        self._word_stats.setText(f"{words} words · {len(text)} chars · ~{est_s}s")

    def _on_insert_scene_tag(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Insert Scene Marker", "Scene id (e.g. intro, demo, outro):"
        )
        if not ok or not name.strip():
            return
        cursor = self._text_edit.textCursor()
        prefix = "" if not self._text_edit.toPlainText().strip() else "\n\n"
        cursor.insertText(f"{prefix}[scene: {name.strip()}]\n")

    def current_engine(self) -> BaseTtsEngine:
        return self._engines[self._engine_combo.currentIndex()]

    def _current_speed(self) -> float:
        return round(self._speed_slider.value() / 100.0, 2)

    def _on_speed_changed(self, value: int) -> None:
        self._speed_label.setText(f"{value / 100.0:.2f}x")
        self._update_word_stats()

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

    def _refresh_profiles_tab(self) -> None:
        self._profiles_list.clear()
        for prof in list_profiles():
            label = (
                f"{prof.name}  [{prof.kind}]  ·  lang={prof.language}  ·  "
                f"{prof.sample_count} clip(s) ({prof.total_source_duration_s}s)"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, prof)
            self._profiles_list.addItem(item)

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
        query = self._voice_search.text().strip().lower()

        voices = engine.list_voices(language)
        if gender != _ANY:
            voices = [v for v in voices if v.gender == gender]
        if age != _ANY:
            voices = [v for v in voices if v.age == age]
        if query:
            voices = [
                v
                for v in voices
                if query in v.name.lower()
                or query in v.id.lower()
                or query in v.locale.lower()
            ]
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
    # User actions: files, recording, profiles
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
        self._file_label.setText(os.path.basename(path))
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
        self._sample_label.setText(f"Sample: {os.path.basename(path)}")
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
        self._btn_record.setText("Stop Recording")
        self._status.setText("Recording microphone\u2026 click Stop when done.")

    def _on_record_state(self, state: int) -> None:
        if state != QMediaRecorder.StoppedState:
            return
        self._btn_record.setText("Record Mic")
        path = self._record_path
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            self._sample_path = path
            self._sample_label.setText(f"Recorded: {os.path.basename(path)}")
            self._btn_save_profile.setEnabled(True)
            self._status.setText("Voice sample recorded.")
        self._record_path = None

    def _on_record_error(self, *_args) -> None:
        self._btn_record.setText("Record Mic")
        self.error("Could not record audio. Check your microphone and try again.")

    def on_save_profile(self) -> None:
        if not self._sample_path or not os.path.isfile(self._sample_path):
            self.error("Record or choose a voice sample first.")
            return
        self._enroll_paths_interactive([self._sample_path])

    def _on_enroll_multiple_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select 1 or more clean voice clips",
            "",
            "Audio files (*.wav *.mp3 *.flac *.ogg *.m4a);;All files (*)",
        )
        if not paths:
            return
        self._enroll_paths_interactive(paths)

    def _enroll_paths_interactive(self, paths: list[str]) -> None:
        name, ok = QInputDialog.getText(
            self, "Save Voice Profile", "Profile name (e.g. 'My Voice'):"
        )
        if not ok or not name.strip():
            return

        self._status.setText("Enrolling voice profile (extracting speaker latents)\u2026")
        self._set_busy(True)
        worker = _EnrollWorker(
            name.strip(), paths, language_subtag(self._selected_language())
        )
        worker.succeeded.connect(self._on_enroll_done)
        worker.failed.connect(self._on_synth_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._worker = worker

    def _on_enroll_done(self, profile) -> None:
        self._set_busy(False)
        self._refresh_xtts_targets(select_id=f"profile:{profile.id}")
        self._refresh_profiles_tab()
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
        self._delete_profile_by_id(data.id[len("profile:") :], data.name)

    def _on_delete_selected_profile_tab(self) -> None:
        item = self._profiles_list.currentItem()
        if item is None:
            return
        prof = item.data(Qt.UserRole)
        self._delete_profile_by_id(prof.id, prof.name)

    def _delete_profile_by_id(self, profile_id: str, display_name: str) -> None:
        reply = QMessageBox.question(
            self,
            "Delete Voice Profile",
            f"Delete saved voice profile '{display_name}'?",
        )
        if reply != QMessageBox.Yes:
            return
        try:
            delete_profile(profile_id)
        except TtsError as exc:
            self.error(str(exc))
            return
        if not self.current_engine().has_voice_catalog:
            self._refresh_xtts_targets()
        self._refresh_profiles_tab()
        self._status.setText(f"Deleted profile: {display_name}")

    def _on_use_selected_profile(self) -> None:
        item = self._profiles_list.currentItem()
        if item is None:
            self.error("Select a voice profile first.")
            return
        prof = item.data(Qt.UserRole)
        for idx, eng in enumerate(self._engines):
            if eng.id == "xtts_v2":
                self._engine_combo.setCurrentIndex(idx)
                self._refresh_xtts_targets(select_id=f"profile:{prof.id}")
                self._tabs.setCurrentIndex(0)
                self._status.setText(f"Selected voice profile: {prof.name}")
                return
        self.error("XTTS v2 engine is not installed.")

    def on_show_training_info(self) -> None:
        self._tabs.setCurrentIndex(2)

    # ----------------------------------------------------------------- #
    # Video voiceover studio tab actions
    # ----------------------------------------------------------------- #
    def _on_copy_to_video_tab(self) -> None:
        self._video_script_edit.setPlainText(self._text_edit.toPlainText())

    def _on_pick_video_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder", self._video_dir_edit.text()
        )
        if folder:
            self._video_dir_edit.setText(folder)

    def _on_pick_target_mp4(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select target video", "", "Video files (*.mp4 *.mov *.webm);;All files (*)"
        )
        if path:
            self._video_mp4_edit.setText(path)

    def on_render_video_voiceover(self) -> None:
        script = self._video_script_edit.toPlainText().strip()
        if not script:
            self.error("Enter a storyboard script (or copy it from Speech Studio).")
            return

        voice = self.selected_voice()
        engine = self.current_engine()
        kwargs: dict[str, _t.Any] = {
            "script": script,
            "output_dir": self._video_dir_edit.text().strip() or get_outputs_dir(),
            "output_name": self._video_name_edit.text().strip() or "voiceover",
            "audio_format": self._video_fmt_combo.currentText(),
            "fps": int(self._video_fps_combo.currentData() or 30),
            "default_engine": engine.id,
            "default_language": self._selected_language(),
            "default_speed": self._current_speed(),
            "default_pause_ms": int(self._video_pause_spin.value()),
            "trim_silence_flag": self._chk_trim.isChecked(),
            "normalize_loudness_flag": self._chk_normalize.isChecked(),
            "video_path": self._video_mp4_edit.text().strip() or None,
            "burn_subtitles": self._chk_burn_subs.isChecked(),
        }
        if voice is not None:
            if voice.id.startswith("profile:"):
                kwargs["default_voice_profile"] = voice.id[len("profile:") :]
            elif voice.kind == "sample":
                kwargs["default_reference_audio"] = voice.id
            else:
                kwargs["default_voice_id"] = voice.id

        self._status.setText("Rendering multi-scene video voiceover\u2026")
        self._set_busy(True)
        worker = _VideoVoiceoverWorker(kwargs)
        worker.succeeded.connect(self._on_video_done)
        worker.failed.connect(self._on_synth_failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._worker = worker

    def _on_video_done(self, data: dict) -> None:
        self._set_busy(False)
        self._video_master_path = data.get("master_audio_path")
        self._btn_play_video_master.setEnabled(bool(self._video_master_path))
        lines = [
            f"Master Audio: {data['master_audio_path']} ({data['total_duration_s']}s / {data['total_frames']} frames @ {data['fps']} fps)",
            f"SRT Captions: {data['srt_path']}",
            f"VTT Captions: {data['vtt_path']}",
            f"Timeline JSON: {data['timeline_json_path']}",
        ]
        if data.get("muxed_video_path"):
            lines.append(f"Muxed Video: {data['muxed_video_path']}")
        lines.append("\nScenes:")
        for sc in data.get("scenes", []):
            lines.append(
                f"  - [{sc['id']}] {sc['start_ms']}..{sc['end_ms']} ms "
                f"(frames {sc['start_frame']}..{sc['end_frame']})"
            )
        self._video_result_view.setPlainText("\n".join(lines))
        self._status.setText("Video voiceover rendered.")

    def _on_play_video_master(self) -> None:
        if self._video_master_path and os.path.isfile(self._video_master_path):
            self._start_playback(self._video_master_path)

    # ----------------------------------------------------------------- #
    # Speech studio synthesis & playback
    # ----------------------------------------------------------------- #
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
            trim_silence=self._chk_trim.isChecked(),
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
    # Player signals & scrubbing
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

    def _start_scrub(self) -> None:
        self._scrubbing = True

    def _stop_scrub(self) -> None:
        self._scrubbing = False
        self._player.set_position(self._progress.value())

    # ----------------------------------------------------------------- #
    # Misc helpers
    # ----------------------------------------------------------------- #
    def _set_busy(self, busy: bool) -> None:
        self._busy_bar.setRange(0, 0 if busy else 1)
        self._busy_bar.setValue(0)
        self._btn_play.setEnabled(not busy)
        self._btn_export.setEnabled(not busy)
        self._btn_render_video.setEnabled(not busy)
        self._btn_sample.setEnabled(not busy)
        self._btn_record.setEnabled(not busy)
        if not busy:
            self._on_xtts_target_changed()
        else:
            self._btn_save_profile.setEnabled(False)
            self._btn_delete_profile.setEnabled(False)

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
    app.setStyle("Fusion")
    app.setApplicationName("Voice Studio")
    try:
        window = MainWindow()
    except TtsError as exc:
        QMessageBox.critical(None, "Startup error", str(exc))
        return 1
    window.show()
    return app.exec_()
